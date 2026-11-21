"""Download excluded-fold weights and score supplemental training text locally.

No observation text or labels are sent to the model host. Only checkpoint file
paths are requested from the existing private training volume.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from network import load, encode, predict


def run(args):
    import modal
    import torch
    from transformers.utils import logging
    logging.set_verbosity_error()
    torch.set_num_threads(4)
    source = args.inputs
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    allowed = {'id', 'kind', 'path', 'fold', 'left', 'right', 'numeric', 'input_sha256'}
    if not all(set(r) == allowed for r in rows):
        raise ValueError('Unexpected input field')
    for row in rows:
        fields = {k: row[k] for k in ('left', 'right', 'numeric')}
        assert hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest() == row['input_sha256']
        assert int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16) % 3 == row['fold']
    volume = modal.Volume.from_name('srr-semantic-training-20261120')
    args.weights.mkdir(parents=True, exist_ok=True)
    outputs = [np.zeros((len(rows), width), dtype=np.float32) for width in (4, 8, 8)]
    tokens = np.zeros(len(rows), dtype=np.int32)
    receipts = {}

    def download(remote, target, expected=None):
        if target.exists() and (expected is None or hashlib.sha256(target.read_bytes()).hexdigest() == expected):
            return
        temporary = target.with_suffix(target.suffix + '.partial')
        with temporary.open('wb') as stream:
            for chunk in volume.read_file(remote):
                stream.write(chunk)
        if expected and hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError('Fold weight digest differs from the remote recipe')
        temporary.replace(target)

    for fold in (0, 1, 2):
        directory = args.weights / str(fold)
        directory.mkdir(exist_ok=True)
        remote = args.remote.rstrip('/') + '/' + str(fold)
        download(remote + '/recipe.json', directory / 'recipe.json')
        recipe = json.loads((directory / 'recipe.json').read_bytes())
        assert recipe['fold'] == fold and recipe['folds'] == 3
        for name, expected in recipe['weights'].items():
            download(remote + '/' + name, directory / name, expected)
        for entry in volume.listdir(remote):
            name = Path(entry.path).name
            if name not in recipe['weights'] and name != 'recipe.json':
                download(remote + '/' + name, directory / name)
        print(f'Verified excluded-question fold{fold} checkpoint', flush=True)
        selected = np.asarray([i for i, r in enumerate(rows) if r['fold'] == fold], dtype=int)
        model, tokenizer, recipe = load(directory)
        tensors = encode(tokenizer, [rows[i] for i in selected], recipe['max_length'])
        values = predict(model.to(args.device), tensors, device=args.device, batch=4)
        for destination, value in zip(outputs, values):
            destination[selected] = value
        tokens[selected] = tensors['attention_mask'].sum(1).numpy()
        receipts[str(fold)] = {'nodes': len(selected), 'weights': recipe['weights']}
        del model
        if args.device == 'mps':
            torch.mps.empty_cache()
        print(f'Locally scored {len(selected)} fold{fold} observations', flush=True)
    np.savez_compressed(args.output, state=outputs[0], plain=outputs[1], guided=outputs[2], tokens=tokens)
    receipt = {'inputs_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'folds': receipts, 'references_present': False, 'fresh_data_used': False,
               'observation_text_uploaded': False, 'device': args.device,
               'volume': 'srr-semantic-training-20261120', 'remote': args.remote}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'folds'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--remote', default='/verifier-large-v1')
    p.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    run(p.parse_args())
