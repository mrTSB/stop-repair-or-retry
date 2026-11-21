"""Fetch a completed, checksummed observation model from the private study volume."""

import argparse
import hashlib
import json
from pathlib import Path


def fetch(args):
    import modal
    volume = modal.Volume.from_name('srr-semantic-training-20261120')
    args.output.mkdir(parents=True, exist_ok=True)

    def get(remote, local):
        temporary = local.with_suffix(local.suffix + '.partial')
        with temporary.open('wb') as stream:
            for chunk in volume.read_file(remote):
                stream.write(chunk)
        temporary.replace(local)

    # The summary is written last, after every fold and the OOF predictions.
    get(args.remote + '/training-summary.json', args.output / 'training-summary.json')
    for name in ('contract.json', 'oof.npz'):
        get(args.remote + '/' + name, args.output / name)
    full = args.output / 'full'
    full.mkdir(exist_ok=True)
    for entry in volume.listdir(args.remote + '/full'):
        name = Path(entry.path).name
        get(args.remote + '/full/' + name, full / name)
        print('Fetched ' + name, flush=True)
    recipe = json.loads((full / 'recipe.json').read_bytes())
    for name, expected in recipe['weights'].items():
        if hashlib.sha256((full / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Downloaded weight digest mismatch: ' + name)
    receipt = {'volume': 'srr-semantic-training-20261120', 'remote': args.remote,
               'all_weight_digests_verified': True,
               'files': {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(args.output.rglob('*')) if p.is_file()
                         and p.name != 'download-receipt.json'}}
    (args.output / 'download-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(json.loads((args.output / 'training-summary.json').read_bytes())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--output', type=Path, required=True)
    fetch(parser.parse_args())
