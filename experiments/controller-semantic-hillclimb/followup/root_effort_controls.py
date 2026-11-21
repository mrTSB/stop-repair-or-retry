"""Match the short actor across ordinary and deeper reviews on all192training roots."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading

from banks import ordinary, committed
from medium_late_pairs import APP
from reasoning_pilot import short_repair, inference
from scoring import correct
from transport import resilient


def run(args):
    import modal
    rows, trees, judges, _ = ordinary(args.data, 'train')
    by_id = {r['id']: r for r in rows}
    chosen = json.loads((args.root / 'contract.json').read_bytes())['selection']
    for item in committed(args.root / 'controls.jsonl'):
        if item['kind'] == 'judge':
            judges[item['id'], item['path']] = item['response']
    pilot = {(r['id'], r['kind']): r['response'] for r in committed(args.pilot)}
    pairs = {(r['id'], r['kind'], r['path']): r['response'] for r in committed(args.root / 'train.jsonl')}
    if len(chosen) != 192 or len(pairs) != 768:
        raise ValueError('Complete all192root pairs first')
    recipe = {'selection_sha256': hashlib.sha256((args.root / 'contract.json').read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'pilot_sha256': hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
              'rule': 'Every192prespecified training root, same short template and compact seed, ordinary review only.',
              'fresh_data_used': False}
    contract = args.root / 'effort-controls-contract.json'
    if contract.exists() and json.loads(contract.read_bytes()) != recipe:
        raise ValueError('Changed effort-control experiment')
    contract.write_text(json.dumps(recipe, indent=2) + '\n')
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.root / 'effort-control-transport-errors.jsonl')
    path = args.root / 'effort-controls.jsonl'
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = committed(path)
        known = {r['id']: r['response'] for r in saved}
        if len(saved) != len(known):
            raise ValueError('Duplicate effort control')
        lock = threading.Lock()
        completed = 0

        def one(item):
            nonlocal completed
            qid = item['id']
            safe = {**by_id[qid], 'answer': ''}
            prompt = short_repair(safe, trees[qid][''], judges[qid, ''])
            if qid not in known:
                cached = pilot.get((qid, 'standard-short-repair'))
                node = cached if cached is not None else inference(request, safe, prompt, False, 128)
                assert node['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
                assert node['seed'] == pairs[qid, 'solver', 'guided']['seed']
                with lock:
                    stream.write(json.dumps({'id': qid, 'response': node}) + '\n')
                    stream.flush()
                    known[qid] = node
            with lock:
                completed += 1
                if completed % 24 == 0 or completed == 192:
                    print(f'Matched ordinary-short controls {completed}/192', flush=True)
            return known[qid]

        with ThreadPoolExecutor(max_workers=2) as pool:
            controls = list(pool.map(one, chosen))
    result = {'questions': 192, 'split': 'training only', 'fresh_data_used': False, 'methods': {}}
    for name, values in [('ordinary-short', controls),
                          ('medium-short', [pairs[r['id'], 'solver', 'guided'] for r in chosen])]:
        result['methods'][name] = {'correct': sum(correct(n, by_id[r['id']]['answer']) for r, n in zip(chosen, values)),
                                   'mean_solver_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in values) / 192}
    result['paired_wins_medium'] = sum(not correct(a, by_id[r['id']]['answer']) and correct(b, by_id[r['id']]['answer'])
                                        for r, a, b in zip(chosen, controls, [pairs[r['id'], 'solver', 'guided'] for r in chosen]))
    result['paired_losses_medium'] = sum(correct(a, by_id[r['id']]['answer']) and not correct(b, by_id[r['id']]['answer'])
                                          for r, a, b in zip(chosen, controls, [pairs[r['id'], 'solver', 'guided'] for r in chosen]))
    (args.root / 'effort-controls-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'root', 'pilot'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
