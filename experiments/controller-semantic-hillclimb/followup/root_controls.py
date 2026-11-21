"""Complete matched historical root controls without changing the paired bank."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import threading

from banks import ordinary, committed
from medium_late_pairs import APP
from scoring import correct
from transport import resilient
import extended_acquire as old


def run(args):
    import modal
    rows, trees, judges, _ = ordinary(args.data, 'train')
    by_id = {r['id']: r for r in rows}
    chosen = json.loads((args.root / 'contract.json').read_bytes())['selection']
    pairs = {(r['id'], r['kind'], r['path']): r['response'] for r in committed(args.root / 'train.jsonl')}
    expected = {(r['id'], kind, path) for r in chosen
                for kind, paths in [('judge', ['']), ('solver', ['guided', 'guided/repair', 'guided/retry'])]
                for path in paths}
    if set(pairs) != expected:
        raise ValueError('Finish every medium-root pair first')
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.root / 'control-transport-errors.jsonl')
    old.request = request
    path = args.root / 'controls.jsonl'
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = committed(path)
        known = {(r['id'], r['kind'], r['path']): r['response'] for r in saved}
        assert len(known) == len(saved)
        lock = threading.Lock()
        for (qid, kind, p), node in known.items():
            if kind == 'solver':
                trees[qid][p] = node
            else:
                judges[qid, p] = node

        def one(item):
            qid = item['id']
            row = {**by_id[qid], 'answer': ''}
            operations = [('judge', '')] if (qid, '') not in judges else []
            if 'guided' not in trees[qid]:
                operations.append(('solver', 'guided'))
            for kind, p in operations:
                node = old.call(row, p, trees[qid], judges, kind == 'judge')
                with lock:
                    stream.write(json.dumps({'id': qid, 'kind': kind, 'path': p, 'response': node}) + '\n')
                    stream.flush()
                if kind == 'solver':
                    trees[qid][p] = node
                else:
                    judges[qid, p] = node
            return trees[qid]['guided']

        with ThreadPoolExecutor(max_workers=2) as pool:
            controls = list(pool.map(one, chosen))
    methods = {}
    for name, values in (
        ('original-guided', controls),
        ('medium-guided', [pairs[r['id'], 'solver', 'guided'] for r in chosen]),
        ('medium-review', [pairs[r['id'], 'judge', ''] for r in chosen])):
        methods[name] = {'n': len(values),
                         'correct': sum(correct(n, by_id[r['id']]['answer']) for r, n in zip(chosen, values)),
                         'mean_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in values) / len(values)}
    result = {'questions': len(chosen), 'split': 'training only', 'fresh_data_used': False,
              'methods': methods, 'extra_historical_control_responses': len(committed(path)),
              'rule': 'Every selected question contributes one matched original guided control; no outcome-based filtering.'}
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    run(p.parse_args())
