"""Reuse completed development responses for the excluded-calibration hybrid.

Only development caches are imported. Every virtual policy call is charged by
the ordinary evaluator. All available responses are copied without inspecting
correctness. The parent's source and models remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import development_mixed_mentor as study


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    while not (args.parent / 'summary.json').exists() or not (args.models / 'complete.json').exists():
        time.sleep(30)
    identity = json.loads((args.parent / 'identity.json').read_bytes())
    summary = json.loads((args.parent / 'summary.json').read_bytes())
    if identity['test_used'] or summary['test_used'] or summary['development_questions'] != 256:
        raise ValueError('Only the complete development bank may be imported')
    if identity['dev_questions_sha256'] != sha(args.data / 'splits/dev-questions.jsonl'):
        raise ValueError('Development split changed')
    args.output.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name in ('dev-observations.jsonl', 'private-neural-cache.pkl'):
        source = args.parent / name
        if source.exists():
            raw = source.read_bytes()
            (args.output / name).write_bytes(raw)
            copied[name] = hashlib.sha256(raw).hexdigest()
    receipt = {'source_sha256': sha(__file__), 'parent_summary_sha256': sha(args.parent / 'summary.json'),
               'parent_identity_sha256': sha(args.parent / 'identity.json'), 'copied': copied,
               'selection_by_correctness': False, 'test_used': False,
               'charging': 'Each virtual policy call charged, including exact reused responses.'}
    (args.output / 'cache-import.json').write_text(json.dumps(receipt, indent=2) + '\n')
    args.collect, args.wait = True, True
    study.run(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'models', 'output', 'parent', 'b-reference'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    run(p.parse_args())
