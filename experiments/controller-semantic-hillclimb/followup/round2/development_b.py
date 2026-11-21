"""Select the independent ordinary LLM policy on all new development questions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary
from late_pairs import llm_policy
from study import fixed
from study_common import BASELINES, statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {'source_sha256': sha(__file__), 'policy_sha256': sha(sys.modules['late_pairs'].__file__),
                'dev_questions_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
                'test_used': False, 'stop_rules': ['action', 'agreement', .8, .95, .995, 1.],
                'reuse_review': [False, True], 'family': 'ordinary independent LLM judge'}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    complete = args.data / 'dev-ordinary-complete.json'
    while not complete.exists():
        time.sleep(30)
    receipt = json.loads(complete.read_bytes())
    for kind, name in (('base', 'dev-observations.jsonl'), ('late', 'dev-late-observations.jsonl')):
        assert sha(args.data / name) == receipt[kind + '_bank_sha256']
    rows, trees, _, judges = ordinary(args.data, 'dev')
    baseline = {name: statistics([fixed(r, trees[r['id']], *p) for r in rows]) for name, p in BASELINES.items()}
    candidates = []
    for stop in identity['stop_rules']:
        for reuse in identity['reuse_review']:
            settings = {'judge_stop': stop, 'reuse_review': reuse}
            values = [llm_policy(r, trees[r['id']], judges, settings) for r in rows]
            candidates.append({'settings': settings, 'late': True, 'family': 'B', 'dev': statistics(values)})
    best = max(candidates, key=lambda x: (x['dev']['accuracy'], -x['dev']['mean_compute']))
    result = {'B': best, 'baselines': baseline, 'development_questions': len(rows), 'test_used': False,
              'B_above_highest_baseline': best['dev']['accuracy'] - max(v['accuracy'] for v in baseline.values()),
              'data_receipt_sha256': sha(complete)}
    (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
