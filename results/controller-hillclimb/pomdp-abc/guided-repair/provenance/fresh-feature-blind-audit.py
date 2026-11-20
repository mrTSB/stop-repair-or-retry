"""Recompute features during hidden-reference and paid-observation-only replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'experiments/controller-hillclimb/pomdp-abc/guided-repair'))
from extended_study import evidence, methods, records
from context_pomdp import evaluate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
frozen = json.loads((args.output/'frozen.json').read_bytes())
assert frozen['evidence'] == evidence(args.data)
meta = json.loads((args.data/'test-collection.json').read_bytes())
assert meta['completed'] == meta['questions'] == 600
rows, trees, judges = records(args.data, 'test', False)
configs = methods(frozen)
checked, replays = 0, 0
for row in rows:
    for name, model, settings, method in configs:
        if model is not None:
            model._scores.clear()
        result = evaluate(row, trees[row['id']], judges, model, settings, method)
        parts = result['path'].split('/') if result['path'] else []
        paths = ['']+['/'.join(parts[:i]) for i in range(1, len(parts)+1)]
        paid = [step['path'] for step in result['decisions'] if step['queried']]
        pruned_tree = {path: trees[row['id']][path] for path in paths}
        pruned_judges = {(row['id'],path): judges[row['id'],path] for path in paid}
        expected = {k:v for k,v in result.items() if k != 'correct'}
        for replacement in ('', '9999999999999999999999999'):
            if model is not None:
                model._scores.clear()
            blinded = evaluate({**row, 'answer':replacement}, pruned_tree, pruned_judges,
                               model, settings, method)
            assert expected == {k:v for k,v in blinded.items() if k != 'correct'}, (row['id'],name)
            replays += 1
        checked += 1
policy = {k:frozen[k] for k in ('selected','models','baselines','random_control')}
report = {'questions':len(rows),'adaptive_policy_outcomes':checked,
          'reference_replacement_replays':replays,'feature_caches_cleared_before_each_run':True,
          'unvisited_solver_and_unacquired_review_nodes_removed':True,
          'policy_sha256':hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest(),
          'passed':True}
args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
