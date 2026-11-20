"""Independently reconcile policy charges and replay with hidden/unvisited data removed."""

import argparse
import json
from pathlib import Path

import numpy as np

from extended_acquire import JUDGE_ACTIVE, JUDGE_TOTAL, digest, read, stem
from extended_study import evidence, methods, records, training_archive
from context_pomdp import evaluate
from study import fixed
from scoring import correct


def prefixes(path):
    parts = path.split('/') if path else []
    return [''] + ['/'.join(parts[:i]) for i in range(1, len(parts)+1)]


def audit(source, output):
    frozen_path = output / 'frozen.json'
    frozen = json.loads(frozen_path.read_bytes())
    if frozen['evidence'] != evidence(source):
        raise ValueError('Frozen evidence changed')
    rows, trees, judges = records(source, 'test', False)
    metadata = json.loads((source / 'test-collection.json').read_bytes())
    if metadata['completed'] != len(rows) or metadata['freeze_sha256'] != digest(frozen_path):
        raise ValueError('The full frozen holdout has not completed')
    previous = list(training_archive(source))
    previous += [r for split in ('train', 'dev') for r in read(source / 'splits' / f'{split}-questions.jsonl')]
    if {stem(r['question']) for r in rows} & {stem(r['question']) for r in previous}:
        raise ValueError('Training and test normalized question text overlaps')
    policies = methods(frozen)
    count = 0
    for row in rows:
        tree = trees[row['id']]
        for name, config in frozen['baselines'].items():
            result = fixed(row, tree, config['action'], config['depth'])
            expected = sum(tree[p]['input_tokens']+tree[p]['output_tokens'] for p in prefixes(result['path']))
            if result['solver_tokens'] != expected or result['compute'] != expected*4/3.09:
                raise ValueError('Baseline charge failed independent reconciliation')
            if result['correct'] != int(correct(tree[result['path']], row['answer'])):
                raise ValueError('Baseline grade failed independent reconciliation')
            count += 1
        for name, model, settings, method in policies:
            result = evaluate(row, tree, judges, model, settings, method)
            paid_paths = prefixes(result['path'])
            acquired = [d['path'] for d in result['decisions'] if d['queried']]
            solver = sum(tree[p]['input_tokens']+tree[p]['output_tokens'] for p in paid_paths)
            llm = sum(judges[row['id'], p]['input_tokens']+judges[row['id'], p]['output_tokens'] for p in acquired)
            np.testing.assert_allclose(result['compute'], (4*solver+JUDGE_ACTIVE*llm)/3.09)
            np.testing.assert_allclose(result['total_parameter_compute'], (4*solver+JUDGE_TOTAL*llm)/3.09)
            if result['solver_tokens'] != solver or result['judge_tokens'] != llm:
                raise ValueError('Recorded raw token charge differs')
            if len(paid_paths) != result['solver_calls'] or len(acquired) != result['judge_calls']:
                raise ValueError('Recorded call count differs')
            if result['solver_calls'] > 3 or result['judge_calls'] > 2:
                raise ValueError('Inference horizon exceeded')
            if name == 'A' and acquired:
                raise ValueError('A acquired an LLM observation')
            if result['correct'] != int(correct(tree[result['path']], row['answer'])):
                raise ValueError('Adaptive grade failed independent reconciliation')
            pruned_tree = {p: tree[p] for p in paid_paths}
            pruned_judges = {(row['id'], p): judges[row['id'], p] for p in acquired}
            blinded = evaluate({**row, 'answer': ''}, pruned_tree, pruned_judges, model, settings, method)
            del blinded['correct']
            ungraded = {k: v for k, v in result.items() if k != 'correct'}
            if blinded != ungraded:
                raise ValueError('Hidden labels or unvisited branches affected policy decisions')
            count += 1
    print(json.dumps({'questions': len(rows), 'policy_outcomes_audited': count,
                      'checked': ['fixed horizon', 'grades', 'all visited solver and acquired LLM tokens',
                                  'active/total parameter compute', 'role separation',
                                  'hidden-reference invariance', 'no access to unvisited branches',
                                  'normalized train/test question disjointness']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data, args.output)


if __name__ == '__main__':
    main()
