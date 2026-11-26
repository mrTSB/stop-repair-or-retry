"""Replay the latest A/B/C and five baselines from paid cached observations.

Invoke through scripts/reproduction/run.py to retain archived path identities.
No fitting, neural forward, cloud call, or reserved test read is performed.
"""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / 'experiments/controller-semantic-hillclimb/followup/round2'
sys.path.insert(0, str(AREA))

from threadpoolctl import threadpool_limits
import development_math_specialization as a_runtime
import development_bayesian_hybrid_v3 as c_runtime

io = c_runtime.io
RUNS = ROOT / 'runs/controller-semantic-followup/round2'
FIELDS = ('id', 'path', 'solver_calls', 'solver_tokens', 'judge_calls', 'judge_tokens',
          'compute', 'total_parameter_compute', 'verifier_components', 'verifier_calls',
          'verifier_token_slots')


def same(actual, expected, *, grade=True):
    for key in (*FIELDS, *(('correct',) if grade else ())):
        if key in actual or key in expected:
            if actual.get(key) != expected.get(key):
                raise ValueError(f'Replay differs for {actual.get("id")}: {key}')


def paid(item, tree, judges):
    parts = item['path'].split('/') if item['path'] else []
    visited = [''] + ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
    observed = {p: tree[p] for p in visited}
    acquired = {(item['id'], d['path']): judges[item['id'], d['path']]
                for d in item['decisions'] if d['queried']}
    return observed, acquired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    counts = {'policy_outcomes': 0, 'blind_paid_replays': 0}
    results = {}
    guards = c_runtime.original.prior
    with guards.guard.no_test_reads(), guards.guarded.no_model_calls(), threadpool_limits(limits=1):
        a_folder = RUNS / 'development-math-specialization-v1'
        a_spec = io.read(a_folder / 'registration.json')
        a_args = SimpleNamespace(**{k: Path(v) for k, v in a_spec['paths'].items()})
        data, values, _, _, _, record = a_runtime.inputs(a_args)
        if record != a_spec:
            raise ValueError('A input registration differs')
        rows, trees, _, old_observer, _, _, _ = data
        a_summary = io.read(a_folder / 'summary.json')
        selected = a_summary['selected_A']
        if io.sha(selected['model_path']) != selected['model_sha256']:
            raise ValueError('A selected model changed')
        a = pickle.loads(Path(selected['model_path']).read_bytes())
        a.math.classifier = old_observer
        a.science.classifier = a_runtime.caches.dense_cache(data, values)
        expected_a = io.read(a_folder / 'private-candidate-items.json')[selected['key']]
        results['A'] = []
        for row, expected in zip(rows, expected_a, strict=True):
            item = a_runtime.policy.evaluate(row, trees[row['id']], {}, a, {}, 'A')
            same(item, expected)
            observed, acquired = paid(item, trees[row['id']], {})
            blind = a_runtime.policy.evaluate({**row, 'answer': ''}, observed, acquired, a, {}, 'A')
            same(blind, item, grade=False)
            results['A'].append(item)
            counts['blind_paid_replays'] += 1
        for name, setting in a_runtime.original.BASELINES.items():
            results[name] = [a_runtime.original.fixed(row, trees[row['id']], *setting) for row in rows]
        print('A and five fixed baselines reproduced', flush=True)

        c_folder = RUNS / 'development-bayesian-model-hybrid-v3'
        spec = io.read(c_folder / 'registration.json')
        with c_runtime.configured():
            bundle, _, _, record = c_runtime.original.inputs(
                SimpleNamespace(previous=Path(spec['previous']), output=c_folder))
        if record != spec:
            raise ValueError('C input registration differs')
        c_rows, c_trees, c_judges, _, caches, _, _, b_rows, b_trees, b_judges, *_ = bundle
        summary = io.read(c_folder / 'summary.json')
        b_expected = io.read(RUNS / 'development-subject-experts-v1/private-selected-items.json')['B']
        results['B'] = []
        for row, expected in zip(b_rows, b_expected, strict=True):
            item = guards.policy.evaluate_B(row, b_trees[row['id']], b_judges, summary['B']['settings'])
            same(item, expected)
            observed, acquired = paid(item, b_trees[row['id']], b_judges)
            blind = guards.policy.evaluate_B({**row, 'answer': ''}, observed, acquired, summary['B']['settings'])
            same(blind, item, grade=False)
            results['B'].append(item)
            counts['blind_paid_replays'] += 1
        print('B reproduced', flush=True)

        chosen = summary['C_candidate']
        if io.sha(chosen['model_path']) != chosen['model_sha256']:
            raise ValueError('C selected model changed')
        model = pickle.loads(Path(chosen['model_path']).read_bytes())
        c_expected = io.read(c_folder / 'private-selected-items.json')
        results['C'] = []
        with ExitStack() as stack:
            for child in model.models.values():
                parents = child.models if isinstance(child, c_runtime.original.policy.Planner) else [child]
                for parent in parents:
                    stack.enter_context(c_runtime.original.canonical.cached_model(parent, caches))
            for row, expected in zip(c_rows, c_expected, strict=True):
                item = c_runtime.original.policy.routed(row, c_trees[row['id']], c_judges, model, {}, 'C')
                same(item, expected)
                observed, acquired = paid(item, c_trees[row['id']], c_judges)
                blind = c_runtime.original.policy.routed({**row, 'answer': ''}, observed, acquired, model, {}, 'C')
                same(blind, item, grade=False)
                results['C'].append(item)
                counts['blind_paid_replays'] += 1
        print('C reproduced', flush=True)
        table = []
        expected_table = io.read(AREA / 'verified_bayesian_checkpoint_v1.json')['rows']
        for expected in expected_table:
            name = expected['method']
            items = results[name]
            value = {'method': name, 'n': len(items), 'correct': sum(v['correct'] for v in items),
                     'accuracy': sum(v['correct'] for v in items) / len(items),
                     'mean_compute': math.fsum(v['compute'] for v in items) / len(items)}
            for key in ('n', 'correct', 'accuracy', 'mean_compute'):
                if not math.isclose(value[key], expected[key], rel_tol=1e-12, abs_tol=1e-8):
                    raise ValueError('Headline table differs: ' + name + '/' + key)
            table.append(value)
            counts['policy_outcomes'] += len(items)
        if not {r['id'] for r in rows} == {r['id'] for r in c_rows} == {r['id'] for r in b_rows}:
            raise ValueError('Controller question panels differ')
        report = {'passed': True, **counts, 'rows': table, 'test_used': False,
                  'fresh_validation': False, 'new_model_calls': 0, 'new_training': False}
        io.write(args.output / 'summary.json', report)
        io.write(args.output / 'per-question.json', results)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
