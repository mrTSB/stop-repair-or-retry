"""Compose the improved A with the unchanged mentor and replay full development."""
import argparse
import copy
from itertools import product
from pathlib import Path
import pickle
import sys

from threadpoolctl import threadpool_limits

import development_nonlinear_completion_hybrid as previous
import development_continuous_completion as new_a
import continuous_completion_hybrid as policy
import clustered_completion_pomdp as clustered

io, canonical, a_source, c_source = previous.io, previous.canonical, previous.a_source, previous.c_source
SCHEMA = 'continuous-Complete16-mentor8-full-development-v1'
dense_cache = previous.dense_cache


def inputs(args):
    old_args = previous.saved_args(args.previous)
    bundle, adata, values, model, old_record = previous.inputs(old_args)
    old_spec = io.read(args.previous/'registration.json'); old_done = io.read(args.previous/'complete.json')
    if (old_record != old_spec or old_done['schema'] != previous.SCHEMA
            or old_done['registration_sha256'] != io.sha(args.previous/'registration.json') or old_done['test_used']):
        raise ValueError('Require exact completed prior hybrid comparison')
    io.verify(old_spec['source_files']); io.verify(old_spec['input_evidence'])
    io.verify({str(args.previous/n): h for n, h in old_done['artifacts'].items()})
    spec = io.read(args.new_a/'registration.json'); done = io.read(args.new_a/'complete.json')
    if (spec['schema'] != new_a.SCHEMA or done['schema'] != new_a.SCHEMA
            or done['registration_sha256'] != io.sha(args.new_a/'registration.json')
            or done['test_used'] or spec['dev_questions_sha256'] != old_spec['dev_questions_sha256']):
        raise ValueError('New A must use the identical development panel without fresh-test access')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.new_a/n): h for n, h in done['artifacts'].items()})
    summary = io.read(args.new_a/'summary.json'); chosen = summary['selected_A']
    if (summary['baselines'] != old_spec['baselines'] or chosen['policy_module'] != 'continuous_completion_pomdp'
            or summary['test_used'] or summary['new_neural_calls'] or summary['new_cloud_calls']):
        raise ValueError('Require completed continuous-A result with unchanged baselines')
    path = Path(chosen['model_path'])
    if io.sha(path) != chosen['model_sha256'] or path.parent.resolve() != args.new_a.resolve():
        raise ValueError('Selected A artifact changed')
    selected = pickle.loads(path.read_bytes())
    science = selected.science
    if (type(science) is not clustered.Planner or type(science.kernel) is not clustered.Kernel
            or science.fit_receipt['state_count'] != 16 or science.feature_family != 'raw20_plus_project256'
            or not science.fit_receipt['complete_is_terminal'] or science.fit_receipt['maximum_additional_solver_calls'] != 2
            or selected.science_settings != {'penalty': .02, 'posterior_weight': .5}):
        raise ValueError('Require the exact retained clustered Complete16 posterior-blend policy')
    model.science = copy.deepcopy(science); model.science_settings = copy.deepcopy(selected.science_settings)
    model.fit_receipt = {**model.fit_receipt, 'schema': SCHEMA, 'science_A': science.fit_receipt,
        'science_belief_update': 'Half finite-filter, half paid-history posterior; approximate estimator',
        'base_A_model_sha256': chosen['model_sha256'], 'transfer_and_mentor_unchanged': True,
        'source_hybrid_model_sha256': old_spec['input_evidence'][str((old_args.model/'hybrid.pkl').resolve())]}
    old_summary = io.read(args.previous/'summary.json')
    if old_summary['selected_C'] != old_record['previous_C']:
        raise ValueError('Explicit retention update needed if prior hybrid improved C')
    record = {**old_record, 'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**old_spec['input_evidence'], **spec['input_evidence'],
            **{str((args.previous/n).resolve()): h for n, h in old_done['artifacts'].items()},
            **{str((args.new_a/n).resolve()): h for n, h in done['artifacts'].items()},
            str((args.previous/'complete.json').resolve()): io.sha(args.previous/'complete.json'),
            str((args.new_a/'complete.json').resolve()): io.sha(args.new_a/'complete.json')},
        'paths': {n: str(getattr(args, n).resolve()) for n in ('previous', 'new_a', 'output')},
        'A': chosen, 'science_posterior_weight': .5, 'new_transfer_fit': False,
        'composition': 'Replace only the unqueried A with selected clustered/continuous A; same mentor and transfer',
        'matched_complete_controls': summary['matched_complete_controls']}
    return bundle, adata, values, model, record


def replay(args, bundle, adata, values, model, record):
    rows, _, trees, judges, _, _, caches = bundle[:7]
    deploy = copy.deepcopy(model); model.science.classifier = dense_cache(adata, values)
    candidates, all_items, grid = [], {}, []
    with canonical.cached_model(model.math_C, caches):
        old_items = [canonical.previous.raw_runtime.evaluate(r, trees[r['id']], judges, model.math_C,
                    record['previous_C']['settings'], 'C') for r in rows]
        if canonical.previous.measured(old_items) != record['previous_C']['dev']:
            raise ValueError('Retained C no longer reproduces exactly')
        math = {r['id']: item for r, item in zip(rows, old_items) if r['subject'] == 'math'}
        for penalty, threshold in product(record['penalties'], record['thresholds']):
            setting = {'penalty': penalty, 'threshold': threshold}
            full = {'by_subject': {'math': record['math_settings'], 'chemistry': setting, 'physics': setting}}
            items = [policy.evaluate(r, trees[r['id']], judges, model, full, 'C') for r in rows]
            for row, item in zip(rows, items):
                if row['subject'] == 'math':
                    for k in ('path', 'correct', 'decisions', 'compute', 'total_parameter_compute', 'verifier_token_slots', 'solver_tokens', 'judge_tokens'):
                        if item[k] != math[row['id']][k]: raise ValueError('Changed retained math route or costs: '+k)
                if item['solver_calls'] > 3 or item['judge_calls'] > 1: raise ValueError('Exceeded solver/reviewer horizon')
            grid.append((full, items))
        for utility in record['utilities']:
            selected = {s: max(range(21), key=lambda i: (
                (v := canonical.previous.measured([x for r, x in zip(rows, grid[i][1]) if r['subject'] == s]))['accuracy']
                - utility*v['mean_compute']/1000, v['accuracy'], -v['mean_compute'])) for s in ('chemistry', 'physics')}
            settings = {'by_subject': {'math': record['math_settings'], **{s: grid[i][0]['by_subject'][s] for s, i in selected.items()}}}
            items = [grid[0][1][i] if r['subject'] == 'math' else grid[selected[r['subject']]][1][i] for i, r in enumerate(rows)]
            grid.append((settings, items))
    path = args.output/'hybrid.pkl'; io.atomic_bytes(path, pickle.dumps(deploy, protocol=5))
    for index, (settings, items) in enumerate(grid):
        key = 'complete-hybrid-'+str(index); all_items[key] = items
        candidates.append({'key': key, 'model_path': str(path.resolve()), 'model_sha256': io.sha(path),
            'policy_module': 'continuous_completion_hybrid', 'settings': settings, 'dev': canonical.previous.measured(items)})
    if len(candidates) != 26: raise ValueError('Incomplete registered grid')
    eligible = [c for c in [record['previous_C'], *candidates] if c['dev']['mean_compute'] < record['B']['dev']['mean_compute']]
    selected = max(eligible, key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
    summary = {'schema': SCHEMA, 'selected_C': selected, 'candidates': candidates,
        **{k: record[k] for k in ('A', 'B', 'previous_C', 'baselines', 'matched_complete_controls')},
        'C_above_higher_A_B': selected['dev']['accuracy']-max(record[k]['dev']['accuracy'] for k in ('A', 'B')),
        'C_compute_saved_vs_B': 1-selected['dev']['mean_compute']/record['B']['dev']['mean_compute'],
        'previous_C_exactly_reproduced': True, 'math_exactly_preserved': True, 'development_questions': 256,
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False, 'fresh_validation': False}
    io.write(args.output/'summary.json', summary); io.write(args.output/'private-candidate-items.json', all_items)
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'artifacts': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}, 'test_used': False})
    print({'selected_C': selected['dev'], 'new_best_correct': max(c['dev']['correct'] for c in candidates),
           'gap': summary['C_above_higher_A_B']}, flush=True)



def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'replay'))
    for n in ('previous', 'new_a', 'output'): p.add_argument('--'+n.replace('_', '-'), type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with c_source.guard.no_test_reads(), a_source.original.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered hybrid replay')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior replay')
            replay(args, *bundle)


if __name__ == '__main__': main()
