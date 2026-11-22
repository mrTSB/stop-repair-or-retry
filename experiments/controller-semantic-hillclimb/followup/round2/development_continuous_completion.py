"""Register full-panel development comparison of two paid-posterior blends."""
import argparse
import copy
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from threadpoolctl import threadpool_limits

import development_clustered_completion as parent
import continuous_completion_pomdp as policy

io, original = parent.io, parent.original
SCHEMA = 'clustered-Complete16-continuous-belief-development-v1'
WEIGHTS = (.5, 1.)


def inputs(args):
    spec = io.read(args.previous/'registration.json'); done = io.read(args.previous/'complete.json')
    old_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    data, models, values, old_path, expected = parent.inputs(old_args)
    if (spec != expected or done['schema'] != parent.SCHEMA
            or done['registration_sha256'] != io.sha(args.previous/'registration.json')
            or done['test_used'] or old_args.output.resolve() != args.previous.resolve()):
        raise ValueError('Require the exact completed clustered observation study')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.previous/n): h for n, h in done['artifacts'].items()})
    summary = io.read(args.previous/'summary.json')
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'],
            **{str((args.previous/n).resolve()): h for n, h in done['artifacts'].items()},
            str((args.previous/'complete.json').resolve()): io.sha(args.previous/'complete.json')},
        'paths': {n: str(getattr(args, n).resolve()) for n in ('previous', 'output')},
        'dev_questions_sha256': spec['dev_questions_sha256'],
        'previous_A_candidates': [*spec['previous_A_candidates'], summary['selected_A']],
        'weights': list(WEIGHTS), 'penalties': list(original.PENALTIES),
        'families': list(models), 'candidate_count': 32, 'development_questions': 256,
        'runtime_belief': 'Fixed blend of finite filter and paid-history posterior; approximate, not exact Bayesian filtering',
        'future_values': 'Unchanged exact finite-observation two-step Bellman backup',
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False, 'fresh_validation': False,
        'selection': 'Full-development accuracy then compute; retain stronger previous A and all controls'}
    return data, models, values, old_path, record


def replay(args, data, models, values, old_path, record):
    rows, trees, base, old_observer, base_summary, _, bundle = data
    live = copy.deepcopy(base); live.math.classifier = live.science.classifier = old_observer
    old_items = [parent.previous.routed.evaluate(r, trees[r['id']], {}, live, {}, 'A') for r in rows]
    if old_items != io.read(old_path/'private-candidate-items.json')[base_summary['completion_A']['key']]:
        raise ValueError('Original paid decisions/costs changed')
    math = {r['id']: v for r, v in zip(rows, old_items) if r['subject'] == 'math'}
    if original.statistics(list(math.values())) != base_summary['math_metrics']: raise ValueError('Math changed')
    baselines, matched = parent.previous.previous.controls(rows, trees, math, base_summary)
    observer = parent.previous.previous.CachedDense(old_observer.directory, old_observer.temperature)
    observer._recipe = old_observer._recipe; observer._retriever = old_observer._retriever
    observer._backgrounds = old_observer._backgrounds
    observer.expected = {(v['id'], v['path']): v['input_sha256'] for v in bundle['metadata']}
    observer.saved = {(v['id'], v['path']): (*[values[k][i] for k in ('state', 'plain', 'guided')], int(values['token_slots'][i]))
                      for i, v in enumerate(bundle['metadata'])}
    observer.joint = {(v['id'], v['path']): values['joint256'][i] for i, v in enumerate(bundle['metadata'])}
    candidates, all_items = [], {}
    old_grid = io.read(args.previous/'private-candidate-items.json')
    for family, fitted in models.items():
        model = copy.deepcopy(fitted); model.classifier = observer
        for index, penalty in enumerate(original.PENALTIES):
            zero = [math[r['id']] if r['subject'] == 'math' else policy.evaluate_science(
                r, trees[r['id']], {}, model, {'penalty': penalty, 'posterior_weight': 0.}, 'A') for r in rows]
            if zero != old_grid[family+'-'+str(index)]: raise ValueError('Zero blend must exactly reproduce prior decisions and charges')
        for weight in WEIGHTS:
            settings_grid = [{'penalty': p, 'posterior_weight': weight} for p in original.PENALTIES]
            grid = [[math[r['id']] if r['subject'] == 'math' else policy.evaluate_science(
                r, trees[r['id']], {}, model, setting, 'A') for r in rows] for setting in settings_grid]
            chosen = {s: max(range(7), key=lambda i: original.fit_rank({'dev': original.statistics(
                [v for r, v in zip(rows, grid[i]) if r['subject'] == s])})) for s in ('chemistry', 'physics')}
            grid.append([math[r['id']] if r['subject'] == 'math' else grid[chosen[r['subject']]][i] for i, r in enumerate(rows)])
            settings_grid.append({'by_subject': {s: settings_grid[i] for s, i in chosen.items()}})
            for index, (items, setting) in enumerate(zip(grid, settings_grid)):
                name = family+'-w'+str(weight)+'-'+str(index); path = args.output/(name+'.pkl')
                deploy = policy.SubjectPlanner(copy.deepcopy(base.math), copy.deepcopy(fitted), base.math_settings, setting,
                    {'schema': SCHEMA, 'math_unchanged': True, 'source_study': io.sha(args.previous/'complete.json')})
                io.atomic_bytes(path, pickle.dumps(deploy, protocol=5))
                candidates.append({'key': name, 'model_path': str(path.resolve()), 'model_sha256': io.sha(path),
                    'policy_module': 'continuous_completion_pomdp', 'settings': {}, 'science_settings': setting,
                    'dev': original.statistics(items)})
                all_items[name] = items
    if len(candidates) != record['candidate_count']: raise ValueError('Candidate grid changed')
    selected = max([*record['previous_A_candidates'], *candidates], key=original.fit_rank)
    summary = {'schema': SCHEMA, 'selected_A': selected, 'candidates': candidates,
        'previous_A_candidates': record['previous_A_candidates'], 'baselines': baselines, 'matched_complete_controls': matched,
        'development_questions': 256, 'zero_blend_exactly_reproduced': True, 'math_exactly_reproduced': True,
        'A_above_highest_baseline': selected['dev']['accuracy']-max(v['accuracy'] for v in baselines.values()),
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False, 'fresh_validation': False}
    io.write(args.output/'private-candidate-items.json', all_items); io.write(args.output/'summary.json', summary)
    io.write(args.output/'identity.json', {'schema': SCHEMA, 'selected_A': selected,
        'dev_questions_sha256': record['dev_questions_sha256'], 'registration_sha256': args.registration_sha256,
        'policy_module': 'continuous_completion_pomdp', 'policy_sha256': io.sha(policy.__file__), 'test_used': False})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'artifacts': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False})
    print({'selected_A': selected['dev'], 'new_candidates': [(v['key'], v['dev']['correct']) for v in candidates]}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'replay'))
    for n in ('previous', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with parent.previous.previous.parent_io.no_test_reads(), original.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered study')
        with io.lock(args.output/'.run.lock'):
            parent.previous.previous.preserve_partial(args.output); replay(args, *bundle)


if __name__ == '__main__': main()
