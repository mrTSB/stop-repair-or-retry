"""Full-panel A comparison of screened expanded-math forecasts; retain best A153."""
import argparse
import copy
from itertools import product
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import development_continuous_completion as parent
import development_nonlinear_completion_hybrid as caches
import fit_expanded_math as fitted
import development_math_specialization as ancestry
import continuous_completion_pomdp as policy
import continuous_repairability as math_policy

io, original = parent.io, parent.original
SCHEMA = 'expanded-math768-Complete16-full-development-v1'
WEIGHTS = (0., .5, 1.)


def inputs(args):
    spec = io.read(args.previous/'registration.json'); done = io.read(args.previous/'complete.json')
    old_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    data, values, _, _, _, expected = ancestry.inputs(old_args)
    if (spec != expected or done['schema'] != ancestry.SCHEMA or done['test_used']
            or done['registration_sha256'] != io.sha(args.previous/'registration.json')):
        raise ValueError('Require the completed retained math-specialized A comparison')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.previous/n): h for n, h in done['artifacts'].items()})
    summary = io.read(args.previous/'summary.json'); selected = summary['selected_A']; path = Path(selected['model_path'])
    if selected['policy_module'] != 'continuous_completion_pomdp' or io.sha(path) != selected['model_sha256']:
        raise ValueError('Selected full-panel A changed')
    base = pickle.loads(path.read_bytes())
    fspec = io.read(args.model/'registration.json'); fdone = io.read(args.model/'complete.json')
    if (fspec['schema'] != fitted.SCHEMA or fdone['schema'] != fitted.SCHEMA
            or fdone['registration_sha256'] != io.sha(args.model/'registration.json')
            or fdone['new_neural_calls'] or fdone['cloud_calls'] or fdone['development_or_test_read']):
        raise ValueError('Require completed screened TRAIN math fit')
    io.verify(fspec['source_files']); io.verify(fspec['input_evidence'])
    io.verify({str(args.model/n): h for n, h in fdone['digests'].items()})
    path = args.model/'math.pkl'
    if io.sha(path) != fdone['model_sha256']: raise ValueError('Math model changed')
    math = pickle.loads(path.read_bytes())
    if math.fit_receipt != fdone['fit_receipt']: raise ValueError('Math fit receipt changed')
    for name in ('prior', 'transition', 'cost'):
        np.testing.assert_array_equal(getattr(math.dynamics['math'], name), getattr(base.math.dynamics['math'], name))
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], **fspec['input_evidence'],
            **{str((args.previous/n).resolve()): h for n, h in done['artifacts'].items()},
            **{str((args.model/n).resolve()): h for n, h in fdone['digests'].items()},
            str((args.previous/'complete.json').resolve()): io.sha(args.previous/'complete.json'),
            str((args.model/'complete.json').resolve()): io.sha(args.model/'complete.json')},
        'paths': {n: str(getattr(args, n).resolve()) for n in ('previous', 'model', 'output')},
        'dev_questions_sha256': spec['dev_questions_sha256'], 'A': selected,
        'previous_A_candidates': [*spec['previous_A_candidates'], selected],
        'penalties': list(original.PENALTIES), 'posterior_weights': list(WEIGHTS), 'candidate_count': 21,
        'development_questions': 256, 'science_policy_exactly_retained': True, 'new_math_training_questions': 768,
        'original_five_baselines_unchanged': True,
        'matched_controls': 'Recompute full-panel Complete controls with the same selected math route; preserve old controls too',
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False, 'fresh_validation': False}
    return data, values, base, math, summary, record


def replay(args, data, values, base, math, previous_summary, record):
    rows, trees, _, old_observer, _, _, _ = data
    deploy_base = copy.deepcopy(base); deploy_math = copy.deepcopy(math)
    base.math.classifier = old_observer; base.science.classifier = caches.dense_cache(data, values)
    old_items = [policy.evaluate(r, trees[r['id']], {}, base, {}, 'A') for r in rows]
    saved = io.read(args.previous/'private-candidate-items.json')[record['A']['key']]
    if old_items != saved: raise ValueError('Previous A decisions or costs changed')
    math.classifier = old_observer
    candidates, all_items = [], {}
    for index, (penalty, weight) in enumerate(product(original.PENALTIES, WEIGHTS)):
        setting = {'penalty': penalty, 'posterior_weight': weight}
        items = [math_policy.evaluate(r, trees[r['id']], {}, math, setting, 'A') if r['subject'] == 'math'
                 else old_items[i] for i, r in enumerate(rows)]
        model = copy.deepcopy(deploy_base); model.math = copy.deepcopy(deploy_math); model.math_settings = setting
        model.fit_receipt = {**model.fit_receipt, 'schema': SCHEMA, 'math_model_sha256': io.sha(args.model/'math.pkl'),
                            'science_policy_unchanged': True}
        name = 'expanded-math-'+str(index); path = args.output/(name+'.pkl'); io.atomic_bytes(path, pickle.dumps(model, protocol=5))
        candidates.append({'key': name, 'model_path': str(path.resolve()), 'model_sha256': io.sha(path),
            'policy_module': 'continuous_completion_pomdp', 'settings': {}, 'math_settings': setting,
            'science_settings': base.science_settings, 'dev': original.statistics(items)})
        all_items[name] = items
    selected = max([*record['previous_A_candidates'], *candidates], key=original.fit_rank)
    selected_items = all_items.get(selected['key'], old_items)
    if original.statistics(selected_items) != selected['dev']: raise ValueError('Selected metrics/items differ')
    baseline = {n: original.statistics([original.fixed(r, trees[r['id']], *s) for r in rows]) for n, s in original.BASELINES.items()}
    if baseline != previous_summary['baselines']: raise ValueError('Original five baselines changed')
    math_items = {r['id']: v for r, v in zip(rows, selected_items) if r['subject'] == 'math'}
    matched = {}
    for name, prefix, conditional in original.MATCHED:
        items = [math_items[r['id']] if r['subject'] == 'math' else original.matched_item(r, trees[r['id']], prefix, conditional) for r in rows]
        matched[name] = {'full256_with_retained_math_A': original.statistics(items),
            'science': {s: original.statistics([v for r, v in zip(rows, items)
                if (r['subject'] != 'math' if s == 'all' else r['subject'] == s)]) for s in ('all', 'chemistry', 'physics')}}
        if matched[name]['science'] != previous_summary['matched_complete_controls'][name]['science']:
            raise ValueError('A matched science control changed')
    summary = {'schema': SCHEMA, 'selected_A': selected, 'candidates': candidates,
        'previous_A_candidates': record['previous_A_candidates'], 'baselines': baseline,
        'matched_complete_controls': matched, 'previous_matched_complete_controls': previous_summary['matched_complete_controls'],
        'math_metrics': original.statistics(list(math_items.values())), 'science_policy_exactly_retained': True, 'new_math_training_questions': 768,
        'old_A_exactly_reproduced': True, 'development_questions': 256,
        'A_above_highest_baseline': selected['dev']['accuracy']-max(v['accuracy'] for v in baseline.values()),
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False, 'fresh_validation': False}
    io.write(args.output/'summary.json', summary); io.write(args.output/'private-candidate-items.json', all_items)
    io.write(args.output/'identity.json', {'schema': SCHEMA, 'selected_A': selected, 'dev_questions_sha256': record['dev_questions_sha256'],
        'registration_sha256': args.registration_sha256, 'policy_module': 'continuous_completion_pomdp', 'test_used': False})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'artifacts': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}, 'test_used': False})
    print({'selected_A': selected['dev'], 'math': summary['math_metrics'],
           'new_candidates': [(v['key'], v['dev']['correct']) for v in candidates]}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'replay'))
    for n in ('previous', 'model', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with parent.parent.previous.previous.parent_io.no_test_reads(), original.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered math development study')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior outputs')
            replay(args, *bundle)


if __name__ == '__main__': main()
