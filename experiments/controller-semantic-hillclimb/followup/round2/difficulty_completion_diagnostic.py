"""Whole-policy TRAIN crossfit for persistent-difficulty Complete32 POMDPs."""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import policy_crossfit_diagnostic as previous
import difficulty_completion_pomdp as candidate

io, fixed, nested = previous.io, previous.fixed, previous.nested
SCHEMA = 'persistent-difficulty-Complete32-whole-policy-TRAIN-v1'


def inputs(args):
    rows, trees, observer, data, parent = previous.inputs(args)
    prior = io.read(args.control/'registration.json'); done = io.read(args.control/'complete.json')
    expected = {**parent, 'paths': {**parent['paths'], 'output': str(args.control.resolve())}}
    if prior != expected or done['registration_sha256'] != io.sha(args.control/'registration.json'):
        raise ValueError('Require the same sealed whole-policy comparison and data')
    io.verify(prior['source_files']); io.verify(prior['input_evidence'])
    io.verify({str(args.control/k): v for k, v in done['digests'].items()})
    controls = io.read(args.control/'private-held-items.json')
    strongest = max(controls, key=lambda k: (previous.statistics(list(controls[k].values()))['accuracy'],
        -previous.statistics(list(controls[k].values()))['mean_compute'], k))
    states = {r['id']: candidate.labels(r, trees[r['id']]) for r in rows}
    labels = np.asarray([states[r['id']][r['path']] for r in data['records']])
    np.testing.assert_array_equal(labels%16, data['labels'])
    data = {**data, 'labels': labels}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**parent['input_evidence'],
            **{str((args.control/k).resolve()): v for k, v in done['digests'].items()},
            str((args.control/'complete.json').resolve()): io.sha(args.control/'complete.json')},
        'paths': {k: str(getattr(args, k).resolve()) for k in ('expanded_fit', 'control', 'output')},
        'training_questions': 1280, 'common_held_questions': 768, 'state_count': 32,
        'latent_difficulty': 'Fewer than3 correct endpoints among all10 paired TRAIN responses, constant across a question',
        'difficulty_at_inference': 'Hidden; predicted from paid raw/dense observations, never observed from labels',
        'transition_support': 'Difficulty preserved and next current correctness equals selected current action outcome',
        'families': list(nested.FAMILIES), 'forecast': '96 ExtraTrees,depth8,leaf24,maxfeatures0.7,5%Laplace prior',
        'outer_question_folds': 3, 'inner_question_emission_folds': 3, 'hyperparameter_search': False,
        'smoothing': 32., 'observation_kernel': 'Original30-symbol marginal current/action-correctness kernel',
        'penalty': .02, 'posterior_weight': .5, 'maximum_solver_continuations': 2,
        'strongest_control': strongest, 'control_rule': 'Best accuracy then least compute among all three sealed whole-policy controls',
        'screen': 'At least2pp held-question accuracy above strongest control and no higher mean compute',
        'label_sha256': fixed.array_sha(labels), 'cloud_calls': 0, 'new_neural_calls': 0,
        'solver_unchanged': True, 'development_or_test_read': False, 'fresh_validation': False}
    return rows, trees, observer, data, controls, record


def fit_fold(rows, trees, observer, data, family, fold):
    outer = np.asarray([r['fold'] for r in data['records']]); inner = np.asarray([nested.inner_fold(r['id']) for r in data['records']])
    train = outer != fold; held = (outer == fold)&(np.arange(len(outer)) < 3*768)
    receipt = nested.group_receipt(data['records'], train, held); p = np.full((len(outer), 32), np.nan)
    x, y = data['features'][family], data['labels']; receipt['emission_folds'] = []
    for k in range(3):
        fit, validation = train&(inner != k), train&(inner == k)
        receipt['emission_folds'].append(nested.group_receipt(data['records'], fit, validation))
        model = candidate.fit_forecaster(x[fit], y[fit]); p[validation] = model.predict(x[validation], data['valid'][validation])
    if not np.isfinite(p[train]).all() or np.isfinite(p[~train]).any(): raise ValueError('Emission prediction scope changed')
    forecaster = candidate.fit_forecaster(x[train], y[train]); fit_ids = {r['id'] for r, v in zip(data['records'], train) if v}
    train_rows = [r for r in rows if r['id'] in fit_ids]; train_trees = {r['id']: trees[r['id']] for r in train_rows}
    model = candidate.fit_dynamics(train_rows, train_trees, observer,
        [r for r, v in zip(data['records'], train) if v], p[train], forecaster, family, receipt)
    held_ids = {r['id'] for r, v in zip(data['records'], held) if v}
    if held_ids&fit_ids: raise ValueError('Fitting question entered held evaluation')
    return {r['id']: candidate.evaluate(r, trees[r['id']], model) for r in rows if r['id'] in held_ids}, receipt, model


def diagnose(args, rows, trees, observer, data, controls, record):
    ids = [r['id'] for r in rows[:768]]; control = controls[record['strongest_control']]
    if set(control) != set(ids): raise ValueError('Changed common evaluation questions')
    results, receipts, reports = {}, {}, {}; draws = np.random.default_rng(238031).integers(0, 768, (2000, 768))
    for family in nested.FAMILIES:
        results[family], receipts[family] = {}, []
        for fold in range(3):
            items, receipt, model = fit_fold(rows, trees, observer, data, family, fold)
            if set(items)&set(results[family]): raise ValueError('Duplicate held question')
            results[family].update(items); receipts[family].append(receipt)
            io.atomic_bytes(args.output/f'{family}-fold{fold}.pkl', pickle.dumps(model, protocol=5))
            print({'family': family, 'fold': fold, 'correct': sum(int(v['correct']) for v in items.values()), 'n': len(items)}, flush=True)
        if set(results[family]) != set(ids): raise ValueError('Incomplete common held panel')
        items = results[family]
        accuracy = np.asarray([int(items[k]['correct'])-int(control[k]['correct']) for k in ids])
        compute = np.asarray([items[k]['compute']-control[k]['compute'] for k in ids])
        reports[family] = {'metrics': previous.statistics(list(items.values())), 'accuracy_delta': float(accuracy.mean()),
            'compute_delta': float(compute.mean()), 'accuracy_delta_CI95': np.quantile(accuracy[draws].mean(1), [.025, .975]).tolist(),
            'screen_passed': bool(accuracy.mean() >= .02 and compute.mean() <= 0)}
    result = {'schema': SCHEMA, 'strongest_control': record['strongest_control'],
        'control_metrics': {k: previous.statistics(list(v.values())) for k, v in controls.items()},
        'reports': reports, 'common_questions': 768, 'new_neural_calls': 0, 'cloud_calls': 0,
        'development_or_test_read': False, 'fresh_validation': False}
    io.write(args.output/'private-held-items.json', results); io.write(args.output/'fold-receipts.json', receipts)
    io.write(args.output/'summary.json', result); io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'any_screen_passed': any(v['screen_passed'] for v in reports.values()),
        'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'diagnose'))
    for name in ('expanded-fit', 'control', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}, flush=True); return
        if not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256 or io.read(args.output/'registration.json') != record:
            raise ValueError('Changed difficulty experiment')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior fitting outputs')
            diagnose(args, *bundle)


if __name__ == '__main__': main()
