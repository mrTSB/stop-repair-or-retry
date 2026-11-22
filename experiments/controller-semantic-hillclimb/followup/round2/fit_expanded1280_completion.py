"""Nested TRAIN test of more paired data, with deployable fits only after screening.

Compare old and expanded forecasts on the identical original768 held questions.
The new512 also receive out-of-question predictions, reported separately.
The frozen encoder and four-setting forest procedure remain unchanged.
"""
import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import expanded1280_completion_data as dataset
import nonlinear_completion_forecast as forecast
import regularized_completion_pomdp as runtime
import clustered_completion_pomdp as clustered

io, fixed, nested = dataset.io, dataset.fixed, dataset.nested
SCHEMA = 'expanded1280-forest-Complete16-TRAIN-fit-v1'


def inputs(args):
    rows, trees, observer, data, old_data, evidence = dataset.inputs(args)
    spec = io.read(args.previous_diagnostic/'registration.json'); done = io.read(args.previous_diagnostic/'complete.json')
    if (spec['schema'] != 'expanded768-nonlinear-joint-belief-TRAIN-v1'
            or done['registration_sha256'] != io.sha(args.previous_diagnostic/'registration.json')
            or spec['paths']['features'] != str(args.first_features.resolve())):
        raise ValueError('Require the original768 nested forest diagnostic')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.previous_diagnostic/n): h for n, h in done['digests'].items()})
    old_fit = io.read(Path(spec['paths']['categorical'])/'registration.json')
    if (old_fit['records_sha256'] != fixed.fingerprint(old_data['records'])
            or old_fit['labels_sha256'] != fixed.array_sha(old_data['labels'])
            or old_fit['valid_sha256'] != fixed.array_sha(old_data['valid'])
            or old_fit['feature_sha256'] != {k: fixed.array_sha(v) for k, v in old_data['features'].items()}):
        raise ValueError('Old held-question comparison data changed')
    with np.load(args.previous_diagnostic/'private-oof.npz', allow_pickle=False) as archive:
        control = archive[nested.FAMILIES[1]].copy()
    if control.shape != (2304, 16): raise ValueError('All768 original held-question forecasts required')
    evidence.update({**spec['input_evidence'], **{str((args.previous_diagnostic/n).resolve()): h for n, h in done['digests'].items()},
        str((args.previous_diagnostic/'complete.json').resolve()): io.sha(args.previous_diagnostic/'complete.json')})
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {k: str(getattr(args, k).resolve()) for k in ('first_features', 'second_features', 'previous_diagnostic', 'output')},
        'training_questions': 1280, 'decision_nodes': 3840, 'common_comparison_questions': 768,
        'families': list(nested.FAMILIES), 'settings': list(forecast.SETTINGS), 'trees': forecast.TREES,
        'prior_weight': forecast.PRIOR_WEIGHT, 'outer_folds': 3, 'inner_folds': 3,
        'selection': 'Unchanged four-setting forest search minimizes inner question-held joint log loss',
        'screen': 'Per family, lower joint loss AND mean-four Brier on identical old768 held questions than retained dense forest',
        'new512_reporting': 'All excluded-question predictions retained; never an outcome-selected subset',
        'records_sha256': fixed.fingerprint(data['records']), 'labels_sha256': fixed.array_sha(data['labels']),
        'valid_sha256': fixed.array_sha(data['valid']), 'feature_sha256': {k: fixed.array_sha(v) for k, v in data['features'].items()},
        'bootstrap_draws': 2000, 'bootstrap_seed': 238031, 'smoothing': 32., 'state_count': 16,
        'maximum_solver_continuations': 2, 'new_neural_calls': 0, 'cloud_calls': 0,
        'additional_observation_partition': 'Same24valid/6invalid KMeans on nested TRAIN child forecasts; refit emissions only',
        'development_or_test_read': False, 'controller_accuracy_claim': False}
    return rows, trees, observer, data, control, record


def comparison(labels, old, new, records):
    groups = [[i for i, r in enumerate(records) if r['id'] == qid] for qid in sorted({r['id'] for r in records})]
    if len(groups) != 768 or any(len(g) != 3 for g in groups): raise ValueError('Compare all identical old768 groups')
    draws = np.random.default_rng(238031).integers(0, 768, (2000, 768))
    before, after = fixed.losses(labels, old), fixed.losses(labels, new); report = {}
    for metric in before:
        delta = np.asarray([(after[metric][g]-before[metric][g]).mean() for g in groups])
        report[metric] = {'retained768_dense_forest': float(before[metric].mean()), 'expanded1280': float(after[metric].mean()),
            'delta': float(delta.mean()), 'paired_question_CI95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()}
    return report


def fit(args, rows, trees, observer, data, control, record):
    models, reports, oof, folds, clustered_models = {}, {}, {}, {}, {}
    for family in nested.FAMILIES:
        p = np.full((3840, 16), np.nan); folds[family] = []
        for fold in range(3):
            held, values, model, receipt = forecast.outer_prediction(data, family, fold)
            p[held] = values; folds[family].append(receipt)
            io.atomic_bytes(args.output/(family+f'-outer{fold}.pkl'), pickle.dumps(model, protocol=5))
            print({'family': family, 'outer_fold_complete': fold, 'selected': receipt['selected']}, flush=True)
        if not np.isfinite(p).all(): raise ValueError('Incomplete nested training forecasts')
        oof[family] = p
        compared = comparison(data['labels'][:2304], control, p[:2304], data['records'][:2304])
        passed = compared['joint_log_loss']['delta'] < 0 and compared['mean_four_marginal_Brier']['delta'] < 0
        reports[family] = {'common768_comparisons': compared, 'screen_passed': bool(passed),
            'additional512': {k: float(v.mean()) for k, v in fixed.losses(data['labels'][2304:], p[2304:]).items()},
            'all1280': {k: float(v.mean()) for k, v in fixed.losses(data['labels'], p).items()}}
        if passed:
            selected, scores = forecast.select(data, family, np.ones(3840, dtype=bool))
            forecaster = forecast.fit(data['features'][family], data['labels'], selected)
            obs = runtime.DenseObserver(observer.directory, observer.temperature, calibration=observer.calibration) if family == nested.FAMILIES[1] else observer
            obs._recipe = observer._recipe
            policy = runtime.fit_dynamics(rows, trees, obs, data['records'], p, forecaster, family,
                {'registration_sha256': args.registration_sha256, 'outer_folds': folds[family],
                 'final_settings': selected, 'final_inner_search': scores, 'expanded1280': True})
            path = args.output/(family+'.pkl'); io.atomic_bytes(path, pickle.dumps(policy, protocol=5))
            models[family] = {'model_sha256': io.sha(path), 'fit_receipt': policy.fit_receipt}
            variant = clustered.rebin(policy, rows, trees, data['records'], p)
            path = args.output/(family+'-clustered.pkl'); io.atomic_bytes(path, pickle.dumps(variant, protocol=5))
            clustered_models[family] = {'model_sha256': io.sha(path), 'fit_receipt': variant.fit_receipt}
    io.npz(args.output/'private-oof.npz', oof); io.write(args.output/'fold-receipts.json', folds)
    io.write(args.output/'summary.json', {'schema': SCHEMA, 'reports': reports, 'models': models, 'clustered_models': clustered_models,
        'training_questions': 1280, 'controller_accuracy_claim': False, 'development_or_test_read': False,
        'paired_CI_scope': 'Descriptive, fitted folds fixed'})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'models': models, 'clustered_models': clustered_models, 'training_questions': 1280, 'any_screen_passed': bool(models), 'new_neural_calls': 0,
        'cloud_calls': 0, 'development_or_test_read': False,
        'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'complete': True, 'passed_families': list(models), 'reports': reports}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'fit'))
    for n in ('first_features', 'second_features', 'previous_diagnostic', 'output'): p.add_argument('--'+n.replace('_', '-'), type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered expanded training fit')
        with io.lock(args.output/'.fit.lock'):
            if any(p.name not in ('registration.json', '.fit.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior fit')
            fit(args, *bundle)


if __name__ == '__main__': main()
