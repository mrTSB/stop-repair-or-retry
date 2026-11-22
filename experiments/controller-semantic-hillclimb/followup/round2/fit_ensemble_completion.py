"""Fixed half-linear/half-forest TRAIN screen and finite POMDP fit.

No ensemble-weight tuning, neural pass, cloud call or DEV/TEST read.
Both components' forecasts exclude each question from fitting and tuning.
"""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np

import fit_nonlinear_completion as source
import ensemble_completion_pomdp as runtime

io, fixed, nested = source.io, source.fixed, source.nested
SCHEMA = 'expanded768-half-ensemble-Complete16-fit-v1'
FAMILY = nested.FAMILIES[1]


def inputs(args):
    spec = io.read(args.nonlinear/'registration.json'); done = io.read(args.nonlinear/'complete.json')
    old_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    rows, trees, observer, data, nonlinear_oof, expected = source.inputs(old_args)
    if expected != spec or done['schema'] != source.SCHEMA or done['registration_sha256'] != io.sha(args.nonlinear/'registration.json'):
        raise ValueError('Nonlinear fit changed')
    io.verify(spec['source_files']); io.verify(spec['input_evidence']); io.verify({str(args.nonlinear/n): h for n, h in done['digests'].items()})
    diagnostic = io.read(old_args.diagnostic/'registration.json')
    categorical = Path(diagnostic['paths']['categorical']); completed = io.read(categorical/'complete.json')
    registered = io.read(categorical/'registration.json')
    if completed['registration_sha256'] != io.sha(categorical/'registration.json'): raise ValueError('Categorical seal changed')
    io.verify(registered['source_files']); io.verify(registered['input_evidence']); io.verify({str(categorical/n): h for n, h in completed['digests'].items()})
    models = [pickle.loads((folder/(FAMILY+'.pkl')).read_bytes()) for folder in (categorical, args.nonlinear)]
    if any(m.feature_family != FAMILY for m in models): raise ValueError('Different feature families')
    with np.load(categorical/'private-oof.npz', allow_pickle=False) as archive: linear_oof = archive[FAMILY].copy()
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], **registered['input_evidence'],
            **{str((args.nonlinear/n).resolve()): h for n, h in done['digests'].items()},
            **{str((categorical/n).resolve()): h for n, h in completed['digests'].items()},
            str((args.nonlinear/'complete.json').resolve()): io.sha(args.nonlinear/'complete.json'),
            str((categorical/'complete.json').resolve()): io.sha(categorical/'complete.json')},
        'paths': {n: str(getattr(args, n).resolve()) for n in ('nonlinear', 'output')},
        'training_questions': 768, 'family': FAMILY, 'weights': [.5, .5], 'weight_selection': False,
        'screen': 'Joint log loss and mean-four Brier both lower than retained forest', 'smoothing': 32.,
        'state_count': 16, 'one_shared_encoder_forward': True, 'new_neural_calls': 0, 'cloud_calls': 0,
        'development_or_test_read': False, 'controller_accuracy_claim': False}
    return rows, trees, data, models, linear_oof, nonlinear_oof[FAMILY], record


def fit(args, rows, trees, data, models, linear, forest, record):
    mixed = .5*(linear+forest); old = fixed.losses(data['labels'], forest); new = fixed.losses(data['labels'], mixed)
    groups = [[i for i, r in enumerate(data['records']) if r['id'] == qid] for qid in sorted({r['id'] for r in data['records']})]
    draws = np.random.default_rng(238031).integers(0, len(groups), (2000, len(groups))); report = {}
    for key in old:
        delta = np.array([(new[key][g]-old[key][g]).mean() for g in groups])
        report[key] = {'forest': float(old[key].mean()), 'ensemble': float(new[key].mean()), 'delta': float(delta.mean()),
                      'paired_question_CI95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()}
    passed = report['joint_log_loss']['delta'] < 0 and report['mean_four_marginal_Brier']['delta'] < 0
    results = {}
    if passed:
        fitted = source.runtime.fit_dynamics(rows, trees, models[1].classifier, data['records'], mixed,
            tuple(m.forecaster for m in models), FAMILY,
            {'registration_sha256': args.registration_sha256, 'fixed_ensemble_weights': [.5, .5], 'both_components_nested_question_excluded': True})
        model = runtime.Planner(**vars(fitted)); path = args.output/(FAMILY+'.pkl')
        io.atomic_bytes(path, pickle.dumps(model, protocol=5)); results[FAMILY] = {'model_sha256': io.sha(path), 'fit_receipt': model.fit_receipt}
    io.write(args.output/'summary.json', {'schema': SCHEMA, 'reports': report, 'screen_passed': passed,
        'paired_CI_scope': 'Descriptive, fitted folds fixed', 'controller_accuracy_claim': False})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'models': results, 'training_questions': 768, 'screen_passed': passed, 'new_neural_calls': 0, 'cloud_calls': 0,
        'development_or_test_read': False, 'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'screen_passed': passed, 'reports': report}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'fit'))
    for n in ('nonlinear', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls():
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Registered ensemble changed')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior fits')
            fit(args, *bundle)


if __name__ == '__main__': main()
