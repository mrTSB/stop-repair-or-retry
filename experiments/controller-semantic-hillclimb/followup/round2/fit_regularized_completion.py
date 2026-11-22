"""Register and fit two TRAIN-only Complete16 policies from nested forecasts."""
import argparse
import pickle
from pathlib import Path
import sys
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

import nested_dense_diagnostic as nested
import regularized_completion_pomdp as runtime

io, fixed, extraction, probe = nested.io, nested.fixed, nested.extraction, nested.probe
SCHEMA = 'nested-regularized-Complete16-TRAIN-fit-v1'


def final_forecaster(data, family):
    """Select final hyperparameters only on TRAIN; emissions use nested OOF."""
    x, y, valid = data['features'][family], data['labels'], data['valid']
    folds = np.asarray([nested.inner_fold(r['id']) for r in data['records']])
    if set(folds) != {0, 1, 2}: raise ValueError('Require all final selection folds')
    results = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for settings in nested.candidates(family):
            oof = np.full((len(y), 16), np.nan)
            for fold in range(3):
                train, held = folds != fold, folds == fold
                nested.group_receipt(data['records'], train, held)
                model = nested.learner(x[train], y[train], settings)
                oof[held] = nested.predict(model, x[held], valid[held])
            if not np.isfinite(oof).all(): raise ValueError('Incomplete final TRAIN hyperparameter selection')
            loss = fixed.losses(y, oof)
            results.append({'settings': settings, 'training_selection_joint_loss': float(loss['joint_log_loss'].mean())})
        index = min(range(len(results)), key=lambda i: (results[i]['training_selection_joint_loss'], i))
        selected = results[index]['settings']
        model = nested.learner(x, y, selected)
    return model, {'selected': selected, 'search': results,
        'scores_are_training_selection_not_validation': True,
        'emissions_use_separate_nested_OOF': True, 'warnings': [str(w.message) for w in captured]}


def inputs(args):
    registration = io.read(args.nested / 'registration.json')
    complete = io.read(args.nested / 'complete.json')
    if registration['schema'] != nested.SCHEMA or complete['schema'] != nested.SCHEMA or not complete['screen_passed']:
        raise ValueError('Require the completed successful separate nested TRAIN diagnostic')
    io.verify(registration['source_files']); io.verify(registration['input_evidence'])
    io.verify({str(args.nested / n): v for n, v in complete['digests'].items()})
    if registration['paths']['base_fit'] != str(args.base_fit.resolve()) or registration['paths']['neural'] != str(args.neural.resolve()):
        raise ValueError('Nested diagnostic belongs to another training feature archive')
    bundle = extraction.training_bundle(args.base_fit)
    values, _ = extraction.verify_complete(args.neural, io.sha(args.neural / 'registration.json'), bundle)
    data = probe.table(bundle['rows'], bundle['trees'], bundle['metadata'], values)
    if (registration['records_sha256'] != fixed.fingerprint(data['records'])
            or registration['labels_sha256'] != fixed.array_sha(data['labels'])
            or registration['valid_sha256'] != fixed.array_sha(data['valid'])
            or registration['feature_sha256'] != {k: fixed.array_sha(v) for k, v in data['features'].items()}):
        raise ValueError('Nested TRAIN features, labels or alignment changed')
    with np.load(args.nested / 'private-oof.npz', allow_pickle=False) as archive:
        forecasts = {name: archive[name].copy() for name in nested.FAMILIES}
    evidence = {**registration['input_evidence'],
        **{str((args.nested / n).resolve()): v for n, v in complete['digests'].items()},
        str((args.nested / 'complete.json').resolve()): io.sha(args.nested / 'complete.json'),
        str((args.neural / 'neural-complete.json').resolve()): io.sha(args.neural / 'neural-complete.json')}
    spec = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {n: str(getattr(args, n).resolve()) for n in ('base_fit', 'neural', 'nested', 'output')},
        'families': list(nested.FAMILIES), 'questions': 256, 'decision_nodes': 768,
        'hyperparameters': {n: nested.candidates(n) for n in nested.FAMILIES},
        'final_selection': 'Whole-question3fold TRAIN minimum joint loss; deterministic candidate order ties',
        'emission_forecasts': 'Previously sealed nested outer-held predictions; never final-selection OOF',
        'smoothing': 32., 'state_count': 16, 'exact_finite_Bellman': True,
        'new_neural_calls': 0, 'cloud_calls': 0, 'development_or_test_read': False,
        'controller_accuracy_claim': False, 'runtime_module': 'regularized_completion_pomdp'}
    return bundle, data, forecasts, spec


def fit(args, bundle, data, forecasts, spec):
    with io.lock(args.output / '.fit.lock'), threadpool_limits(limits=1):
        if (not args.registration_sha256 or io.sha(args.output / 'registration.json') != args.registration_sha256
                or io.read(args.output / 'registration.json') != spec):
            raise ValueError('Explicit registered fit changed')
        if any(p.name not in ('registration.json', '.fit.lock') for p in args.output.iterdir()):
            raise ValueError('Preserve prior fit outputs')
        models = {}
        for family in nested.FAMILIES:
            forecaster, receipt = final_forecaster(data, family)
            original = bundle['observer']
            if family == nested.FAMILIES[1]:
                observer = runtime.DenseObserver(original.directory, original.temperature, calibration=original.calibration)
                observer._recipe = original._recipe.copy()
            else:
                observer = original
            model = runtime.fit_dynamics(bundle['rows'], bundle['trees'], observer, data['records'], forecasts[family],
                forecaster, family, {'registration_sha256': args.registration_sha256,
                'nested_complete_sha256': io.sha(args.nested / 'complete.json'), 'final_forecaster': receipt})
            path = args.output / (family + '.pkl'); io.atomic_bytes(path, pickle.dumps(model, protocol=5))
            models[family] = {'model_sha256': io.sha(path), 'fit_receipt': model.fit_receipt}
            print({'fitted': family, 'selected': receipt['selected']}, flush=True)
        io.verify(spec['source_files']); io.verify(spec['input_evidence'])
        io.write(args.output / 'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
            'models': models, 'training_questions': 256, 'new_neural_calls': 0, 'cloud_calls': 0,
            'development_or_test_read': False, 'controller_accuracy_claim': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'register', 'fit'))
    for n in ('base-fit', 'neural', 'nested', 'output'): parser.add_argument('--' + n, type=Path, required=True)
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls():
        bundle, data, forecasts, spec = inputs(args)
        if args.phase == 'plan':
            print({k: spec[k] for k in ('schema', 'questions', 'families', 'new_neural_calls')}); return
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output / 'registration.json', spec)
            print({'registration_sha256': io.sha(args.output / 'registration.json'), 'fit_started': False}); return
        fit(args, bundle, data, forecasts, spec)


if __name__ == '__main__': main()
