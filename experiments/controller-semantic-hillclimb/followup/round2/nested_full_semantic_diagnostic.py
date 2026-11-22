"""Separate TRAIN-only nested regularization/compression study of sealed features.

The earlier fixed-C dense diagnostic remains a failed experiment. This new
study tests whether its dimensionality/regularization caused overfitting. Both
control and enriched procedures select C using inner question-held folds;
Both feature procedures use identical three-C searches without PCA. No DEV/test selection,
encoder calls, deployed policy, or retrospective replacement of old results.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import extract_full_semantic_completion as extraction
import full_semantic_completion_probe as probe

fixed, io = extraction.fixed, extraction.io
SCHEMA = 'TRAIN-completion-nested-full1024-v1'
CS = (.001, .01, .1)
FAMILIES = probe.VARIANTS
INNER_SALT = 'completion-nested-dense-inner-v1'


def inner_fold(qid):
    return int(hashlib.sha256((INNER_SALT + ':' + qid).encode()).hexdigest()[:8], 16) % 3


def candidates(family):
    if family not in FAMILIES:
        raise ValueError('Only the two registered feature procedures')
    return [{'C': c, 'pca_components': None} for c in CS]


def learner(x, y, settings):
    if len(np.unique(y)) == 1:
        return DummyClassifier(strategy='prior').fit(x, y)
    if settings['pca_components'] is not None: raise ValueError('No PCA in this registered study')
    return make_pipeline(StandardScaler(), LogisticRegression(C=settings['C'], max_iter=2000)).fit(x, y)


def predict(model, x, valid):
    probability = np.full((len(x), 16), 1e-9)
    probability[:, model.classes_.astype(int)] += model.predict_proba(x)
    return np.asarray([fixed.core.mask_invalid(p, v) for p, v in zip(probability, valid)])


def group_receipt(records, train, held):
    a = {r['id'] for r, flag in zip(records, train) if flag}
    b = {r['id'] for r, flag in zip(records, held) if flag}
    if not a or not b or a & b:
        raise ValueError('Training and held question groups overlap or are empty')
    return {'training_questions': len(a), 'held_questions': len(b),
            'training_ids_sha256': fixed.fingerprint(sorted(a)), 'held_ids_sha256': fixed.fingerprint(sorted(b))}


def outer_prediction(data, family, fold):
    records = data['records']
    held = np.asarray([r['fold'] == fold for r in records])
    outer_train = ~held
    receipt = group_receipt(records, outer_train, held)
    inner = np.asarray([inner_fold(r['id']) for r in records])
    if set(inner[outer_train]) != {0, 1, 2}:
        raise ValueError('All three separately salted inner question folds are required')
    x, y, valid = data['features'][family], data['labels'], data['valid']
    searches = []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for settings in candidates(family):
            oof = np.full((len(y), 16), np.nan)
            folds = []
            for inner_value in range(3):
                train = outer_train & (inner != inner_value)
                validation = outer_train & (inner == inner_value)
                folds.append({'inner_fold': inner_value, **group_receipt(records, train, validation)})
                model = learner(x[train], y[train], settings)
                oof[validation] = predict(model, x[validation], valid[validation])
            if not np.isfinite(oof[outer_train]).all() or np.isfinite(oof[held]).any():
                raise ValueError('Inner selection must cover only outer-training questions')
            scores = fixed.losses(y[outer_train], oof[outer_train])
            searches.append({'settings': settings, 'inner_joint_log_loss': float(scores['joint_log_loss'].mean()),
                'inner_mean_four_marginal_Brier': float(scores['mean_four_marginal_Brier'].mean()), 'folds': folds})
        # Fixed order breaks exact ties. No outer-held outcome enters selection.
        selected = min(range(len(searches)), key=lambda i: (searches[i]['inner_joint_log_loss'], i))
        settings = searches[selected]['settings']
        model = learner(x[outer_train], y[outer_train], settings)
        probability = predict(model, x[held], valid[held])
    return held, probability, model, {'outer_fold': fold, **receipt, 'selected': settings,
        'inner_search': searches, 'fit_and_predict_seconds': time.perf_counter() - started,
        'warnings': [str(w.message) for w in captured], 'outer_held_labels_used_for_fit_or_selection': False}


def inputs(args):
    original = io.read(args.neural / 'registration.json')
    if original['base_fit'] != str(args.base_fit.resolve()):
        raise ValueError('Dense extraction belongs to another completed TRAIN fit')
    io.verify(original['source_files']); io.verify(original['input_evidence'])
    bundle = extraction.training_bundle(args.base_fit)
    values, complete = extraction.verify_complete(args.neural, io.sha(args.neural / 'registration.json'), bundle)
    data = probe.table(bundle['rows'], bundle['trees'], bundle['metadata'], values)
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': original['input_evidence'],
        'neural_complete_sha256': io.sha(args.neural / 'neural-complete.json'),
        'neural_registration_sha256': io.sha(args.neural / 'registration.json'),
        'paths': {k: str(getattr(args, k).resolve()) for k in ('base_fit', 'neural', 'output')},
        'records_sha256': fixed.fingerprint(data['records']), 'labels_sha256': fixed.array_sha(data['labels']),
        'valid_sha256': fixed.array_sha(data['valid']),
        'feature_sha256': {k: fixed.array_sha(v) for k, v in data['features'].items()},
        'questions': 256, 'nodes': 768, 'families': {name: candidates(name) for name in FAMILIES},
        'outer_folds': 'Original SHA256(question_id) modulo3, all three prefixes grouped',
        'inner_salt': INNER_SALT, 'inner_folds': 3,
        'selection': 'Minimum inner question-held joint loss; fixed candidate order breaks exact ties',
        'PCA': 'None; scaler fits only on the training fold',
        'outer_test_scaler_or_PCA_fit': False, 'temperature': 1., 'max_iter': 2000,
        'screen': 'Full1024 procedure joint loss AND mean4-marginal Brier lower than compressed256 procedure',
        'screen_scope': 'TRAIN diagnostic only; paired question CIs descriptive with fitted folds fixed',
        'old_failed_fixed_C_dense_result_preserved': True,
        'new_neural_calls': 0, 'cloud_calls': 0, 'development_or_test_read': False,
        'runtime_or_policy_selected': False, 'environment': {n: importlib.metadata.version(n)
            for n in ('numpy', 'scikit-learn', 'torch', 'transformers')}}
    return data, record


def run(args, data, record):
    with io.lock(args.output / '.diagnostic.lock'):
        return run_locked(args, data, record)


def run_locked(args, data, record):
    if (not args.registration_sha256 or io.sha(args.output / 'registration.json') != args.registration_sha256
            or io.read(args.output / 'registration.json') != record):
        raise ValueError('Explicit registered source, cohort or nested procedure changed')
    if any(p.name not in ('registration.json', '.diagnostic.lock') for p in args.output.iterdir()):
        raise ValueError('Preserve previous diagnostic outputs; no automatic replay')
    predictions, receipts = {}, {}
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            predictions[family] = np.full((768, 16), np.nan); receipts[family] = []
            for fold in range(3):
                held, probability, model, receipt = outer_prediction(data, family, fold)
                predictions[family][held] = probability
                name = f'{family}-outer{fold}.pkl'
                io.atomic_bytes(args.output / name, pickle.dumps(model, protocol=5))
                receipts[family].append({**receipt, 'model_sha256': io.sha(args.output / name)})
                print({'family': family, 'completed_outer_fold': fold, 'selected': receipt['selected']}, flush=True)
            if not np.isfinite(predictions[family]).all():
                raise ValueError('All held outer predictions must complete before comparison')
        report = probe.diagnostics(data, predictions)
        report.pop('neural_refitting_or_nested_CV_claim')
        report['nested_hyperparameter_selection'] = True
        report['prior_fixed_C_dense_failure_replaced'] = False
        report['neural_refitting'] = False
        report['nested_forecaster_model_selection'] = True
        io.npz(args.output / 'private-oof.npz', predictions)
        io.write(args.output / 'fold-receipts.json', receipts)
        io.write(args.output / 'summary.json', {'schema': SCHEMA, 'reports': report,
            'outer_selected_settings': {f: [v['selected'] for v in r] for f, r in receipts.items()},
            'fit_warnings': {f: [v['warnings'] for v in r] for f, r in receipts.items()},
            'questions': 256, 'nodes': 768, 'new_neural_calls': 0, 'cloud_calls': 0,
            'development_or_test_read': False, 'runtime_or_policy_selected': False})
        io.verify(record['source_files']); io.verify(record['input_evidence'])
        io.write(args.output / 'complete.json', {'schema': SCHEMA,
            'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file()},
            'screen_passed': report['screen_passed'], 'new_neural_calls': 0, 'cloud_calls': 0,
            'development_or_test_read': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'register', 'run'))
    for name in ('base-fit', 'neural', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls():
        data, record = inputs(args)
        if args.phase == 'plan':
            print({k: record[k] for k in ('schema', 'questions', 'nodes', 'families')}); return
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output / 'registration.json', record)
            print({'registration_sha256': io.sha(args.output / 'registration.json'), 'fit_started': False}); return
        run(args, data, record)


if __name__ == '__main__':
    main()
