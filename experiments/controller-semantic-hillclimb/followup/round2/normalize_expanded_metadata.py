"""Archive a prediction-identical correction of tiny constant-feature variances.

The original expanded fit is immutable. Weighted StandardScaler can store a
negative roundoff residual for an exactly constant feature, while already
using scale1. Only such variances in[-1e-30,0) are replaced by0. Every learned
mean/scale/coefficient/intercept/temperature and every transformed feature
and prediction must remain exactly equal. No refit or held-out data is used.
"""
import argparse
import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import expanded_mentor_io as io
import fit_expanded_mentor as fitted
import metadata_paid_audit as paid_audit
import paid_review_metadata as paid

SCHEMA = 'round2-expanded-metadata-variance-normalization-v1'
TOLERANCE = 1e-30


def normalize_adapter(adapter, features):
    adapter.validate()
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != 32 or not np.isfinite(x).all():
        raise ValueError('Require finite unchanged32-column TRAIN metadata inputs')
    original = copy.deepcopy(adapter)
    state = original.state(); scale = original.fitted['model']['scale']
    negative = np.flatnonzero(scale.var_ < 0)
    if (np.any(scale.var_[negative] < -TOLERANCE)
            or not np.array_equal(scale.scale_[negative], np.ones(len(negative)))
            or any(np.ptp(x[:, i]) != 0 for i in negative)):
        raise ValueError('Negative variance is not bounded roundoff on an exact scale1 constant feature')
    before_x = scale.transform(x)
    before_p = original.fitted['model'].predict_proba(x)
    records = [{'features': list(row), 'answer_present': True} for row in x]
    before_calibrated = paid.diagnostic.calibrated(paid.diagnostic.predict(original.fitted, records),
                                                   original.temperature, records)
    result = copy.deepcopy(original)
    result.fitted['model']['scale'].var_[negative] = 0.
    updated = result.state()
    for name in ('features', 'indexes', 'mean', 'scale', 'classes', 'coefficients', 'intercepts', 'temperature'):
        if state[name] != updated[name]: raise ValueError('A numerical learned prediction parameter changed')
    after_x = result.fitted['model']['scale'].transform(x)
    after_p = result.fitted['model'].predict_proba(x)
    after_calibrated = paid.diagnostic.calibrated(paid.diagnostic.predict(result.fitted, records),
                                                  result.temperature, records)
    if any(not np.array_equal(a, b) for a, b in (
            (before_x, after_x), (before_p, after_p), (before_calibrated, after_calibrated))):
        raise ValueError('Variance normalization changed transformed features or predictions')
    proof = {'schema': SCHEMA, 'tolerance': TOLERANCE, 'changed_indexes': negative.tolist(),
        'old_variances': scale.var_[negative].tolist(), 'new_variances': [0.]*len(negative),
        'original_learned_state_sha256': paid.digest(state), 'normalized_learned_state_sha256': paid.digest(updated),
        'mean_scale_coefficients_intercepts_temperature_unchanged': True,
        'transformed_features_identical': True, 'raw_predictions_identical': True,
        'calibrated_predictions_identical': True, 'training_feature_rows': len(x),
        'features_sha256': fitted.core.array_digest(x),
        'transformed_sha256': fitted.core.array_digest(before_x),
        'raw_predictions_sha256': fitted.core.array_digest(before_p),
        'calibrated_predictions_sha256': fitted.core.array_digest(before_calibrated),
        'labels_used_for_change': False, 'refitted': False, 'development_or_test_used': False}
    result.receipt = {**result.receipt, 'learned_state_sha256': paid.digest(updated),
                      'constant_variance_normalization': proof}
    result.validate(); paid_audit.validate_adapter_state(result.state(), result.receipt)
    if original.state() != state or adapter.state() != state:
        raise ValueError('Original adapter was mutated')
    return result, proof


def protected(model):
    return io.structural_digest((model.base, model.mentor, model.transfer, model.base_module, model.base_settings))


def source_inputs(parent):
    registration_sha = io.sha(parent/'registration.json')
    completed = fitted.verify_complete(parent, registration_sha)
    if (completed['paired_review_questions'], completed['metadata_records'], completed['metadata_questions']) != (320, 448, 320):
        raise ValueError('Wrong expanded source cohort')
    records = [json.loads(line) for line in (parent/'private-metadata-features.jsonl').read_text().splitlines()]
    if len(records) != 448: raise ValueError('Incomplete original TRAIN feature matrix')
    x = np.asarray([r['features'] for r in records], dtype=float)
    return completed, x


def registration(args, completed):
    return {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'parent': str(args.parent.resolve()), 'output': str(args.output.resolve()),
        'parent_complete_sha256': io.sha(args.parent/'complete.json'),
        'parent_registration_sha256': io.sha(args.parent/'registration.json'),
        'parent_feature_sha256': io.sha(args.parent/'private-metadata-features.jsonl'),
        'parent_raw_nested_oof_sha256': io.sha(args.parent/'metadata-oof.npz'),
        'metadata_models': [r for r in completed['models'] if r['metadata_review']],
        'tolerance': TOLERANCE, 'neural_calls': 0, 'cloud_calls': 0, 'refitted': False,
        'development_or_test_used': False, 'historical_byte_identity_claim': False}


def verify_complete(directory):
    contract = io.read(directory/'registration.json'); complete = io.read(directory/'complete.json')
    parent = Path(contract['parent'])
    original, _ = source_inputs(parent)
    if (contract['schema'] != SCHEMA or complete['schema'] != SCHEMA
            or complete['registration_sha256'] != io.sha(directory/'registration.json')
            or contract['parent_complete_sha256'] != io.sha(parent/'complete.json')
            or complete['parent_complete_sha256'] != contract['parent_complete_sha256']
            or len(complete['models']) != 6 or complete['refitted'] or complete['development_or_test_used']):
        raise ValueError('Changed expanded normalization completion')
    io.verify(contract['source_files'])
    io.verify({str(directory/name): value for name, value in complete['artifacts'].items()})
    parent_models = {r['key']: r for r in original['models']}
    if set(parent_models) != {r['key'] for r in complete['models']}:
        raise ValueError('Normalization omitted an original candidate')
    for record in complete['models']:
        path = Path(record['model_path'])
        if (io.sha(path) != record['sha256'] or record['parent_sha256'] != parent_models[record['key']]['sha256']
            or io.structural_digest(pickle.loads(path.read_bytes())) != record['saved_structural_sha256']):
            raise ValueError('Changed normalized or retained model')
        if record['metadata_review']:
            model = pickle.loads(path.read_bytes())
            paid_audit.validate_adapter_state(model.metadata_adapter.state(), model.metadata_adapter.receipt)
    return complete


def run(args):
    with fitted.inputs.parent_fit.no_neural_inference(), threadpool_limits(limits=1):
        completed, x = source_inputs(args.parent)
        contract = registration(args, completed)
        if args.mode == 'register':
            if args.output.exists(): raise ValueError('Use a new normalization output')
            args.output.mkdir(parents=True); io.write(args.output/'registration.json', contract)
            print({'registered': True, 'registration_sha256': io.sha(args.output/'registration.json')}, flush=True)
            return
        if not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256:
            raise ValueError('Require explicit reviewed normalization registration SHA')
        if io.read(args.output/'registration.json') != contract:
            raise ValueError('Changed registered normalization inputs/source')
        with io.lock(args.output/'.normalize.lock'):
            if (args.output/'complete.json').exists(): verify_complete(args.output); return
            adapter = pickle.loads((args.parent/'adapter.pkl').read_bytes())
            normalized, proof = normalize_adapter(adapter, x)
            fitted.save_model(args.output/'adapter.pkl', normalized)
            fitted.save_json(args.output/'trained-linear.json', {'state': normalized.state(), 'receipt': normalized.receipt})
            models, proofs = [], {}
            for record in completed['models']:
                source = args.parent/(record['key']+'.pkl')
                if not record['metadata_review']:
                    models.append({**record, 'model_path': str(source.resolve()), 'parent_sha256': record['sha256'],
                                   'normalization_applied': False})
                    continue
                parent = pickle.loads(source.read_bytes()); before = protected(parent)
                model = copy.deepcopy(parent)
                model.metadata_adapter, item_proof = normalize_adapter(parent.metadata_adapter, x)
                if item_proof != proof: raise ValueError('Metadata candidates do not share the same original fitted head')
                model.fit_receipt['metadata_adapter'] = model.metadata_adapter.receipt
                model.fit_receipt['variance_normalization_registration_sha256'] = args.registration_sha256
                if protected(model) != before or protected(parent) != before:
                    raise ValueError('A non-adapter hybrid component changed')
                path = args.output/(record['key']+'.pkl')
                saved = fitted.save_model(path, model)
                models.append({**record, **saved, 'model_path': str(path.resolve()),
                    'parent_sha256': record['sha256'], 'normalization_applied': True,
                    'protected_state_sha256': before})
                proofs[record['key']] = item_proof
            fitted.save_json(args.output/'normalization-proof.json', {'adapter': proof, 'models': proofs,
                'raw_nested_oof_untouched_sha256': io.sha(args.parent/'metadata-oof.npz')})
            io.verify(contract['source_files'])
            if io.sha(args.parent/'complete.json') != contract['parent_complete_sha256']:
                raise ValueError('Original fit changed')
            result = {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
                'parent_directory': str(args.parent.resolve()), 'parent_complete_sha256': contract['parent_complete_sha256'],
                'models': models, 'regular_A4_artifact': {**completed['regular_A4_artifact'],
                    'model_path': str((args.parent/(completed['regular_A4_artifact']['key']+'.pkl')).resolve())},
                'base_A': completed['base_A'], 'paired_review_questions': 320, 'metadata_records': 448,
                'refitted': False, 'neural_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False,
                'artifacts': {p.name: io.sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and not p.name.startswith('.')}}
            io.write(args.output/'complete.json', result); verify_complete(args.output)
            print({'complete': True, 'changed_variance_indexes': proof['changed_indexes'],
                   'predictions_unchanged': True}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'normalize'))
    for name in ('parent', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256')
    run(parser.parse_args())


if __name__ == '__main__': main()
