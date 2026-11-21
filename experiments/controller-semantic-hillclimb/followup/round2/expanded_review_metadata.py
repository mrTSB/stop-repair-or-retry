"""New-source448-row paid metadata fit with question/subject-balanced weights.

All320 root reviews and all128 existing science siblings are retained. The
old64 questions are excluded uniformly. The existing32 paid features and
runtime adapter are unchanged; this new calibration contract is explicit.
No neural, cloud, development or test operation is available in this module.
"""
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import expanded_mentor_spec as weights
import review_metadata_spec as features
import review_metadata_diagnostic as inherited
from paid_review_metadata import Adapter, digest, signal
from prepare_mixed_mentor import fingerprint

SCHEMA = 'round2-expanded-newsource-paid-metadata-v1'


def validate(records):
    if len(records) != 448 or len({r['id'] for r in records}) != 320:
        raise ValueError('Require exactly448 records on320 new-source questions')
    if Counter(r['protocol'] for r in records) != {'root': 320, 'critical': 64, 'independent': 64}:
        raise ValueError('Changed root/critical/independent coverage')
    roots = {r['id']: r for r in records if r['protocol'] == 'root'}
    if Counter(r['subject'] for r in roots.values()) != weights.REVIEW_COUNTS:
        raise ValueError('Changed320-question observed subject mix')
    if len({(r['id'], r['protocol']) for r in records}) != 448:
        raise ValueError('Duplicate review sibling')
    critical = {r['id'] for r in records if r['protocol'] == 'critical'}
    independent = {r['id'] for r in records if r['protocol'] == 'independent'}
    if critical != independent or any(roots[q]['subject'] == 'math' for q in critical):
        raise ValueError('Require both siblings on the same64 science questions')
    for row in records:
        if (row['bank'] != 'new' or row['subject'] != roots[row['id']]['subject']
            or row['target'] not in (0, 1) or len(row['features']) != len(features.FEATURES)
            or not np.isfinite(row['features']).all() or (row['target'] and not row['answer_present'])):
            raise ValueError('Invalid metadata target/feature/observed subject')


def merge_records(parent_records, bundle):
    """Input parent diagnostic has already passed its frozen source audit."""
    if (len(parent_records) != 320 or len({r['id'] for r in parent_records}) != 192
        or Counter(r['protocol'] for r in parent_records) != {'root': 192, 'critical': 64, 'independent': 64}):
        raise ValueError('Require the original complete320/192 diagnostic')
    old = [r for r in parent_records if r['bank'] == 'old']
    if len(old) != 64 or any(r['protocol'] != 'root' for r in old):
        raise ValueError('Uniform old64 exclusion changed')
    records = [dict(r) for r in parent_records if r['bank'] == 'new']
    if len(records) != 256 or len({r['id'] for r in records}) != 128:
        raise ValueError('Original new-source128 diagnostic subset changed')
    roots = {r['id']: r for r in records if r['protocol'] == 'root'}
    rows = {r['id']: r for r in bundle['rows']}
    paired = {qid for qid, path in bundle['judges'] if path == ''}
    if not set(roots) <= paired or len(paired - set(roots)) != 192:
        raise ValueError('Require exactly192 additional root reviews')
    for qid in sorted(paired):
        row, current, judge = rows[qid], bundle['trees'][qid][''], bundle['judges'][qid, '']
        entry = {'id': qid, 'subject': row['subject'], 'bank': 'new', 'protocol': 'root',
            'features': inherited.features({**row, 'answer': ''}, current, judge, 'root').tolist(),
            'target': int(inherited.correct(judge, row['answer'])), 'answer_present': bool(judge['answer']),
            'review_sha256': fingerprint(judge), 'current_sha256': fingerprint(current)}
        if qid in roots:
            if roots[qid] != entry:
                raise ValueError('Original paid root feature/label alignment changed')
        else:
            if row['subject'] == 'math':
                raise ValueError('Expansion may add only the registered science questions')
            records.append(entry)
    records.sort(key=lambda r: (r['id'], ('root', 'critical', 'independent').index(r['protocol'])))
    validate(records)
    return records


def train_model(records):
    y = np.asarray([r['target'] for r in records], dtype=int)
    w = np.asarray(weights.question_weights(records))
    indexes = list(range(len(features.FEATURES)))
    weighting = weights.weight_receipt(records, w.tolist())
    if len(set(y.tolist())) == 1:
        return {'name': 'metadata_logistic', 'constant': True, 'indexes': indexes,
                'prior': float((w @ y + 1.) / (w.sum()+2.)), 'fit_weights': weighting}
    x = np.asarray([r['features'] for r in records], dtype=float)
    model = Pipeline([('scale', StandardScaler()), ('model', LogisticRegression(
        C=1., max_iter=1000, random_state=features.SEED))])
    model.fit(x, y, scale__sample_weight=w, model__sample_weight=w)
    return {'name': 'metadata_logistic', 'model': model, 'indexes': indexes, 'fit_weights': weighting}


def temperature_search(records, probabilities):
    w = weights.question_weights(records)
    search = [{'temperature': t, 'loss': inherited.binary_loss([r['target'] for r in records],
                inherited.calibrated(probabilities, t, records), w)} for t in features.TEMPERATURES]
    return min(search, key=lambda r: (r['loss'], r['temperature']))['temperature'], search


def outer_prediction(records, fold):
    training = [r for r in records if inherited.fold(r['id']) != fold]
    held = [r for r in records if inherited.fold(r['id']) == fold]
    train_ids, held_ids = {r['id'] for r in training}, {r['id'] for r in held}
    if not train_ids or not held_ids or train_ids & held_ids:
        raise ValueError('Invalid whole-question outer fold')
    predictions = np.full(len(training), np.nan)
    inner_receipts = []
    inner_models = []
    for inner in range(3):
        fitting = [r for r in training if inherited.fold(r['id'], True) != inner]
        indexes = [i for i, r in enumerate(training) if inherited.fold(r['id'], True) == inner]
        validation = [training[i] for i in indexes]
        fit_ids, validation_ids = {r['id'] for r in fitting}, {r['id'] for r in validation}
        if not fit_ids or not validation_ids or fit_ids & validation_ids or (fit_ids | validation_ids) & held_ids:
            raise ValueError('Outer-held question reached metadata temperature selection')
        model = train_model(fitting)
        predictions[indexes] = inherited.predict(model, validation)
        inner_models.append(model)
        inner_receipts.append({'fold': inner, 'training_ids': sorted(fit_ids),
            'held_ids': sorted(validation_ids), 'fit_weights': model['fit_weights'],
            'training_records_sha256': fingerprint(fitting), 'held_records_sha256': fingerprint(validation)})
    if not np.isfinite(predictions).all():
        raise ValueError('Incomplete inner held-question predictions')
    temperature, search = temperature_search(training, predictions)
    model = train_model(training)
    raw = inherited.predict(model, held)
    nested = inherited.calibrated(raw, temperature, held)
    receipt = {'outer_fold': fold, 'training_ids': sorted(train_ids), 'held_ids': sorted(held_ids),
        'fit_weights': model['fit_weights'], 'temperature': temperature, 'temperature_search': search,
        'inner': inner_receipts, 'inner_raw_predictions_sha256': fingerprint(predictions.tolist()),
        'outer_raw_predictions_sha256': fingerprint(raw.tolist()),
        'nested_predictions_sha256': fingerprint(nested.tolist())}
    return held, raw, nested, model, inner_models, receipt


def fit(records, registration_sha256):
    validate(records)
    lookup = {(r['id'], r['protocol']): i for i, r in enumerate(records)}
    raw, nested = np.full(len(records), np.nan), np.full(len(records), np.nan)
    receipts, outer_models, inner_models = [], [], []
    for fold in range(3):
        held, p, calibrated, model, inner, receipt = outer_prediction(records, fold)
        indexes = [lookup[r['id'], r['protocol']] for r in held]
        raw[indexes], nested[indexes] = p, calibrated
        receipts.append(receipt); outer_models.append(model); inner_models.append(inner)
    if not np.isfinite(raw).all() or not np.isfinite(nested).all():
        raise ValueError('Incomplete320-question metadata OOF grid')
    temperature, search = temperature_search(records, raw)
    model = train_model(records)
    if model.get('constant'):
        raise ValueError('Cannot deploy a binary adapter without both TRAIN classes')
    adapter = Adapter(model, temperature, {})
    adapter.receipt = {'schema': SCHEMA, 'learned_state_sha256': digest(adapter.state()),
        'training_questions': 320, 'training_records': 448, 'runtime_temperature': temperature,
        'registration_sha256': registration_sha256, 'root_tables_use_nested_oof': True,
        'neural_calls': 0, 'fit_weights': model['fit_weights'], 'old_questions': 0,
        'subject_target': weights.TARGET, 'question_normalization_unit': 'independent eligible question'}
    adapter.validate()
    receipt = {'schema': SCHEMA, 'records_sha256': fingerprint(records), 'features': features.FEATURES,
        'C': 1., 'seed': features.SEED, 'temperature_grid': features.TEMPERATURES,
        'weights': model['fit_weights'], 'outer': receipts, 'runtime_temperature_search': search,
        'raw_oof_sha256': fingerprint(raw.tolist()), 'nested_oof_sha256': fingerprint(nested.tolist()),
        'strict_nested_question_calibration': True, 'root_tables_use_final_temperature': False,
        'historical_byte_identity_claim': False, 'neural_calls': 0, 'cloud_calls': 0,
        'development_or_test_used': False}
    return {'adapter': adapter, 'raw': raw, 'nested': nested, 'outer_models': outer_models,
            'inner_models': inner_models, 'receipt': receipt}


def root_signals(records, probabilities, bundle):
    validate(records)
    if np.asarray(probabilities).shape != (len(records),) or not np.isfinite(probabilities).all():
        raise ValueError('Misaligned nested metadata probabilities')
    aligned, signals = [], {}
    for row, p in zip(records, probabilities):
        if row['protocol'] != 'root':
            continue
        qid = row['id']; judge = bundle['judges'][qid, '']; current = bundle['trees'][qid]['']
        if (row['current_sha256'], row['review_sha256']) != (fingerprint(current), fingerprint(judge)):
            raise ValueError('Nested root forecast lost its paid input alignment')
        signals[qid] = signal(judge, current, float(p))
        aligned.append({'id': qid, 'subject': row['subject'], 'nested_probability': float(p),
                        'signal': signals[qid], 'outer_fold': inherited.fold(qid),
                        'current_sha256': row['current_sha256'], 'review_sha256': row['review_sha256']})
    if set(signals) != {qid for qid, path in bundle['judges']} or len(signals) != 320:
        raise ValueError('Need exactly320 aligned root metadata observations')
    return signals, aligned
