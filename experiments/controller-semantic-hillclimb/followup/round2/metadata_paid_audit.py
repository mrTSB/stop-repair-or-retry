"""Independent paid-metadata audit for root-review hybrids; no dataset access.

The caller's existing policy lock must cover replay: the adapter interception
is scoped, as is the original neural ledger. Existing physical cost checks
remain in force. Stored CPU timing is checked, never replaced by a new timing.
"""
from contextlib import contextmanager
import math
from unittest.mock import patch

import numpy as np

import paid_review_metadata as paid
import review_metadata_diagnostic as diagnostic
import review_metadata_spec as specification
import prospective_audit as original

digest = paid.digest


def validate_adapter_state(state, receipt):
    n = len(specification.FEATURES)
    if (list(state['features']) != list(specification.FEATURES) or state['indexes'] != list(range(n))
            or state['classes'] != [0, 1] or digest(state) != receipt['learned_state_sha256']):
        raise ValueError('Frozen metadata feature/model identity changed')
    for name, shape in (('mean', (n,)), ('variance', (n,)), ('scale', (n,)),
                        ('coefficients', (1, n)), ('intercepts', (1,))):
        values = np.asarray(state[name], dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('Invalid frozen metadata array: ' + name)
    if (np.min(state['scale']) <= 0 or np.min(state['variance']) < 0
            or not math.isfinite(state['temperature']) or state['temperature'] <= 0):
        raise ValueError('Invalid frozen metadata scaling or temperature')


def expected_observation(state, receipt, subject, current, judge):
    """Evaluate only frozen scalar/linear arithmetic on public paid metadata."""
    validate_adapter_state(state, receipt)
    entry = paid.public_entry({'subject': subject, 'answer': ''}, current, judge)
    values = diagnostic.features(entry['row'], entry['current'], entry['judge'], 'root')
    standardized = (values - np.asarray(state['mean'])) / np.asarray(state['scale'])
    logit = float(np.asarray(state['coefficients'])[0] @ standardized + state['intercepts'][0])
    probability = float(np.clip(1. / (1. + np.exp(-np.clip(logit, -700., 700.))), 1e-6, 1.-1e-6))
    calibrated_logit = (math.log(probability) - math.log1p(-probability)) / state['temperature']
    probability = float(1. / (1. + math.exp(-float(np.clip(calibrated_logit, -40., 40.)))))
    if not judge['answer']:
        probability = 1e-6
    probability = float(np.clip(probability, 1e-6, 1.-1e-6))
    return {'kind': 'paid-review-metadata', 'path': '', 'input_sha256': digest(entry),
        'feature_sha256': digest(values.tolist()), 'model_sha256': receipt['learned_state_sha256'],
        'probability': probability, 'neural_calls': 0, 'encoder_token_slots': 0, 'include_reviews': False}


def verify_observation(observed, expected):
    if set(observed) != set(expected) | {'cpu_seconds'}:
        raise ValueError('Unexpected or missing metadata receipt fields')
    for key, value in expected.items():
        if key == 'probability':
            if not math.isfinite(observed[key]) or not math.isclose(observed[key], value, rel_tol=0., abs_tol=1e-12):
                raise ValueError('Metadata probability differs from frozen linear arithmetic')
        elif observed[key] != value:
            raise ValueError('Paid metadata receipt mismatch: ' + key)
    if type(observed['cpu_seconds']) not in (int, float) or not math.isfinite(observed['cpu_seconds']) or observed['cpu_seconds'] < 0:
        raise ValueError('Invalid measured metadata CPU timing')


def audit_metadata(item, paid_tree, paid_reviews, encoders, state, receipt, subject):
    validate_adapter_state(state, receipt)
    acquired = [d['path'] for d in item.get('decisions', []) if d['queried']]
    if acquired not in ([], ['']) or set(acquired) != {path for _, path in paid_reviews}:
        raise ValueError('Metadata C permits one optional root review')
    observations = item.get('metadata_observations')
    if (not isinstance(observations, list) or len(observations) != len(acquired)
            or item.get('metadata_calls') != len(observations) or item.get('review_neural_calls') != 0
            or item.get('include_reviews') is not False):
        raise ValueError('Metadata calls disagree with the paid root-review decision')
    if any(entry['kind'] != 'solver' for entry in encoders):
        raise ValueError('Metadata C must not charge or invoke a review neural pass')
    observed_paths = {entry['path'] for entry in encoders}
    expected_paths = {d['path'] for d in item['decisions']}
    if observed_paths != expected_paths or '' not in observed_paths:
        raise ValueError('Regular root/child neural observations must remain charged')
    if observations:
        if len(paid_reviews) != 1:
            raise ValueError('Missing paid root review')
        judge = next(iter(paid_reviews.values()))
        verify_observation(observations[0], expected_observation(state, receipt, subject, paid_tree[''], judge))
    elapsed = sum(record['cpu_seconds'] for record in observations)
    total = item.get('metadata_cpu_seconds')
    if type(total) not in (int, float) or not math.isfinite(total) or total < 0 or not math.isclose(total, elapsed, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Metadata CPU timing total differs from its paid receipts')
    ledger_sha = digest(observations)
    if item.get('metadata_ledger_sha256') != ledger_sha:
        raise ValueError('Paid metadata ledger changed after the decision')
    return {'passed': True, 'metadata_calls': len(observations), 'metadata_cpu_seconds': elapsed,
            'metadata_ledger_sha256': ledger_sha, 'review_neural_calls': 0}


@contextmanager
def capture(adapter, row, tree, reviews):
    state, receipt = adapter.state(), adapter.receipt
    validate_adapter_state(state, receipt)
    observations = []
    method = type(adapter).predict_paid
    def wrapped(self, safe, current, judge):
        if (self is not adapter or safe.get('answer') != '' or safe['id'] != row['id']
                or safe['subject'] != row['subject'] or digest(current) != digest(tree[''])
                or (row['id'], '') not in reviews or digest(judge) != digest(reviews[row['id'], ''])):
            raise ValueError('Metadata adapter saw a nonpaid prefix or reference')
        probability, observation = method(self, safe, current, judge)
        expected = expected_observation(state, receipt, row['subject'], current, judge)
        verify_observation(observation, expected)
        if not math.isclose(probability, observation['probability'], rel_tol=0., abs_tol=1e-12):
            raise ValueError('Runtime metadata probability differs from its receipt')
        observations.append(dict(observation))
        return probability, observation
    with patch.object(type(adapter), 'predict_paid', wrapped):
        yield observations


def replay_paid_inputs(function, row, available_tree, available_reviews, family, model=None):
    if family != 'C' or model is None or not hasattr(model, 'metadata_adapter'):
        return original.replay_paid_inputs(function, row, available_tree, available_reviews, family, model)
    if model.review_observer is not None or model.mentor.review_observer is not None:
        raise ValueError('Metadata replacement retained a neural review observer')
    adapter = model.metadata_adapter
    def checked(safe, tree, reviews):
        with capture(adapter, safe, tree, reviews) as captured:
            item = function(safe, tree, reviews)
        if item.get('metadata_observations') != captured:
            raise ValueError('Runtime metadata receipts differ from actual same-attempt calls')
        value = digest(captured)
        if 'metadata_ledger_sha256' in item and item['metadata_ledger_sha256'] != value:
            raise ValueError('Runtime metadata ledger hash is inconsistent')
        item['metadata_ledger_sha256'] = value
        return item
    item, encoders = original.replay_paid_inputs(checked, row, available_tree, available_reviews, family, model)
    paid_tree = original.prefix(available_tree, item['path'])
    paid_reviews = {(row['id'], d['path']): available_reviews[row['id'], d['path']]
                    for d in item['decisions'] if d['queried']}
    audit_metadata(item, paid_tree, paid_reviews, encoders, adapter.state(), adapter.receipt, row['subject'])
    return item, encoders
