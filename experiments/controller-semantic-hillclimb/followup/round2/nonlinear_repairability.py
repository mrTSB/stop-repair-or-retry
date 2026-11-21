"""Small nonlinear paid-feature forecasts inside the existing eight-state POMDP.

Only the state forecaster and its matched emission observations change. Every
emission forecast excludes its question from both tree fitting and temperature
selection. The frozen old-only neural observations are independent of these
512 training questions; this module never runs a neural model.
"""

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import repairability_pomdp as core

SEED = 2026112307
TEMPERATURES = (.5, .75, 1., 1.5, 2., 3.)
LEAVES = (7, 15)


def identity(values):
    return hashlib.sha256(json.dumps(sorted(set(values))).encode()).hexdigest()


def folds_for(qids, salt=''):
    return np.asarray([int(hashlib.sha256((salt + str(q)).encode()).hexdigest()[:8], 16) % 3 for q in qids])


def temperature_scale(probabilities, temperature):
    value = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.)) / temperature
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def choose_temperature(probabilities, labels):
    losses = [(float(-np.log(temperature_scale(probabilities, t)[np.arange(len(labels)), labels]).mean()), t)
              for t in TEMPERATURES]
    loss, temperature = min(losses, key=lambda pair: (pair[0], abs(pair[1] - 1.), pair[1]))
    return temperature, [{'temperature': t, 'mean_log_loss': value} for value, t in losses]


@dataclass
class ConstantModel:
    label: int
    classes_: object = field(init=False)

    def __post_init__(self):
        self.classes_ = np.asarray([self.label])

    def predict_proba(self, x):
        return np.ones((len(x), 1))


def raw_fit(x, y, max_leaf_nodes):
    if max_leaf_nodes not in LEAVES:
        raise ValueError('Use the prespecified seven- or fifteen-leaf family')
    if len(np.unique(y)) == 1:
        return ConstantModel(int(y[0]))
    model = HistGradientBoostingClassifier(max_leaf_nodes=max_leaf_nodes, max_iter=100,
        learning_rate=.1, l2_regularization=5., min_samples_leaf=30, max_bins=255,
        early_stopping=False, random_state=SEED, categorical_features=None)
    with threadpool_limits(limits=1):
        return model.fit(x, y)


def aligned_probability(model, x):
    result = np.full((len(x), 8), 1e-12)
    with threadpool_limits(limits=1):
        result[:, model.classes_.astype(int)] += model.predict_proba(x)
    return result / result.sum(axis=1, keepdims=True)


@dataclass
class Forecaster:
    estimator: object
    temperature: float
    classes_: object = field(default_factory=lambda: np.arange(8))

    def predict_proba(self, x):
        return temperature_scale(aligned_probability(self.estimator, np.asarray(x)), self.temperature)


def crossfit(x, y, qids, max_leaf_nodes):
    x, y, qids = np.asarray(x), np.asarray(y, dtype=int), np.asarray(qids)
    outer = folds_for(qids)
    raw, calibrated, receipts = np.zeros((len(y), 8)), np.zeros((len(y), 8)), []
    for fold in (0, 1, 2):
        fit_mask, held = outer != fold, outer == fold
        fit_ids, held_ids = set(qids[fit_mask]), set(qids[held])
        if not fit_ids or not held_ids or fit_ids & held_ids:
            raise ValueError('Question-excluded forecasts need three nonempty disjoint outer folds')
        tx, ty, tq = x[fit_mask], y[fit_mask], qids[fit_mask]
        inner = folds_for(tq, 'temperature-inner-' + str(fold) + ':')
        predictions, inner_receipts = np.zeros((len(ty), 8)), []
        for part in (0, 1, 2):
            fitting, validating = inner != part, inner == part
            train_ids, calibration_ids = set(tq[fitting]), set(tq[validating])
            if not train_ids or not calibration_ids or train_ids & calibration_ids or (train_ids | calibration_ids) & held_ids:
                raise ValueError('Temperature fitting must exclude the outer held question')
            estimator = raw_fit(tx[fitting], ty[fitting], max_leaf_nodes)
            predictions[validating] = aligned_probability(estimator, tx[validating])
            inner_receipts.append({'fold': part, 'fit_ids_sha256': identity(train_ids),
                'calibration_ids_sha256': identity(calibration_ids), 'fit_questions': len(train_ids),
                'calibration_questions': len(calibration_ids), 'outer_held_overlap': 0})
        temperature, losses = choose_temperature(predictions, ty)
        estimator = raw_fit(tx, ty, max_leaf_nodes)
        raw[held] = aligned_probability(estimator, x[held])
        calibrated[held] = temperature_scale(raw[held], temperature)
        receipts.append({'fold': fold, 'fit_ids_sha256': identity(fit_ids), 'held_ids_sha256': identity(held_ids),
            'fit_questions': len(fit_ids), 'held_questions': len(held_ids), 'question_overlap': 0,
            'temperature': temperature, 'temperature_selection': losses, 'inner_folds': inner_receipts})
    temperature, losses = choose_temperature(raw, y)
    final = Forecaster(raw_fit(x, y, max_leaf_nodes), temperature)
    return final, calibrated, {'outer_folds': receipts, 'final_temperature': temperature,
        'final_temperature_selection': losses, 'final_temperature_fits_only_training_OOF_labels': True,
        'emission_temperature_excludes_own_question': True,
        'raw_oof_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
        'calibrated_oof_sha256': hashlib.sha256(calibrated.tobytes()).hexdigest()}


def training_table(rows, trees, neural_lookup):
    samples, features, states, qids = [], [], [], []
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Training questions must be unique')
    for row in rows:
        qid = row['id']
        for path in ('', 'repair', 'retry'):
            children = [(path + '/' if path else '') + a for a in ('repair', 'retry')]
            if any(p not in trees[qid] for p in (path, *children)):
                raise ValueError('Use every paired plain training response')
            state = core.state(core.correct(trees[qid][path], row['answer']),
                               core.correct(trees[qid][children[0]], row['answer']),
                               core.correct(trees[qid][children[1]], row['answer']))
            features.append(core.observables(row, path, trees[qid], neural_lookup[qid, path]))
            samples.append((row, path)); states.append(state); qids.append(qid)
    x, y = np.asarray(features, dtype=float), np.asarray(states, dtype=int)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite cached training features')
    return samples, x, y, qids


def fit(rows, trees, observer, neural_lookup, smoothing=8., max_leaf_nodes=7):
    # Keep the already-audited structural transition support and cost estimator.
    # Its temporary logistic forecaster/emissions are replaced immediately by
    # this fixed nonlinear family and question-excluded nonlinear emissions.
    with threadpool_limits(limits=1):
        model = core.fit(rows, trees, observer, neural_lookup, smoothing, regularization=1.)
    samples, x, y, qids = training_table(rows, trees, neural_lookup)
    model.forecaster, oof, calibration = crossfit(x, y, qids, max_leaf_nodes)
    support = np.ones((2, 8, core.N_OBS)); support[:, core.CURRENT == 1, -1] = 0.

    def emission_counts(subject=None):
        counts = .1 * support.copy()
        for i, (row, path) in enumerate(samples):
            if not path or (subject is not None and row['subject'] != subject):
                continue
            signal = core.observation(oof[i], bool(trees[row['id']][path]['answer']))
            counts[('repair', 'retry').index(path), y[i], signal] += 1
        return counts

    global_counts = emission_counts()
    pooled = global_counts / global_counts.sum(-1, keepdims=True)
    for subject, dynamics in model.dynamics.items():
        counts = emission_counts(subject) + smoothing * pooled
        dynamics.emission = counts / counts.sum(-1, keepdims=True)
    model.fit_receipt.pop('calibration_C', None)
    model.fit_receipt.update({'fit_module': 'nonlinear_repairability',
        'forecast': 'HistGradientBoostingClassifier; eight-state joint forecast',
        'forecaster_hyperparameters': {'max_leaf_nodes': max_leaf_nodes, 'max_iter': 100,
            'learning_rate': .1, 'l2_regularization': 5., 'min_samples_leaf': 30,
            'max_bins': 255, 'early_stopping': False, 'random_state': SEED, 'categorical_features': None},
        'temperature_grid': list(TEMPERATURES), 'calibration': calibration,
        'neural_observer': 'Frozen old-only checkpoint; no new512 training questions in neural fitting',
        'forecaster_and_temperature_exclude_own_question_in_emission_fit': True,
        'structural_transitions_and_costs_unchanged': True,
        'feature_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
        'training_question_ids_sha256': identity(qids)})
    return model
