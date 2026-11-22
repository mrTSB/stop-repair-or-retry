"""Learn a context-dependent mentor prior while preserving A's current truth mass."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import completion_mentor_transfer as transfer
import nonlinear_completion_forecast as forest

CS = (.001, .01, .1)
CURRENT = transfer.mentor.REWARD.astype(int)
MENTOR = np.arange(8) % 2


def condition(probability, current_mass):
    p = np.asarray(probability, dtype=float); mass = np.asarray(current_mass, dtype=float)
    if (p.shape != (len(mass), 8) or not np.isfinite(p).all() or not np.isfinite(mass).all()
            or np.any(p <= 0) or np.any((mass < 0) | (mass > 1))):
        raise ValueError('Require strictly positive eight-state probabilities and valid current mass')
    result = np.zeros_like(p)
    for value in (0, 1):
        mask = CURRENT == value
        result[:, mask] = p[:, mask]/p[:, mask].sum(1, keepdims=True)*(mass if value else 1-mass)[:, None]
    return result


def kernels(records, smoothing=32.):
    support = (transfer.plain.CURRENT[:, None] == transfer.mentor.REWARD[None, :]).astype(float)
    if len({r['id'] for r in records}) != len(records) or not records: raise ValueError('Require unique training questions')
    counts = .1*support.copy()
    for r in records:
        if not support[r['source'], r['target']]: raise ValueError('Changed current truth across modes')
        counts[r['source'], r['target']] += 1
    pooled = counts/counts.sum(1, keepdims=True); result = {}
    for subject in transfer.runtime.SCIENCE:
        local = .1*support.copy()
        for r in records:
            if r['subject'] == subject: local[r['source'], r['target']] += 1
        local += smoothing*pooled; result[subject] = local/local.sum(1, keepdims=True)
    transfer.runtime.validate_transfer(result)
    return result


@dataclass
class Forecast:
    classifier: object
    prior: np.ndarray

    def predict_proba(self, x):
        p = np.tile(.05*self.prior, (len(x), 1))
        p[:, self.classifier.classes_.astype(int)] += .95*self.classifier.predict_proba(x)
        return p


def fit(x, labels, strength):
    if not np.isin(labels, np.arange(8)).all() or len(np.unique(labels)) < 2:
        raise ValueError('Require supported multistate training labels')
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=strength, max_iter=1000, solver='lbfgs')).fit(x, labels)
    counts = np.bincount(labels, minlength=8)+1.
    return Forecast(classifier, counts/counts.sum())


def loss(labels, p):
    return -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.))


def select(data, family, eligible):
    x, y = data['features'][family], data['labels']
    folds = np.asarray([forest.previous.inner_fold(r['id']) for r in data['records']]); scores = []
    if set(folds[eligible]) != {0, 1, 2}: raise ValueError('All inner question folds required')
    for strength in CS:
        p = np.full((len(y), 8), np.nan)
        for fold in range(3):
            train, held = eligible & (folds != fold), eligible & (folds == fold)
            forest.previous.group_receipt(data['records'], train, held)
            p[held] = fit(x[train], y[train], strength).predict_proba(x[held])
        if not np.isfinite(p[eligible]).all() or np.isfinite(p[~eligible]).any(): raise ValueError('Changed inner prediction scope')
        scores.append({'C': strength, 'joint_log_loss': float(loss(y[eligible], p[eligible]).mean())})
    selected = min(range(len(scores)), key=lambda i: (scores[i]['joint_log_loss'], i))
    return CS[selected], scores


def outer_prediction(data, family, fold):
    held = np.asarray([r['fold'] == fold for r in data['records']]); train = ~held
    receipt = forest.previous.group_receipt(data['records'], train, held)
    strength, scores = select(data, family, train)
    model = fit(data['features'][family][train], data['labels'][train], strength)
    p = condition(model.predict_proba(data['features'][family][held]), data['current_mass'][held])
    table = kernels([r for r, keep in zip(data['transfer_records'], train) if keep])
    baseline = np.asarray([a @ table[r['subject']] for r, a, keep in zip(data['records'], data['A_oof'], held) if keep])
    return held, p, baseline, model, {'fold': fold, **receipt, 'C': strength, 'inner_search': scores,
        'outer_held_labels_used_for_fit_or_selection': False, 'old_transfer_also_excludes_outer_held': True}
