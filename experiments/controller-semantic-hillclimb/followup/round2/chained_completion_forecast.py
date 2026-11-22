"""Conditional binary chain retaining a normalized sixteen-state joint belief."""
from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nested_dense_diagnostic as previous

SETTINGS = tuple({'C': value} for value in (.001, .01, .1))
PRIOR_WEIGHT = .05
BITS = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(float)


@dataclass
class Forecast:
    heads: list
    prior: np.ndarray

    @property
    def classes_(self): return np.arange(16)

    def predict_proba(self, x):
        x = np.asarray(x); result = np.ones((len(x), 16))
        for step, head in enumerate(self.heads):
            for prefix in range(2**step):
                states = np.arange(prefix*2**(4-step), (prefix+1)*2**(4-step))
                inputs = np.c_[x, np.tile(BITS[states[0], :step], (len(x), 1))]
                predicted = head.predict_proba(inputs)
                p = predicted[:, list(head.classes_).index(1)] if 1 in head.classes_ else np.zeros(len(x))
                result[:, states] *= np.where(BITS[states, step][None, :] == 1, p[:, None], 1-p[:, None])
        return (1-PRIOR_WEIGHT)*result + PRIOR_WEIGHT*self.prior[None, :]


def fit(x, y, settings):
    y = np.asarray(y)
    if len(x) != len(y) or not np.isin(y, np.arange(16)).all(): raise ValueError('Aligned joint states required')
    bits = BITS[y.astype(int)]; heads = []
    for step in range(4):
        features = np.c_[x, bits[:, :step]]; target = bits[:, step].astype(int)
        estimator = (LogisticRegression(C=settings['C'], max_iter=1000, solver='lbfgs')
                     if len(np.unique(target)) == 2 else DummyClassifier(strategy='prior'))
        heads.append(make_pipeline(StandardScaler(), estimator).fit(features, target))
    counts = np.bincount(y.astype(int), minlength=16)+1.
    return Forecast(heads, counts/counts.sum())


def select(data, family, eligible):
    x, y, valid = data['features'][family], data['labels'], data['valid']
    folds = np.array([previous.inner_fold(r['id']) for r in data['records']]); scores = []
    if set(folds[eligible]) != {0, 1, 2}: raise ValueError('All inner question folds required')
    for settings in SETTINGS:
        p = np.full((len(y), 16), np.nan)
        for fold in range(3):
            train, held = eligible & (folds != fold), eligible & (folds == fold)
            previous.group_receipt(data['records'], train, held)
            p[held] = previous.predict(fit(x[train], y[train], settings), x[held], valid[held])
        if not np.isfinite(p[eligible]).all() or np.isfinite(p[~eligible]).any(): raise ValueError('Selection scope changed')
        losses = previous.fixed.losses(y[eligible], p[eligible])
        scores.append({'settings': settings, 'joint_log_loss': float(losses['joint_log_loss'].mean()),
                       'mean_four_marginal_Brier': float(losses['mean_four_marginal_Brier'].mean())})
    winner = min(range(len(scores)), key=lambda i: (scores[i]['joint_log_loss'], i))
    return SETTINGS[winner], scores


def outer_prediction(data, family, fold):
    held = np.array([r['fold'] == fold for r in data['records']]); train = ~held
    receipt = previous.group_receipt(data['records'], train, held)
    settings, scores = select(data, family, train)
    model = fit(data['features'][family][train], data['labels'][train], settings)
    p = previous.predict(model, data['features'][family][held], data['valid'][held])
    return held, p, model, {'outer_fold': fold, **receipt, 'selected': settings, 'inner_search': scores,
                           'outer_held_labels_used_for_fit_or_selection': False}
