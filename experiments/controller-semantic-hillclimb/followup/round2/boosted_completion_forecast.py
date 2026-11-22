"""Smoothed boosted categorical observations for the unchanged sixteen-state POMDP.

Every tree and the smoothing prior see training questions only. This changes
the observation likelihood forecast, not the action horizon or solver.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import nested_dense_diagnostic as previous

SETTINGS = tuple({'l2_regularization': strength} for strength in (10., 100.))
ITERATIONS = 80
PRIOR_WEIGHT = .05


@dataclass
class Forecast:
    forest: object
    prior: np.ndarray

    @property
    def classes_(self): return np.arange(16)

    def predict_proba(self, x):
        values = np.tile(PRIOR_WEIGHT*self.prior, (len(x), 1))
        values[:, self.forest.classes_.astype(int)] += (1-PRIOR_WEIGHT)*self.forest.predict_proba(x)
        return values


def fit(x, y, settings):
    y = np.asarray(y)
    if len(x) != len(y) or not np.isin(y, np.arange(16)).all(): raise ValueError('Aligned joint states required')
    counts = np.bincount(y, minlength=16)+1.
    model = HistGradientBoostingClassifier(max_iter=ITERATIONS, max_leaf_nodes=7,
        learning_rate=.08, min_samples_leaf=24, max_bins=32, early_stopping=False,
        random_state=2381124, **settings).fit(x, y)
    return Forecast(model, counts/counts.sum())


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
