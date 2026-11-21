"""Three binary forecasts reconstructed into a correlated eight-state belief.

Iterative proportional fitting (IPF) adjusts a training-only joint prior to the
three predicted marginals. Univariate multiplicative updates preserve the
prior's interaction odds ratios; this is not an independent Bernoulli product.
Question-grouped cross-fitting supplies every fitted emission observation.
"""

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import repairability_pomdp as core

BITS = np.asarray([[(s >> 2) & 1, (s >> 1) & 1, s & 1] for s in range(8)])
SUBJECTS = ('math', 'chemistry', 'physics')


def reconstruct(prior, marginals, tolerance=1e-9, maximum_iterations=2000):
    prior = np.asarray(prior, dtype=float)
    if prior.shape != (8,) or np.any(prior <= 0.) or not np.isfinite(prior).all():
        raise ValueError('IPF requires a positive, finite training prior on all eight states')
    target = np.asarray(marginals, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all() or np.any((target < 0.) | (target > 1.)):
        raise ValueError('IPF requires three probability marginals')
    # Smooth boundary predictions to retain the positive fitted joint support.
    target = np.clip(target, 1e-6, 1. - 1e-6)
    p = prior / prior.sum()
    for _ in range(maximum_iterations):
        for bit in range(3):
            mask = BITS[:, bit].astype(bool)
            mass = p[mask].sum()
            p[mask] *= target[bit] / mass
            p[~mask] *= (1. - target[bit]) / (1. - mass)
            p /= p.sum()
        if np.max(np.abs(p @ BITS - target)) <= tolerance:
            return p
    raise ValueError('Joint IPF reconstruction did not converge')


@dataclass
class ConstantBinary:
    probability: float
    classes_: object = field(default_factory=lambda: np.array([0, 1]))

    def predict_proba(self, x):
        return np.tile([1. - self.probability, self.probability], (len(x), 1))


@dataclass
class Forecaster:
    binary: list
    priors: dict
    classes_: object = field(default_factory=lambda: np.arange(8))
    _cache: dict = field(default_factory=dict, repr=False)

    def __getstate__(self):
        return {**vars(self), '_cache': {}}

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        keys = [r.tobytes() for r in x]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            features = x[missing]
            marginals = np.stack([m.predict_proba(features)[:, 1] for m in self.binary], axis=1)
            for i, probs in zip(missing, marginals):
                # These are the documented final four paid observables from
                # core.observables: continuation indicator + subject one-hot.
                indicators = x[i, -3:]
                if not np.isclose(indicators.sum(), 1.) or not np.all(np.isin(indicators, (0., 1.))):
                    raise ValueError('Unexpected subject feature encoding')
                subject = SUBJECTS[int(indicators.argmax())]
                stage = int(bool(x[i, -4]))
                self._cache[keys[i]] = reconstruct(self.priors[subject, stage], probs)
        return np.asarray([self._cache[k] for k in keys])


def fit_forecaster(x, y, regularization=1., prior_smoothing=16.):
    x, y = np.asarray(x), np.asarray(y)
    labels = BITS[y]
    binary = []
    for bit in range(3):
        values = labels[:, bit]
        if len(np.unique(values)) == 1:
            model = ConstantBinary(float(values[0]))
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=1000)).fit(x, values)
        binary.append(model)
    pooled = np.bincount(y, minlength=8).astype(float) + .5
    pooled /= pooled.sum()
    priors = {}
    for j, subject in enumerate(SUBJECTS):
        for stage in (0, 1):
            mask = (x[:, -3 + j] == 1.) & (x[:, -4] == stage)
            counts = np.bincount(y[mask], minlength=8).astype(float) + .5 + prior_smoothing * pooled
            priors[subject, stage] = counts / counts.sum()
    return Forecaster(binary, priors)


def crossfit(x, y, qids, regularization=1., prior_smoothing=16.):
    qids = np.asarray(qids)
    folds = np.asarray([int(hashlib.sha256(q.encode()).hexdigest()[:8], 16) % 3 for q in qids])
    out = np.zeros((len(y), 8)); receipts = []
    for fold in (0, 1, 2):
        fit_mask, held = folds != fold, folds == fold
        fit_ids, held_ids = set(qids[fit_mask]), set(qids[held])
        if fit_ids & held_ids or not fit_ids or not held_ids:
            raise ValueError('Question-grouped forecasts need three nonempty disjoint folds')
        model = fit_forecaster(x[fit_mask], y[fit_mask], regularization, prior_smoothing)
        out[held] = model.predict_proba(x[held])
        receipts.append({'fold': fold, 'fit_questions': len(fit_ids), 'held_questions': len(held_ids),
                         'fit_ids_sha256': hashlib.sha256(json.dumps(sorted(fit_ids)).encode()).hexdigest(),
                         'held_ids_sha256': hashlib.sha256(json.dumps(sorted(held_ids)).encode()).hexdigest(),
                         'question_overlap': 0})
    return out, receipts


def fit(rows, trees, observer, neural_lookup, smoothing=16., regularization=1., prior_smoothing=16.):
    # Preserve the original transitions, support restrictions, costs and exact
    # Bellman backup. Only the forecaster and its matched OOF emissions change.
    model = core.fit(rows, trees, observer, neural_lookup, smoothing, regularization)
    samples, features, states, qids = [], [], [], []
    for row in rows:
        qid = row['id']
        for path in ('', 'repair', 'retry'):
            children = [(path + '/' if path else '') + a for a in ('repair', 'retry')]
            s = core.state(core.correct(trees[qid][path], row['answer']),
                           core.correct(trees[qid][children[0]], row['answer']),
                           core.correct(trees[qid][children[1]], row['answer']))
            features.append(core.observables(row, path, trees[qid], neural_lookup[qid, path]))
            samples.append((row, path)); states.append(s); qids.append(qid)
    x, y = np.asarray(features), np.asarray(states)
    oof, folds = crossfit(x, y, qids, regularization, prior_smoothing)
    model.forecaster = fit_forecaster(x, y, regularization, prior_smoothing)
    support = np.ones((2, 8, core.N_OBS)); support[:, core.CURRENT == 1, -1] = 0.

    def emission_counts(subject=None):
        counts = .1 * support.copy()
        for i, (row, path) in enumerate(samples):
            if not path or (subject is not None and row['subject'] != subject):
                continue
            action = ('repair', 'retry').index(path)
            obs = core.observation(oof[i], bool(trees[row['id']][path]['answer']))
            counts[action, y[i], obs] += 1
        return counts

    global_counts = emission_counts()
    pooled = global_counts / global_counts.sum(-1, keepdims=True)
    for subject, d in model.dynamics.items():
        counts = emission_counts(subject) + smoothing * pooled
        d.emission = counts / counts.sum(-1, keepdims=True)
    model.fit_receipt.update({'forecast': 'Three binary logistic forecasts; IPF from training subject/stage joint prior',
                              'prior_smoothing': prior_smoothing, 'ipf_marginal_clip': [1e-6, 1. - 1e-6],
                              'interaction_odds_ratios_preserved': True, 'emission_folds': folds,
                              'forecaster_and_prior_exclude_own_question_in_emission_fit': True})
    return model
