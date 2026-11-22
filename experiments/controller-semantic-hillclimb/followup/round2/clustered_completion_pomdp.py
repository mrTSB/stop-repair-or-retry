"""Finite POMDP observations clustered from the full joint TRAIN forecast.

Keep exactly24 valid and6 invalid observation symbols. Centers use only the
square-root probabilities of excluded-question child forecasts, never labels.
"""
import copy
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

import quantile_completion_pomdp as previous

core = previous.core
Planner = previous.Planner
SubjectPlanner = previous.SubjectPlanner
evaluate_science = previous.evaluate_science
evaluate = previous.evaluate


@dataclass
class Kernel:
    valid_centers: np.ndarray
    invalid_centers: np.ndarray

    def signal(self, p, valid):
        point = np.sqrt(core.mask_invalid(p, valid))
        centers = self.valid_centers if valid else self.invalid_centers
        return int(((centers-point)**2).sum(1).argmin()) + (0 if valid else 24)


def fit_kernel(records, probability, valid):
    p = np.asarray(probability); valid = np.asarray(valid, bool)
    if p.shape != (len(records), 16) or valid.shape != (len(records),): raise ValueError('Aligned full joint forecasts required')
    child = np.asarray([r['path'] in ('repair', 'retry') for r in records])
    normalized = np.asarray([core.mask_invalid(v, flag) for v, flag in zip(p, valid)])
    centers = []
    for flag, count in ((True, 24), (False, 6)):
        chosen = np.sqrt(normalized[child & (valid == flag)])
        if len(chosen) < count: raise ValueError('Insufficient TRAIN child observations')
        model = KMeans(n_clusters=count, random_state=2381124, n_init=10, max_iter=100).fit(chosen)
        centers.append(model.cluster_centers_)
    return Kernel(*centers)


def rebin(base, rows, trees, records, probability):
    expected = [(r['id'], p) for r in rows for p in ('', 'repair', 'retry')]
    if [(r['id'], r['path']) for r in records] != expected: raise ValueError('Forecast ordering changed')
    valid = np.asarray([bool(trees[qid][p]['answer']) for qid, p in expected])
    kernel = fit_kernel(records, probability, valid); forecasts = dict(zip(expected, probability))
    support = np.ones((2, 16, core.N_OBS)); support[:, core.CURRENT == 1, 24:] = 0.
    def counts(group):
        emission = .1*support.copy()
        for row in group:
            tree = trees[row['id']]
            for action, path in enumerate(('repair', 'retry')):
                paths = [path, *[path+'/'+a for a in ('repair', 'retry', 'complete')]]
                state = core.state(*[core.plain.correct(tree[p], row['answer']) for p in paths])
                signal = kernel.signal(forecasts[row['id'], path], bool(tree[path]['answer']))
                emission[action, state, signal] += 1
        return emission
    normalize = lambda p: p/p.sum(-1, keepdims=True)
    pooled = normalize(counts(rows)); dynamics = copy.deepcopy(base.dynamics)
    for subject, d in dynamics.items():
        d.emission = normalize(counts([r for r in rows if r['subject'] == subject])+base.fit_receipt['smoothing']*pooled)
    receipt = {**base.fit_receipt, 'observation_partition': 'KMeans on sqrt joint TRAIN child forecasts:24 valid,6 invalid',
        'kernel_uses_reference_labels': False, 'kernel_seed': 2381124, 'kernel_n_init': 10,
        'emissions_refit_to_new_partition': True,
        'unchanged': ['forecaster', 'prior', 'transition', 'cost', '16 states', '30 observations', 'exact finite Bellman backup']}
    return Planner(copy.deepcopy(base.classifier), copy.deepcopy(base.forecaster), dynamics, receipt, base.feature_family, kernel)
