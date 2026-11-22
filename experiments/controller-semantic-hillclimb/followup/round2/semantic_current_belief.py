"""Correct the current-answer marginal while preserving conditional futures."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CURRENT = (np.arange(16) >> 3).astype(float)


def features(records, beliefs, semantic_logits):
    p = np.clip(np.asarray(beliefs) @ CURRENT, 1e-6, 1-1e-6)
    z = np.clip(np.asarray(semantic_logits), -20., 20.)
    return np.stack([np.log(p/(1-p)), z, np.log(p/(1-p))*z,
        [float(r['subject'] == 'chemistry') for r in records], [float(bool(r['path'])) for r in records]], axis=1)


def fit(x, y):
    if len(np.unique(y)) != 2: raise ValueError('Both correctness classes required for fixed calibration')
    return make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=2381125)).fit(x, y)


def update(beliefs, current, valid):
    p = np.asarray(beliefs).copy(); current = np.clip(np.asarray(current), 1e-7, 1-1e-7)
    for i in range(len(p)):
        if not valid[i]: continue
        prior = np.clip(p[i] @ CURRENT, 1e-12, 1-1e-12)
        p[i] *= np.where(CURRENT, current[i]/prior, (1-current[i])/(1-prior))
        p[i] /= p[i].sum()
    return p
