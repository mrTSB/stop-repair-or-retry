"""Sparse paid-text plus existing numeric outcome forecasts; no answer generation."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SETTINGS = (.1, 1., 10.)
PRIOR_WEIGHT = .05


def public_text(row, path, paid):
    if row.get('answer') != '' or path not in ('', 'repair', 'retry') or set(paid) != {'', path}:
        raise ValueError('Only blank-reference paid prefixes may become lexical observations')
    return ('QUESTION\n'+row['question']+'\nPAID ROOT\n'+paid['']['text']
        +('\nPAID CURRENT '+path+'\n'+paid[path]['text'] if path else ''))


@dataclass
class Forecaster:
    vectorizer: object
    scaler: object
    estimator: object
    prior: np.ndarray
    constant: object = None

    def matrix(self, texts, numeric):
        numbers = np.clip(self.scaler.transform(numeric), -6, 6)/np.sqrt(numeric.shape[1])
        return sparse.hstack((self.vectorizer.transform(texts), sparse.csr_matrix(numbers)), format='csr')

    def predict(self, texts, numeric, valid):
        p = np.zeros((len(texts), 16))
        if self.constant is not None: p[:, self.constant] = 1.
        else:
            p[:, self.estimator.classes_.astype(int)] = self.estimator.predict_proba(self.matrix(texts, numeric))
        p = (1-PRIOR_WEIGHT)*p+PRIOR_WEIGHT*self.prior
        p[~np.asarray(valid, bool), 8:] = 0.
        return p/p.sum(1, keepdims=True)


def fit(texts, numeric, labels, regularization):
    if regularization not in SETTINGS or len(texts) != len(numeric) or len(labels) != len(texts):
        raise ValueError('Use aligned observations and registered regularization')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    vectorizer.fit(texts); scaler = StandardScaler().fit(numeric)
    counts = np.bincount(labels, minlength=16)+1.; prior = counts/counts.sum()
    model = Forecaster(vectorizer, scaler, None, prior)
    classes = np.unique(labels)
    if len(classes) == 1: model.constant = int(classes[0])
    else:
        model.estimator = LogisticRegression(C=regularization, max_iter=500, random_state=2381130)
        model.estimator.fit(model.matrix(texts, numeric), labels)
    return model
