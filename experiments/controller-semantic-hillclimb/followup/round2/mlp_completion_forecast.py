"""Small, regularized nonlinear head over already paid frozen observations."""
from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SETTINGS={'hidden_layer_sizes':(64,),'activation':'tanh','alpha':10.,'learning_rate_init':.001,
    'max_iter':300,'early_stopping':False,'n_iter_no_change':25,'tol':1e-4,'random_state':2381124}
PRIOR_WEIGHT=.05


@dataclass
class Forecast:
    estimator:object
    prior:np.ndarray
    @property
    def classes_(self):return np.arange(16)
    def predict_proba(self,x):
        p=np.tile(PRIOR_WEIGHT*self.prior,(len(x),1))
        p[:,self.estimator.classes_.astype(int)]+=(1-PRIOR_WEIGHT)*self.estimator.predict_proba(x)
        return p


def fit(x,y,unused_forest_settings=None):
    y=np.asarray(y)
    if len(x)!=len(y) or not np.isin(y,np.arange(16)).all():raise ValueError('Aligned joint-state labels required')
    if len(np.unique(y))==1:estimator=DummyClassifier(strategy='prior').fit(x,y)
    else:estimator=make_pipeline(StandardScaler(),MLPClassifier(batch_size=min(128,len(y)),**SETTINGS)).fit(x,y)
    prior=np.bincount(y,minlength=16)+1.
    return Forecast(estimator,prior/prior.sum())
