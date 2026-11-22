"""Four binary correctness forecasts mapped to an explicit sixteen-state belief.

The factorization is a conditional-independence approximation in the learned
initial observation model. The downstream POMDP still uses empirical joint
sixteen-state transitions and its exact finite-observation Bellman backup.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import nested_dense_diagnostic as previous

BITS=np.stack([(np.arange(16)>>shift)&1 for shift in (3,2,1,0)],axis=1)
CS=(.001,.01,.1,1.)


@dataclass
class Forecast:
    scaler:object
    heads:list

    @property
    def classes_(self):return np.arange(16)

    def predict_proba(self,x):
        z=self.scaler.transform(x);marginals=[]
        for head in self.heads:
            p=head.predict_proba(z)
            marginals.append(p[:,list(head.classes_).index(1)] if 1 in head.classes_ else np.zeros(len(z)))
        q=np.clip(np.stack(marginals,axis=1),1e-9,1-1e-9)
        joint=np.prod(np.where(BITS[None,:,:],q[:,None,:],1-q[:,None,:]),axis=2)
        return joint/joint.sum(axis=1,keepdims=True)


def fit(x,y,c):
    x=np.asarray(x);y=np.asarray(y)
    if y.ndim!=1 or len(x)!=len(y) or not np.isin(y,np.arange(16)).all():raise ValueError('Require aligned sixteen-state labels')
    scaler=StandardScaler();z=scaler.fit_transform(x);heads=[]
    for bit in range(4):
        label=BITS[y,bit]
        head=DummyClassifier(strategy='prior') if len(np.unique(label))==1 else LogisticRegression(C=c,max_iter=2000)
        heads.append(head.fit(z,label))
    return Forecast(scaler,heads)


def select(data,family,eligible):
    x,y,valid=data['features'][family],data['labels'],data['valid']
    folds=np.asarray([previous.inner_fold(r['id']) for r in data['records']]);scores=[]
    if set(folds[eligible])!={0,1,2}:raise ValueError('All inner question folds required')
    for c in CS:
        oof=np.full((len(y),16),np.nan)
        for fold in range(3):
            train=eligible&(folds!=fold);held=eligible&(folds==fold)
            previous.group_receipt(data['records'],train,held)
            model=fit(x[train],y[train],c);oof[held]=previous.predict(model,x[held],valid[held])
        if not np.isfinite(oof[eligible]).all() or np.isfinite(oof[~eligible]).any():raise ValueError('Inner selection scope changed')
        losses=previous.fixed.losses(y[eligible],oof[eligible])
        scores.append({'C':c,'mean_four_marginal_Brier':float(losses['mean_four_marginal_Brier'].mean()),
            'joint_log_loss':float(losses['joint_log_loss'].mean())})
    selected=min(range(len(scores)),key=lambda i:(scores[i]['mean_four_marginal_Brier'],i))
    return CS[selected],scores


def outer_prediction(data,family,fold):
    held=np.asarray([r['fold']==fold for r in data['records']]);train=~held
    receipt=previous.group_receipt(data['records'],train,held)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');c,scores=select(data,family,train)
        model=fit(data['features'][family][train],data['labels'][train],c)
        p=previous.predict(model,data['features'][family][held],data['valid'][held])
    return held,p,model,{'outer_fold':fold,**receipt,'selected_C':c,'inner_search':scores,
        'outer_held_labels_used_for_fit_or_selection':False,'warnings':[str(v.message) for v in captured]}
