"""Cross-fitted review reliability as a learned finite POMDP observation channel."""

from dataclasses import dataclass,field
import hashlib

import numpy as np

import difficulty_pomdp as base
from context_pomdp import equal,REVIEW_WIDTH
from scoring import correct


def features(row,path,tree,judge):
    # Only the paid review, current response and ancestors enter this channel.
    verdict=judge.get('verdict',{})
    proof={**judge,'text':str(verdict.get('solution',''))+'\nFinal: '+judge['answer'],
           'confidence':None}
    return (base.visible(row,path,tree)+base.visible(row,path,{**tree,path:proof})+
            [float(judge['p_correct']),int(judge['parse_valid']),
             int(equal(judge['answer'],tree[path]['answer'])),
             min(len(str(verdict.get('solution','')))/2000,2),
             *[int(judge['action']==a) for a in ('stop','repair','retry')]])


def encode(probability,judge,current):
    if not judge['parse_valid']:return REVIEW_WIDTH-1
    pc=probability[1]+probability[3]
    bucket=int(np.searchsorted([.2,.5,.8,.97],pc))
    return (('stop','repair','retry').index(judge['action'])*5+bucket)*2+int(equal(judge['answer'],current['answer']))


@dataclass
class Observer:
    classifier: object
    _cache: dict=field(default_factory=dict,repr=False)

    def __getstate__(self):return {**vars(self),'_cache':{}}

    def signal(self,row,path,tree,judge):
        key=row['id'],path,tree[path]['text'],judge['prompt_sha256'],judge['text']
        if key not in self._cache:
            p=base.probabilities(self.classifier,np.asarray([features(row,path,tree,judge)]))[0]
            self._cache[key]=encode(p,judge,tree[path])
        return self._cache[key]


def fit(rows,trees,judges,leaf=12,smoothing=32.,late_feedback=False,stage_review=True,
        review_leaf=24,contextual=True,transition_leaf=24,context_weight=.7):
    hard={r['id']:int(np.mean([correct(trees[r['id']][p],r['answer']) for p in base.FIXED])<.6) for r in rows}
    entries=[(r,p) for r in rows for p in ('','repair','retry','guided')]
    x=np.asarray([features(r,p,trees[r['id']],judges[r['id'],p]) for r,p in entries])
    y=np.asarray([2*hard[r['id']]+int(correct(trees[r['id']][p],r['answer'])) for r,p in entries])
    folds=np.asarray([int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16)%5 for r,p in entries])
    probabilities=np.zeros((len(entries),4))
    for fold in range(5):
        classifier=base.forest(review_leaf,base.SEED+70+fold).fit(x[folds!=fold],y[folds!=fold])
        probabilities[folds==fold]=base.probabilities(classifier,x[folds==fold])
    classifier=base.forest(review_leaf,base.SEED+80).fit(x,y).set_params(n_jobs=1)
    observer=Observer(classifier)
    signals={(r['id'],p):encode(v,judges[r['id'],p],trees[r['id']][p]) for (r,p),v in zip(entries,probabilities)}
    model=base.fit(rows,trees,judges,leaf,smoothing,late_feedback,stage_review,
                   review_observer=observer,review_signals=signals)
    if contextual:
        from contextual_pomdp import fit as context_fit
        model=context_fit(rows,trees,judges,leaf,smoothing,late_feedback,stage_review,
                          transition_leaf,context_weight,initial_model=model,review_signals=signals)
    model.fit_receipt.update({'review_calibration':True,'review_leaf':review_leaf,
                              'review_grouped_folds':5,'review_observations':len(entries)})
    return model
