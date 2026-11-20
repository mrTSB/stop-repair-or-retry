"""POMDP transitions conditioned on the paid, currently observable response features."""

from dataclasses import dataclass,field

import numpy as np

from margin_data import SEED
import difficulty_pomdp as base
from context_pomdp import encode,mentor_comparison,outcome,REVIEW_WIDTH
from scoring import correct


def predictor(x,y,leaf,seed):
    if not y:return None
    from sklearn.ensemble import ExtraTreesClassifier
    model=ExtraTreesClassifier(n_estimators=64,min_samples_leaf=leaf,max_depth=10,
                              max_features=.7,criterion='log_loss',n_jobs=-1,random_state=seed)
    return model.fit(np.asarray(x),np.asarray(y)).set_params(n_jobs=1)


def success(model,x):
    return base.probabilities(model,x)[:,1]


@dataclass
class Planner(base.Planner):
    regular_predictors: list = field(default_factory=list)
    reviewed_predictors: list = field(default_factory=list)
    context_weight: float = .5
    _current_context: dict = field(default_factory=dict,repr=False)
    _review_context: dict = field(default_factory=dict,repr=False)

    def context(self,row,path,tree,stage,reviewed,weight):
        node=tree[path];key=row['id'],path,node['prompt_sha256'],node['text'],weight
        cache=self._review_context if reviewed else self._current_context
        if key in cache:return cache[key]
        d=self.dynamics[row['subject']]
        x=np.asarray([base.visible(row,path,tree)])
        if reviewed:
            x=np.concatenate((np.repeat(x,REVIEW_WIDTH,axis=0),np.eye(REVIEW_WIDTH)),axis=1)
            transition=d.reviewed_transition[stage].copy()
            heads=self.reviewed_predictors
        else:
            transition=d.transition[stage].copy()[None]
            heads=self.regular_predictors
        for action in (0,1):
            for state in range(4):
                head=heads[action][state]
                if head is None:continue
                h=state//2
                predicted=success(head,x)
                pooled=transition[:,action,state,2*h+1]
                p=np.clip(weight*predicted+(1-weight)*pooled,1e-6,1-1e-6)
                transition[:,action,state,2*h]=1-p
                transition[:,action,state,2*h+1]=p
        result=transition if reviewed else transition[0]
        cache[key]=result
        return result


def fit(rows,trees,judges,leaf=12,smoothing=32.,late_feedback=False,stage_review=True,
        transition_leaf=24,context_weight=.5,initial_model=None,review_signals=None):
    initial=base.fit(rows,trees,judges,leaf,smoothing,late_feedback,stage_review) if initial_model is None else initial_model
    hard={r['id']:int(np.mean([correct(trees[r['id']][p],r['answer']) for p in base.FIXED])<.6) for r in rows}
    xs=[[[] for _ in range(4)] for _ in (0,1)];ys=[[[] for _ in range(4)] for _ in (0,1)]
    rx=[[[] for _ in range(4)] for _ in (0,1)];ry=[[[] for _ in range(4)] for _ in (0,1)]
    for row in rows:
        tree=trees[row['id']]
        for path in ('','repair','retry','guided'):
            current=base.visible(row,path,tree)
            state=2*hard[row['id']]+int(correct(tree[path],row['answer']))
            signal=initial.review_signal(row,path,tree,judges[row['id'],path]) if review_signals is None else review_signals[row['id'],path]
            reviewed=current+np.eye(REVIEW_WIDTH)[signal].tolist()
            for action,name in enumerate(('repair','retry')):
                child=f'{path}/{name}' if path else name
                xs[action][state].append(current)
                ys[action][state].append(int(correct(tree[child],row['answer'])))
                target=('guided' if action==0 else 'retry') if not path else f'{path}/{name}'
                if path and action==0 and late_feedback:target=path+'/reviewed'
                rx[action][state].append(reviewed)
                ry[action][state].append(int(correct(tree[target],row['answer'])))
    regular=[[predictor(xs[a][s],ys[a][s],transition_leaf,SEED+20+a*4+s) for s in range(4)] for a in (0,1)]
    reviewed=[[predictor(rx[a][s],ry[a][s],transition_leaf,SEED+40+a*4+s) for s in range(4)] for a in (0,1)]
    receipt={**initial.fit_receipt,'contextual_transitions':True,'transition_leaf':transition_leaf,
             'context_weight':context_weight,'predictor_trees':64,
             'future_context':'Current action transitions use observed context; unknown future contexts use pooled learned dynamics.'}
    return Planner(initial.classifier,initial.dynamics,receipt,
                   review_observer=initial.review_observer,
                   regular_predictors=regular,reviewed_predictors=reviewed,context_weight=context_weight)


def evaluate(row,tree,judges,model,settings,method):
    if method not in ('A','C','always','never'):raise ValueError('Contextual POMDP implements A/C only')
    d=model.dynamics[row['subject']]
    penalty=settings['penalty']*(settings.get('science_weight',1.) if row['domain']=='science' else 1.)
    reuse=settings.get('reuse_review',True);future=method=='C' and settings.get('plan_reviews',False)
    threshold=settings.get('threshold',0.);weight=settings.get('context_weight',model.context_weight)
    observed=model.observed(row,'',tree);belief=d.update(d.prior,d.initial[:,observed])
    path='';visited=[''];acquired=[];decisions=[]
    for remaining in (2,1):
        stage=2-remaining;condition=model.observed(row,path,tree)
        transition=model.context(row,path,tree,stage,False,weight)
        values=d.values(belief,remaining,penalty,transition,future_reviews=future,threshold=threshold)
        information=None;query=method=='always';review_transitions=None
        if method=='C':
            review_transitions=model.context(row,path,tree,stage,True,weight)
            expected=0.
            likelihood=d.likelihood(stage,condition)
            for signal in range(REVIEW_WIDTH):
                mass=belief*likelihood[:,signal]
                expected+=d.values(mass,remaining,penalty,review_transitions[signal],True,reuse,future,threshold).max()
            expected-=penalty*(belief@d.review_costs[stage])/1000
            information=float(expected-values.max());query=bool(information>threshold)
        judge=None
        if query:
            if (row['id'],path) not in judges:raise base.MissingObservation('judge',path)
            judge=judges[row['id'],path];acquired.append(path);signal=model.review_signal(row,path,tree,judge)
            belief=d.update(belief,d.likelihood(stage,condition)[:,signal])
            if review_transitions is None:review_transitions=model.context(row,path,tree,stage,True,weight)
            transition=review_transitions[signal]
            values=d.values(belief,remaining,penalty,transition,True,reuse,future,threshold)
        action=('stop','repair','retry')[int(values.argmax())]
        decisions.append({'path':path,'belief':belief.tolist(),'action':action,'queried':query,
                          'information_value':information,'contextual_transition':True})
        if action=='stop':break
        edge='guided' if query and not path and action=='repair' else action
        if query and path and action=='repair' and settings.get('late_feedback',False):edge='reviewed'
        path=f'{path}/{edge}' if path else edge
        if remaining==1 and hasattr(getattr(model,'classifier',None),'row_probabilities'):
            if path not in tree:raise base.MissingObservation('solver',path)
            visited.append(path)
            break
        observed=model.observed(row,path,tree);visited.append(path)
        comparison=mentor_comparison(judge,tree[path]) if query and reuse else None
        belief=d.next_belief(belief,transition,('repair','retry').index(action),observed,stage,query,comparison)
    return outcome(row,tree,judges,path,visited,acquired,decisions)
