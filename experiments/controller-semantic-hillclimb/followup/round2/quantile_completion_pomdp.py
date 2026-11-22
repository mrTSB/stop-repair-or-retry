"""Complete16 POMDP with TRAIN-quantile finite observation thresholds.

Only the observation partition and corresponding emission counts change.
Forecasts, transitions, rewards, costs, state count and Bellman backup remain
unchanged. Thresholds use excluded-question TRAIN child forecasts, not labels.
"""
import copy
from dataclasses import dataclass,field

import numpy as np

import regularized_completion_pomdp as previous
import completion_subject_pomdp as routed

core=previous.core


@dataclass
class Kernel:
    current_edges:np.ndarray
    gain_edges:np.ndarray

    def signal(self,p,valid):
        p=core.mask_invalid(p,valid);gains=core.AFTER@p;best=int(np.argmax(gains))
        quality=int(gains[best]>=self.gain_edges[best])
        if not valid:return 24+2*best+quality
        return 6*int(np.searchsorted(self.current_edges,float(p@core.CURRENT)))+2*best+quality


def fit_kernel(records,probability,valid):
    p=np.asarray(probability);valid=np.asarray(valid,bool)
    if p.shape!=(len(records),16) or valid.shape!=(len(records),) or not np.isfinite(p).all():raise ValueError('Invalid aligned forecasts')
    child=np.asarray([r['path'] in ('repair','retry') for r in records])
    if not child.any() or not (child&valid).any():raise ValueError('Require valid TRAIN child observations')
    gains=p@core.AFTER.T;best=gains.argmax(1)
    edges=np.quantile((p@core.CURRENT)[child&valid],[.25,.5,.75])
    thresholds=[]
    for action in range(3):
        selected=child&(best==action)
        # A rare best-action category uses its marginal over all child inputs.
        if selected.sum()<16:selected=child
        thresholds.append(float(np.median(gains[selected,action])))
    return Kernel(edges,np.asarray(thresholds))


@dataclass
class Planner(previous.Planner):
    kernel:object=field(default=None)


def rebin(base,rows,trees,records,probability,kernel=None):
    expected=[(r['id'],p) for r in rows for p in ('','repair','retry')]
    if [(r['id'],r['path']) for r in records]!=expected:raise ValueError('Forecast ordering changed')
    valid=np.asarray([bool(trees[qid][p]['answer']) for qid,p in expected])
    kernel=fit_kernel(records,probability,valid) if kernel is None else kernel
    forecasts={key:p for key,p in zip(expected,probability)}
    support=np.ones((2,16,core.N_OBS));support[:,core.CURRENT==1,24:]=0.
    def counts(group):
        emission=.1*support.copy()
        for row in group:
            for action,path in enumerate(('repair','retry')):
                tree=trees[row['id']];paths=[path,*[path+'/'+a for a in ('repair','retry','complete')]]
                state=core.state(*[core.plain.correct(tree[p],row['answer']) for p in paths])
                observation=kernel.signal(forecasts[row['id'],path],bool(tree[path]['answer']))
                emission[action,state,observation]+=1
        return emission
    norm=lambda x:x/x.sum(-1,keepdims=True)
    pooled=norm(counts(rows));dynamics=copy.deepcopy(base.dynamics);smoothing=base.fit_receipt['smoothing']
    for subject,d in dynamics.items():d.emission=norm(counts([r for r in rows if r['subject']==subject])+smoothing*pooled)
    receipt={**base.fit_receipt,'observation_partition':'Pooled TRAIN child-current quartiles and per-best-action gain medians',
        'kernel_current_edges':kernel.current_edges.tolist(),'kernel_gain_edges':kernel.gain_edges.tolist(),
        'kernel_uses_reference_labels':False,'emissions_refit_to_new_partition':True,
        'unchanged':['forecaster','prior','transition','cost','states','exact finite Bellman backup']}
    return Planner(copy.deepcopy(base.classifier),copy.deepcopy(base.forecaster),dynamics,receipt,base.feature_family,kernel)


def evaluate_science(row,tree,judges,model,settings,family):
    if family!='A' or row['subject'] not in ('chemistry','physics'):raise ValueError('Science POMDP A only')
    settings=settings.get('by_subject',{}).get(row['subject'],settings)
    d,penalty=model.dynamics[row['subject']],settings['penalty'];path='';visited=[''];decisions=[]
    public={**row,'answer':''};belief=model.predicted(public,'',tree)
    for remaining in (2,1):
        action=int(d.values(belief,remaining,penalty).argmax())
        decisions.append({'path':path,'belief':belief.tolist(),'action':core.ACTIONS[action],'queried':False,'information_value':0.})
        if action==0:break
        path=(path+'/' if path else '')+core.ACTIONS[action]
        if path not in tree:raise core.plain.MissingObservation('solver',path)
        visited.append(path)
        if action==3 or remaining==1:break
        p=model.predicted(public,path,tree);obs=model.kernel.signal(p,bool(tree[path]['answer']))
        belief=core.mask_invalid((belief@d.transition[action-1])*d.emission[action-1,:,obs],bool(tree[path]['answer']))
    item=core.plain.outcome(row,tree,{},path,visited,[],decisions)
    paid={v['path']:tree[v['path']] for v in decisions}
    tokens,calls,slots,parameters=model.classifier.charges(public,paid,{},decisions,include_reviews=False)
    item.update(verifier_tokens=tokens,verifier_calls=calls,verifier_token_slots=slots,
        verifier_parameters_billions=parameters,completion_calls=int(path.endswith('complete')),completion_is_new_action=True)
    item['compute']+=parameters*slots/3.09;item['total_parameter_compute']+=parameters*slots/3.09
    return item


SubjectPlanner=routed.Planner


def evaluate(row,tree,judges,model,settings,family):
    if family!='A':raise ValueError('Quantile observation A never requests a review')
    if row['subject']=='math':return routed.evaluate(row,tree,judges,model,settings,family)
    return evaluate_science(row,tree,{},model.science,model.science_settings,family)
