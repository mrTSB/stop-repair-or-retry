"""Static model uncertainty inside a genuine retained-review POMDP.

The queried hidden state is (dynamics-model index, original eight-state
mentor state). Every observation updates both state and model probability.
The unqueried branch retains the first parent's original A policy. Neural
inputs are restricted to the paid prefix; identical recipe/input forwards
share one charge, while distinct observations keep their complete bills.
"""
from dataclasses import dataclass
import numpy as np

import stronger_base_hybrid as core


def normalized(masses):
    values = np.asarray(masses,dtype=float)
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Finite nonnegative model/state masses required')
    total = float(values.sum())
    if total <= 0: raise ValueError('Impossible observation has no posterior')
    return values/total


def posterior(masses,likelihoods):
    likelihoods = np.asarray(likelihoods,dtype=float)
    if np.shape(masses) != likelihoods.shape: raise ValueError('Aligned likelihoods required')
    return normalized(np.asarray(masses)*likelihoods)


def values(dynamics,masses,signal,penalty):
    # Future actions must be shared across hidden models. Averaging each
    # parent's already-maximized value would give free model identity.
    result = np.array([sum(b@core.mentor_core.REWARD for b in masses),0.,0.])
    for action in (0,1):
        continuation = []
        for d,b in zip(dynamics,masses):
            predicted = b@d.reviewed_transition[signal,action]
            observed_masses = (predicted[:,None]*d.emission[action]).T
            continuation.append(observed_masses@d.alpha(action,penalty).T)
        future = np.asarray(continuation).sum(0).max(1).sum()
        cost = sum(b@d.first_costs[action] for d,b in zip(dynamics,masses))
        result[action+1] = future-penalty*cost/1000
    return result


def query_value(dynamics,masses,penalty):
    signals = {d.review.shape[-1] for d in dynamics}
    if len(signals) != 1: raise ValueError('All models must emit the same observation alphabet')
    future = sum(values(dynamics,np.asarray([b*d.review[:,s] for d,b in zip(dynamics,masses)]),s,penalty).max()
                 for s in range(next(iter(signals))))
    return float(future-sum(b@d.review_costs for d,b in zip(dynamics,masses))*penalty/1000)


@dataclass
class Planner:
    models: list
    weights: list
    fit_receipt: dict


@dataclass
class RoutedPlanner:
    models: dict
    experts: dict
    fit_receipt: dict


def routed(row,tree,judges,model,settings,family):
    if settings or family != 'C': raise ValueError('Fixed observed-subject hybrid routing required')
    expert = model.experts[row['subject']]
    child = model.models[expert['model']]
    module = expert['policy_module']
    if module == __name__: return evaluate(row,tree,judges,child,expert['settings'],family)
    if module not in ('stronger_base_hybrid','metadata_review_hybrid'): raise ValueError('Unknown POMDP expert')
    import development_expanded_mentor as runtime
    return runtime.dispatch(child,module,row,tree,judges,expert['settings'])


def evaluate(row,tree,judges,model,settings,family):
    if family != 'C' or len(model.models) != len(model.weights) or len(model.models) < 2:
        raise ValueError('Require a POMDP+LLM model ensemble')
    if not np.isfinite(model.weights).all() or min(model.weights) <= 0 or not np.isclose(sum(model.weights),1.):
        raise ValueError('Fixed positive prior model weights must sum to one')
    selected = settings.get('by_subject',{}).get(row['subject'],settings)
    penalty,threshold = float(selected['penalty']),float(selected.get('threshold',0.))
    if not np.isfinite([penalty,threshold]).all() or min(penalty,threshold) < 0:
        raise ValueError('Finite nonnegative penalty and query threshold required')
    safe = {**row,'answer':''}
    ledger,masses,dynamics,bases = {},[],[],[]
    for child,weight in zip(model.models,model.weights):
        base,base_settings = core.resolve_base(child.base,child.base_module,child.base_settings,row['subject'])
        child.prepare_observer(base.classifier)
        root = base.predicted(safe,'',core.prefix(tree,''))
        core.record_observation(ledger,base.classifier,safe,'',tree)
        masses.append(weight*core.action_core.normalize(root@child.transfer[row['subject']]))
        dynamics.append(child.mentor.dynamics[row['subject']])
        bases.append((base,base_settings,root))
    masses = normalized(masses)
    base,base_settings,root = bases[0]
    information = query_value(dynamics,masses,penalty)-base.dynamics[row['subject']].values(root,2,penalty).max()
    if information <= threshold+1e-9:
        fixed = {**base_settings,'penalty':penalty}
        function = core.continuous.evaluate if 'posterior_weight' in fixed else core.action_core.evaluate
        item = function(safe,tree,{},base,fixed,'A')
        item['correct'] = int(core.action_core.correct(tree[item['path']],row['answer']))
        item['decisions'][0]['information_value'] = float(information)
        for decision in item['decisions']: core.record_observation(ledger,base.classifier,safe,decision['path'],tree)
        return core.bill(item,ledger,{},row)
    if (row['id'],'') not in judges: raise core.action_core.MissingObservation('judge','')
    judge = judges[row['id'],'']
    signals = []
    for child in model.models:
        probability = None
        if child.review_observer is not None:
            child.prepare_observer(child.mentor.classifier)
            probability = core.mentor_core.perception.softmax(child.mentor.classifier.values(safe,'',core.prefix(tree,''),judge)[0],
                                                              child.mentor.classifier.temperature)
            core.record_observation(ledger,child.mentor.classifier,safe,'',tree,judge)
        signals.append(core.mentor_core.signal(judge,tree[''],probability))
    if len(set(signals)) != 1: raise ValueError('Ensemble parents must share the same signal adapter')
    signal = signals[0]
    masses = posterior(masses,[d.review[:,signal] for d in dynamics])
    action = int(values(dynamics,masses,signal,penalty).argmax())
    decisions = [{'path':'','belief':masses.ravel().tolist(),'model_posterior':masses.sum(1).tolist(),
        'action':('stop','repair','retry')[action],'queried':True,'information_value':float(information),'tracked_mentor':True}]
    path,visited = '',['']
    if action:
        first = action-1
        path = 'guided' if first == 0 else 'retry'
        if path not in tree: raise core.action_core.MissingObservation('solver',path)
        visited.append(path)
        observed = []
        for child in model.models:
            child.prepare_observer(child.mentor.classifier)
            observed.append(child.mentor.regular.observed(safe,path,core.prefix(tree,path)))
            core.record_observation(ledger,child.mentor.classifier,safe,path,tree)
        if len(set(observed)) != 1: raise ValueError('Ensemble parents must share child-observation semantics')
        comparison = core.mentor_core.comparison(judge,tree[path])
        predicted = np.asarray([b@d.reviewed_transition[signal,first] for d,b in zip(dynamics,masses)])
        likelihood = np.asarray([d.emission[first,:,observed[0]*3+comparison] for d in dynamics])
        if not tree[path]['answer']: likelihood[:,core.mentor_core.REWARD == 1] = 0.
        masses = posterior(predicted,likelihood)
        action = int(sum(d.alpha(first,penalty)@b for d,b in zip(dynamics,masses)).argmax())
        decisions.append({'path':path,'belief':masses.ravel().tolist(),'model_posterior':masses.sum(1).tolist(),
            'action':('stop','repair','retry')[action],'queried':False,'information_value':None,'tracked_mentor':True})
        if action:
            path += '/'+('reuse' if action == 1 else 'retry')
            if path not in tree: raise core.action_core.MissingObservation('solver',path)
            visited.append(path)
    item = core.mentor_core.outcome(row,tree,judges,path,visited,[''],decisions)
    return core.bill(item,ledger,judges,row)
