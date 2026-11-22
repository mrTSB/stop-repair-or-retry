"""Fixed half-posterior child estimator over the unchanged Complete POMDP.

The finite transition/cost/Bellman model stays intact. The half blend is an
explicit approximate paid-history estimator, not an exact Bayes update or a
new continuous observation likelihood. Root forecasts and math are unchanged.
"""
from dataclasses import dataclass
import copy

import completion_pomdp as core
import completion_subject_pomdp as parent_policy

LIMITATION=('The half blend is an approximate discriminative paid-history belief estimator. '
    'It retains learned transitions, support, costs and the finite Bellman class, but its '
    'actual child update is not the posterior assumed by that finite-observation backup. '
    'No exact-Bayes or continuous-likelihood claim; evaluation is development selection, '
    'not fresh validation.')

WEIGHT=.5
PENALTY=.02


@dataclass
class Planner(parent_policy.Planner):
    posterior_weight: float=WEIGHT


def prefix(tree,path):
    return {p:tree[p] for p in ('',path)}


def update(model,row,path,tree,prior,action,weight):
    if not 0<=weight<=1:raise ValueError('Posterior weight must lie in[0,1]')
    paid=prefix(tree,path);safe={**row,'answer':''};d=model.dynamics[row['subject']]
    posterior=model.predicted(safe,path,paid)
    obs=core.observation(posterior,bool(paid[path]['answer']))
    finite=core.mask_invalid((prior@d.transition[action])*d.emission[action,:,obs],bool(paid[path]['answer']))
    if weight==0:return finite
    return core.mask_invalid((1-weight)*finite+weight*posterior,bool(paid[path]['answer']))


def science_evaluate(row,tree,model,penalty,weight):
    if row['subject'] not in ('chemistry','physics'):raise ValueError('Science-only Complete branch')
    d=model.dynamics[row['subject']];path,visited,decisions='', [''], []
    safe={**row,'answer':''};belief=model.predicted(safe,'',prefix(tree,''))
    for remaining in (2,1):
        action=int(d.values(belief,remaining,penalty).argmax())
        decisions.append({'path':path,'belief':belief.tolist(),'action':core.ACTIONS[action],
            'queried':False,'information_value':0.})
        if not action:break
        path=(path+'/' if path else '')+core.ACTIONS[action]
        if path not in tree:raise core.plain.MissingObservation('solver',path)
        visited.append(path)
        if action==3 or remaining==1:break
        belief=update(model,safe,path,tree,belief,action-1,weight)
    item=core.plain.outcome(row,tree,{},path,visited,[],decisions)
    paid={decision['path']:tree[decision['path']] for decision in decisions}
    tokens,calls,slots,parameters=model.classifier.charges(safe,paid,{},decisions,include_reviews=False)
    extra=parameters*slots/3.09
    item.update(verifier_tokens=tokens,verifier_calls=calls,verifier_token_slots=slots,
        verifier_parameters_billions=parameters,completion_calls=int(path.endswith('complete')),completion_is_new_action=True)
    item['compute']+=extra;item['total_parameter_compute']+=extra
    return item


def build(parent,provenance):
    if (type(parent) is not parent_policy.Planner or type(parent.science) is not core.Planner
        or parent.science_settings!={'penalty':PENALTY}
        or parent.science.fit_receipt.get('state_count')!=16
        or parent.science.fit_receipt.get('state_bits')!=['current_correct','repair_correct','retry_correct','complete_correct']):
        raise ValueError('Require the exact fixed16-state parent and selected.02 science penalty')
    return Planner(copy.deepcopy(parent.math),copy.deepcopy(parent.science),copy.deepcopy(parent.math_settings),
        copy.deepcopy(parent.science_settings),{'schema':'fixed-half-Complete-POMDP-v1',
            'parent':parent.fit_receipt,'provenance':provenance,'science_state_count':16,
            'posterior_weight':WEIGHT,'science_penalty':PENALTY,'full_TRAIN_parent_forecaster_dynamics_unchanged':True,
            'math_route_unchanged':True,'root_forecast_unchanged':True,'finite_Bellman_backup_unchanged':True,
            'approximate_belief_estimator':True,'limitation':LIMITATION,
            'prospective_support':'New action-aware module; old collector requires explicit support'},WEIGHT)


def evaluate(row,tree,judges,model,settings,family):
    if family!='A' or settings or model.posterior_weight!=WEIGHT or model.science_settings!={'penalty':PENALTY}:
        raise ValueError('Only registered fixed-half/.02 A settings are supported')
    if row['subject']=='math':return parent_policy.evaluate(row,tree,{},model,{},'A')
    return science_evaluate(row,tree,model.science,PENALTY,WEIGHT)
