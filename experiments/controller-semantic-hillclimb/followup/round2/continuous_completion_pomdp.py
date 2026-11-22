"""Complete16 planning with an approximate paid-history posterior blend.

Future values keep the TRAIN-fitted finite-observation Bellman backup. The
realized child belief blends that filter with the full sixteen-state forecast
of the paid root and child. This is an approximate estimator, not an exact
continuous-emission Bayesian filter. No extra observation or action is added.
"""
import numpy as np
import completion_subject_pomdp as routed
import completion_pomdp as core


def update(model, subject, prior, action, posterior, valid, weight):
    if weight not in (0., .5, 1.): raise ValueError('Use a registered posterior blend')
    d = model.dynamics[subject]
    p = core.mask_invalid(posterior, valid)
    obs = model.kernel.signal(p, valid)
    finite = core.mask_invalid((prior @ d.transition[action]) * d.emission[action, :, obs], valid)
    if weight == 0.: return finite
    return core.mask_invalid((1.-weight)*finite+weight*p, valid)

def evaluate_science(row,tree,judges,model,settings,family):
    if family!='A' or row['subject'] not in ('chemistry','physics'):raise ValueError('Science POMDP A only')
    settings=settings.get('by_subject',{}).get(row['subject'],settings)
    weight=settings['posterior_weight']
    if weight not in (0., .5, 1.):raise ValueError('Use a registered posterior blend')
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
        p=model.predicted(public,path,tree)
        belief=update(model,row['subject'],belief,action-1,p,bool(tree[path]['answer']),weight)
    item=core.plain.outcome(row,tree,{},path,visited,[],decisions)
    paid={v['path']:tree[v['path']] for v in decisions}
    tokens,calls,slots,parameters=model.classifier.charges(public,paid,{},decisions,include_reviews=False)
    item.update(verifier_tokens=tokens,verifier_calls=calls,verifier_token_slots=slots,
        verifier_parameters_billions=parameters,completion_calls=int(path.endswith('complete')),completion_is_new_action=True)
    item['compute']+=parameters*slots/3.09;item['total_parameter_compute']+=parameters*slots/3.09
    return item


SubjectPlanner=routed.Planner


def evaluate(row,tree,judges,model,settings,family):
    if family!='A':raise ValueError('Continuous Complete A never requests a review')
    if row['subject']=='math':return routed.evaluate(row,tree,judges,model,settings,family)
    return evaluate_science(row,tree,{},model.science,model.science_settings,family)
