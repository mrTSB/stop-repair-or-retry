"""Nested TRAIN-only finite filter versus fixed approximate half-posterior blend.

The finite control uses the original16-state transition/emission/cost formulas.
The blend is an approximate paid-history estimator, NOT exact Bayes. Its actual
belief update differs from the unchanged finite-observation Bellman model; this
diagnostic scores beliefs only and makes no policy or fresh-validation claim.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
from pathlib import Path
import pickle
import sys
import time
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

import completion_pomdp as core
import completion_v4_diagnostic as fixed
import extract_dense_completion as inputs
import expanded_mentor_io as io

SCHEMA='completion-nested-finite-vs-half-belief-TRAIN-v1'
METHODS=('finite_POMDP_filter','approximate_half_posterior_blend')
INNER_NAMESPACE='completion-filter-inner-v1'
REGULARIZATION=.1
SMOOTHING=32.
WEIGHT=.5
LIMITATION=('The half blend is an approximate discriminative paid-history belief estimator. '
    'It retains learned transitions, support, costs and the finite Bellman class, but its actual '
    'child update is not the posterior assumed by that finite-observation backup. No exact-Bayes '
    'or continuous-likelihood claim; this diagnostic does not run or select a policy.')


def inner_fold(qid,outer):
    return int(hashlib.sha256(f'{INNER_NAMESPACE}:{outer}:{qid}'.encode()).hexdigest()[:8],16)%3


def load_inputs(args):
    bundle=inputs.training_bundle(args.base_fit)
    lookup={}
    for record,logits in zip(bundle['metadata'],bundle['old_raw20']):
        key=record['id'],record['path']
        if key in lookup:raise ValueError('Duplicate old neural input')
        lookup[key]=(logits[:4],logits[4:12],logits[12:20],0)
    table=fixed.table(bundle['rows'],bundle['trees'],lookup,lookup)
    encoder_price=bundle['recipe']['parameters']/1e9*bundle['recipe']['max_length']/3.09
    costs=[]
    for record in table['records']:
        tree=bundle['trees'][record['id']];path=record['path'];prefix=path+'/' if path else ''
        values=[]
        for action,edge in enumerate(('repair','retry','complete')):
            node=tree[prefix+edge]
            value=4*(node['input_tokens']+node['output_tokens'])/3.09
            if not path and action<2:value+=encoder_price
            values.append(value)
        costs.append(values)
    data={'records':table['records'],'features':table['features'][fixed.VARIANTS[0]],
        'labels':table['labels'],'valid':table['valid'],'costs':np.asarray(costs)}
    if len(data['records'])!=768 or len({r['id'] for r in data['records']})!=256:
        raise ValueError('Use the complete256 science/768 paid-node TRAIN table')
    receipt={'input_evidence':bundle['evidence'],'source_files':io.sources(sys.modules[__name__]),
        'paths':{'base_fit':str(args.base_fit.resolve()),'output':str(args.output.resolve())},
        'data_sha256':{k:(fixed.fingerprint(v) if k=='records' else fixed.array_sha(v)) for k,v in data.items()},
        'old_neural_alignment_sha256':bundle['old_alignment_sha256'],
        'encoder_recipe_sha256':fixed.sha(Path(bundle['observer'].directory)/'recipe.json')}
    return data,receipt


def probabilities(model,x,valid):
    out=np.full((len(x),16),1e-9)
    out[:,model.classes_.astype(int)]+=model.predict_proba(x)
    return np.asarray([core.mask_invalid(p,v) for p,v in zip(out,valid)])


def fit_dynamics(data,training_ids,signals):
    """Original Complete fit count/support/shrinkage formulas, restricted to IDs."""
    records=data['records'];index={(r['id'],r['path']):i for i,r in enumerate(records)}
    if len(index)!=len(records):raise ValueError('Duplicate paired node')
    subjects={r['id']:r['subject'] for r in records}
    if (not training_ids or any((qid,p) not in index for qid in training_ids for p in ('','repair','retry'))
            or set(signals)!={(qid,p) for qid in training_ids for p in ('','repair','retry')}):
        raise ValueError('Only complete outer-training question signals may fit dynamics')
    support=np.asarray([[[float(core.CURRENT[t]==core.AFTER[a,s]) for t in core.STATES]
        for s in core.STATES] for a in (0,1)])
    emission_support=np.ones((2,16,core.N_OBS));emission_support[:,core.CURRENT==1,24:]=0.
    def counts(group):
        prior=np.ones(16);transition=.1*support.copy();emission=.1*emission_support.copy()
        price,number=np.zeros((2,3,16)),np.zeros((2,3,16))
        for qid in group:
            root=int(data['labels'][index[qid,'']]);prior[root]+=1
            for action,path in enumerate(('repair','retry')):
                nxt=int(data['labels'][index[qid,path]])
                if core.CURRENT[nxt]!=core.AFTER[action,root]:raise ValueError('Paired current truth changed')
                transition[action,root,nxt]+=1;emission[action,nxt,signals[qid,path]]+=1
            for stage,paths in ((0,('',)),(1,('repair','retry'))):
                for path in paths:
                    i=index[qid,path];state=int(data['labels'][i])
                    price[stage,:,state]+=data['costs'][i];number[stage,:,state]+=1
        return prior,transition,emission,price,number
    def normalized(x):return x/x.sum(-1,keepdims=True)
    pooled=counts(sorted(training_ids));result={}
    for subject in ('chemistry','physics'):
        local=counts(sorted(q for q in training_ids if subjects[q]==subject))
        prior=core.normalize(local[0]+SMOOTHING*core.normalize(pooled[0]))
        transition=normalized(local[1]+SMOOTHING*normalized(pooled[1]))
        emission=normalized(local[2]+SMOOTHING*normalized(pooled[2]))
        pooled_cost=np.divide(pooled[3],pooled[4],out=np.full_like(pooled[3],650.),where=pooled[4]>0)
        cost=(local[3]+4*pooled_cost)/(local[4]+4)
        result[subject]=core.Dynamics(prior,transition,emission,cost)
    return result


def fit_fold(data,outer):
    records=data['records'];x,y,valid=data['features'],data['labels'],data['valid']
    held=np.asarray([r['fold']==outer for r in records]);train=~held
    training_ids={r['id'] for r,flag in zip(records,train) if flag}
    held_ids={r['id'] for r,flag in zip(records,held) if flag}
    if not training_ids or not held_ids or training_ids&held_ids:raise ValueError('Outer question overlap')
    inner=np.asarray([inner_fold(r['id'],outer) for r in records])
    if set(inner[train])!={0,1,2}:raise ValueError('Salted inner question folds must all be present')
    inner_prob=np.full((len(records),16),np.nan);inner_receipts=[]
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        for fold in range(3):
            fit=train&(inner!=fold);validation=train&(inner==fold)
            fit_ids={r['id'] for r,flag in zip(records,fit) if flag}
            validation_ids={r['id'] for r,flag in zip(records,validation) if flag}
            if fit_ids&validation_ids or (fit_ids|validation_ids)!=training_ids or (fit_ids|validation_ids)&held_ids:
                raise ValueError('Nested question exclusion failed')
            model=core.classifier_for(x[fit],y[fit],REGULARIZATION)
            inner_prob[validation]=probabilities(model,x[validation],valid[validation])
            inner_receipts.append({'fold':fold,'fit_questions':len(fit_ids),'validation_questions':len(validation_ids),
                'fit_ids_sha256':fixed.fingerprint(sorted(fit_ids)),
                'validation_ids_sha256':fixed.fingerprint(sorted(validation_ids))})
        forecaster=core.classifier_for(x[train],y[train],REGULARIZATION)
    if not np.isfinite(inner_prob[train]).all() or np.isfinite(inner_prob[held]).any():
        raise ValueError('Inner forecasts must cover outertraining only')
    signals={(r['id'],r['path']):core.observation(inner_prob[i],valid[i]) for i,r in enumerate(records) if train[i]}
    dynamics=fit_dynamics(data,training_ids,signals)
    held_probability=probabilities(forecaster,x[held],valid[held])
    receipt={'outer_fold':outer,'training_questions':len(training_ids),'held_questions':len(held_ids),
        'training_ids_sha256':fixed.fingerprint(sorted(training_ids)),
        'held_ids_sha256':fixed.fingerprint(sorted(held_ids)), 'inner_folds':inner_receipts,
        'inner_probabilities_sha256':fixed.array_sha(inner_prob[train]),
        'fit_seconds':time.perf_counter()-started,'warnings':[str(w.message) for w in observed],
        'held_labels_used_by_forecaster_dynamics_or_costs':False}
    return held,held_probability,forecaster,dynamics,receipt


def child_beliefs(dynamics,root,child,action,valid):
    observation=core.observation(child,valid)
    finite=core.mask_invalid((root@dynamics.transition[action])*dynamics.emission[action,:,observation],valid)
    blend=core.mask_invalid((1-WEIGHT)*finite+WEIGHT*child,valid)
    return finite,blend


def evaluate_fold(data,held,raw,dynamics):
    records=[r for r,flag in zip(data['records'],held) if flag]
    by_node={(r['id'],r['path']):p for r,p in zip(records,raw)}
    result={name:[] for name in METHODS}
    for record,p,valid in zip(records,raw,data['valid'][held]):
        if record['path']=='':finite=blend=p
        else:
            finite,blend=child_beliefs(dynamics[record['subject']],by_node[record['id'],''],p,
                ('repair','retry').index(record['path']),valid)
        result[METHODS[0]].append(finite);result[METHODS[1]].append(blend)
    return {name:np.asarray(value) for name,value in result.items()}


def reports(data,predictions):
    records=data['records'];child=np.asarray([bool(r['path']) for r in records]);root=~child
    if child.sum()!=512 or root.sum()!=256:raise ValueError('Require all256 roots and512 children')
    for name,p in predictions.items():
        if p.shape!=(768,16) or not np.isfinite(p).all():raise ValueError('Incomplete posterior diagnostics')
    np.testing.assert_array_equal(predictions[METHODS[0]][root],predictions[METHODS[1]][root])
    scores={name:fixed.losses(data['labels'],p) for name,p in predictions.items()}
    panels={}
    for subject in ('all','chemistry','physics'):
        for path in ('children','','repair','retry'):
            mask=np.asarray([(subject=='all' or r['subject']==subject) and
                (bool(r['path']) if path=='children' else r['path']==path) for r in records])
            panels[subject+'/'+(path or 'root')]={'nodes':int(mask.sum()),'methods':{
                name:{metric:float(value[mask].mean()) for metric,value in loss.items()} for name,loss in scores.items()}}
    ids=sorted({r['id'] for r in records})
    groups=[[i for i,r in enumerate(records) if r['id']==qid and r['path']] for qid in ids]
    if len(ids)!=256 or any(len(g)!=2 for g in groups):raise ValueError('Each question needs both held children')
    draws=np.random.default_rng(fixed.SEED).integers(0,len(ids),size=(fixed.BOOTSTRAPS,len(ids)))
    comparisons={}
    for metric in scores[METHODS[0]]:
        delta=scores[METHODS[1]][metric]-scores[METHODS[0]][metric]
        values=np.asarray([delta[g].mean() for g in groups]);bootstrap=values[draws].mean(1)
        comparisons[metric]={'blend_minus_finite':float(values.mean()),
            'paired_question_CI95':np.quantile(bootstrap,[.025,.975]).tolist()}
    return {'panels':panels,'child_comparisons':comparisons,
        'screen_passed':all(comparisons[m]['blend_minus_finite']<0 for m in ('joint_log_loss','mean_four_marginal_Brier')),
        'nested_forecaster_and_dynamics_question_exclusion':True,'approximate_blend_limitation':LIMITATION,
        'controller_accuracy_claim':False,'fresh_validation':False}


def specification(receipt):
    return {**receipt,'schema':SCHEMA,'training_questions':256,'paid_nodes':768,'primary_child_nodes':512,
        'methods':list(METHODS),'posterior_blend_weight':WEIGHT,'regularization':REGULARIZATION,'temperature':1.,
        'smoothing':SMOOTHING,'outer_folds':3,'inner_folds':3,'inner_namespace':INNER_NAMESPACE,
        'row_weight':1.,'equal_question_training_mass':3.,'equal_question_scoring_mass':'Mean of both paid children',
        'original_state_bits':['current_correct','repair_correct','retry_correct','complete_correct'],
        'preserved_formulas':'Complete16-state support, transition, emission, cost shrinkage; original Dynamics/Bellman class',
        'outer_training_only':'Forecaster/scaler, inner emission observations, transitions, priors and costs exclude outer held questions',
        'old_encoder_excluded_all_new512':True,'new_neural_calls':0,'new_cloud_calls':0,'development_or_test_read':False,
        'screen':'Lower child joint_log_loss AND lower mean4-marginal Brier; question-bootstrap CI descriptive',
        'bootstrap_draws':fixed.BOOTSTRAPS,'bootstrap_seed':fixed.SEED,'approximate_blend_limitation':LIMITATION,
        'root_scores':'Same root forecaster posterior for both; descriptive only',
        'policy_rollouts':False,'threshold_or_penalty_search':False,'runtime_model_fit':False,
        'environment':{'python':list(sys.version_info[:3]),'packages':{n:importlib.metadata.version(n) for n in ('numpy','scikit-learn')}}}


def run(args):
    with fixed.training_only_reads(),fixed.guarded.no_model_calls(),threadpool_limits(limits=1):
        data,receipt=load_inputs(args);registered=specification(receipt)
        if args.mode=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',registered)
            print({'registered':True,'registration_sha256':fixed.sha(args.output/'registration.json'),'scored':False},flush=True);return
        if (not args.registration_sha256 or fixed.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=registered):raise ValueError('Changed reviewed diagnostic registration')
        with io.lock(args.output/'.diagnostic.lock'):
            if any(p.name not in ('registration.json','.diagnostic.lock') for p in args.output.iterdir()):
                raise ValueError('Preserve earlier or partial diagnostic output')
            predictions={name:np.full((768,16),np.nan) for name in METHODS};receipts=[]
            for outer in range(3):
                held,raw,forecaster,dynamics,fold_receipt=fit_fold(data,outer)
                values=evaluate_fold(data,held,raw,dynamics)
                for name in METHODS:predictions[name][held]=values[name]
                path=args.output/f'outer-{outer}.pkl'
                io.atomic_bytes(path,pickle.dumps({'forecaster':forecaster,'dynamics':dynamics},protocol=5))
                receipts.append({**fold_receipt,'model_sha256':fixed.sha(path)})
            report=reports(data,predictions)
            io.npz(args.output/'private-oof.npz',predictions);io.write(args.output/'fold-receipts.json',receipts)
            for path,value in {**registered['source_files'],**registered['input_evidence']}.items():
                if fixed.sha(path)!=value:raise ValueError('Registered evidence changed during CPU-only fitting')
            io.write(args.output/'summary.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
                'reports':report,'training_questions':256,'child_nodes':512,'fold_receipts':receipts,
                'new_neural_calls':0,'new_cloud_calls':0,'development_or_test_read':False})
            io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
                'artifacts':{p.name:fixed.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
                'new_neural_calls':0,'new_cloud_calls':0,'development_or_test_read':False})
            print({'complete':True,'screen_passed':report['screen_passed'],'child_comparisons':report['child_comparisons']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','diagnose'))
    for name in ('base-fit','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');run(parser.parse_args())
