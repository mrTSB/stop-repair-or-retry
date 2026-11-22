"""TRAIN-only whole-policy comparison of old768 and expanded1280 science fits.

Use one fixed, previously studied forest/planner recipe. Outer held questions
are excluded from forecasts, priors, transitions, emissions and costs. Inner
question folds produce emission forecasts without their own fitting examples.
This exploratory diagnostic assesses action accuracy/cost, not just log loss;
it does not revise the earlier likelihood screen or establish a fresh result.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import fit_expanded1280_completion as source
import clustered_completion_pomdp as clustered
import continuous_completion_pomdp as policy

io,fixed,nested,forecast,runtime=source.io,source.fixed,source.nested,source.forecast,source.runtime
SCHEMA='fixed-recipe-whole-POMDP-crossfit-TRAIN-v1'
SETTING={'max_depth':8,'min_samples_leaf':24}
POLICY={'penalty':.02,'posterior_weight':.5}
VARIANTS=(('retained768_dense',768,nested.FAMILIES[1]),
          ('expanded1280_raw',1280,nested.FAMILIES[0]),
          ('expanded1280_dense',1280,nested.FAMILIES[1]))


@dataclass
class CachedObserver:
    _recipe:dict
    values_by_key:dict
    dense_by_key:dict
    response_by_key:dict

    def values(self,row,path,paid):
        if row['answer']!='' or set(paid)!={'',path}:raise ValueError('Only blank-reference paid prefixes')
        for p,node in paid.items():
            if fixed.fingerprint(node)!=self.response_by_key[row['id'],p]:raise ValueError('Changed cached paid response')
        return self.values_by_key[row['id'],path]

    def features(self,row,path,paid):
        self.values(row,path,paid);return self.dense_by_key[row['id'],path]

    def charges(self,row,paid,judges,decisions,include_reviews):
        if judges or include_reviews:raise ValueError('A has no LLM review')
        paths=[d['path'] for d in decisions]
        if len(set(paths))!=len(paths):raise ValueError('Duplicate observation charge')
        slots=sum(self.values(row,p,{q:paid[q] for q in ('',p)})[3] for p in paths)
        return slots,len(paths),slots,self._recipe['parameters']/1e9


def inputs(args):
    fit_spec=io.read(args.expanded_fit/'registration.json');done=io.read(args.expanded_fit/'complete.json')
    fit_args=SimpleNamespace(**{k:Path(v) for k,v in fit_spec['paths'].items()})
    rows,trees,observer,data,_,expected=source.inputs(fit_args)
    if (fit_spec!=expected or done['schema']!=source.SCHEMA
            or done['registration_sha256']!=io.sha(args.expanded_fit/'registration.json')
            or done['development_or_test_read']):raise ValueError('Changed sealed1280 training fit')
    io.verify(fit_spec['source_files']);io.verify(fit_spec['input_evidence'])
    io.verify({str(args.expanded_fit/n):h for n,h in done['digests'].items()})
    second_spec=io.read(fit_args.second_features/'registration.json')
    original_neural=Path(second_spec['paths']['original_neural'])
    cache={};evidence=dict(fit_spec['input_evidence'])
    for folder,filename in ((original_neural,'dense-neural.npz'),(fit_args.first_features,'neural.npz'),(fit_args.second_features,'neural.npz')):
        metadata=[io.json.loads(s) for s in (folder/'metadata.jsonl').read_bytes().splitlines()]
        with np.load(folder/filename,allow_pickle=False) as values:
            for i,row in enumerate(metadata):
                key=row['id'],row['path']
                if key in cache:raise ValueError('Repeated frozen feature key')
                cache[key]=(*[values[k][i].copy() for k in ('state','plain','guided')],int(values['token_slots'][i]))
        for name in ('metadata.jsonl',filename):evidence[str((folder/name).resolve())]=io.sha(folder/name)
    keys=[(r['id'],r['path']) for r in data['records']]
    if set(cache)!=set(keys) or len(keys)!=3840:raise ValueError('Require all aligned1280x3 observations')
    cached=CachedObserver(observer._recipe,cache,dict(zip(keys,data['features'][nested.FAMILIES[1]])),
        {key:r['paid_response_sha256'] for key,r in zip(keys,data['records'])})
    by_id={r['id']:r for r in rows}
    for i,(qid,path) in enumerate(keys):
        public={**by_id[qid],'answer':''};paid={p:trees[qid][p] for p in ('',path)}
        actual=runtime.core.plain.observables(public,path,paid,cached.values(public,path,paid))
        np.testing.assert_array_equal(actual,data['features'][nested.FAMILIES[0]][i])
    evidence[str((args.expanded_fit/'complete.json').resolve())]=io.sha(args.expanded_fit/'complete.json')
    record={'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{'expanded_fit':str(args.expanded_fit.resolve()),'output':str(args.output.resolve())},
        'training_questions':1280,'common_evaluation_questions':768,'variants':[list(v) for v in VARIANTS],
        'forest_setting':SETTING,'policy_setting':POLICY,'trees':forecast.TREES,'prior_weight':forecast.PRIOR_WEIGHT,
        'outer_folds':3,'inner_emission_folds':3,'hyperparameter_search':False,
        'kernel':'24valid/6invalid clustered TRAIN child forecasts','smoothing':32.,
        'screen':'At least2 percentage points higher held-question science accuracy and no higher mean compute than old768 dense control',
        'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,'fresh_validation':False,
        'scope':'Exploratory TRAIN comparison; the fixed recipe was studied previously; no change to previous screens',
        'labels_sha256':fixed.array_sha(data['labels']),'feature_sha256':{k:fixed.array_sha(v) for k,v in data['features'].items()},
        'records_sha256':fixed.fingerprint(data['records']),'maximum_solver_continuations':2}
    return rows,trees,cached,data,record


def fit_fold(rows,trees,observer,data,count,family,fold):
    outer=np.array([r['fold'] for r in data['records']]);inner=np.array([nested.inner_fold(r['id']) for r in data['records']])
    eligible=np.arange(len(outer))<3*count;train=eligible&(outer!=fold)
    held=(np.arange(len(outer))<2304)&(outer==fold)
    receipt=nested.group_receipt(data['records'],train,held);receipt['emission_folds']=[]
    p=np.full((len(outer),16),np.nan)
    for inner_fold in range(3):
        fit=train&(inner!=inner_fold);predict=train&(inner==inner_fold)
        check=nested.group_receipt(data['records'],fit,predict)
        if (fit&held).any() or (predict&held).any():raise ValueError('Outer holdout entered emission fitting')
        model=forecast.fit(data['features'][family][fit],data['labels'][fit],SETTING)
        p[predict]=nested.predict(model,data['features'][family][predict],data['valid'][predict])
        receipt['emission_folds'].append(check)
    if not np.isfinite(p[train]).all() or np.isfinite(p[~train]).any():raise ValueError('Emission prediction scope changed')
    forecaster=forecast.fit(data['features'][family][train],data['labels'][train],SETTING)
    ids={r['id'] for r,flag in zip(data['records'],train) if flag}
    train_rows=[r for r in rows if r['id'] in ids];train_trees={r['id']:trees[r['id']] for r in train_rows}
    records=[r for r,flag in zip(data['records'],train) if flag]
    model=runtime.fit_dynamics(train_rows,train_trees,observer,records,p[train],forecaster,family,receipt)
    model=clustered.rebin(model,train_rows,train_trees,records,p[train])
    held_ids={r['id'] for r,flag in zip(data['records'],held) if flag}
    result={r['id']:policy.evaluate_science(r,trees[r['id']],{},model,POLICY,'A') for r in rows if r['id'] in held_ids}
    if set(result)&ids:raise ValueError('Evaluated a fitting question')
    return result,receipt


def statistics(items):
    return {'n':len(items),'correct':sum(int(v['correct']) for v in items),
        'accuracy':float(np.mean([v['correct'] for v in items])),
        'mean_compute':float(np.mean([v['compute'] for v in items]))}


def diagnose(args,rows,trees,observer,data,record):
    results={};receipts={};ids=[r['id'] for r in rows[:768]]
    for name,count,family in VARIANTS:
        results[name]={};receipts[name]=[]
        for fold in range(3):
            items,receipt=fit_fold(rows,trees,observer,data,count,family,fold)
            if set(items)&set(results[name]):raise ValueError('Duplicate held question')
            results[name].update(items);receipts[name].append(receipt)
            print({'variant':name,'outer_fold_complete':fold,'held_correct':sum(v['correct'] for v in items.values()),'held_questions':len(items)},flush=True)
        if set(results[name])!=set(ids):raise ValueError('Missing common held questions')
    control=results['retained768_dense'];reports={};draws=np.random.default_rng(238031).integers(0,768,(2000,768))
    for name,_,_ in VARIANTS:
        items=results[name];accuracy=np.array([int(items[k]['correct'])-int(control[k]['correct']) for k in ids])
        compute=np.array([items[k]['compute']-control[k]['compute'] for k in ids])
        reports[name]={'metrics':statistics(list(items.values())),
            'accuracy_delta':float(accuracy.mean()),'accuracy_delta_CI95':np.quantile(accuracy[draws].mean(1),[.025,.975]).tolist(),
            'compute_delta':float(compute.mean()),'compute_delta_CI95':np.quantile(compute[draws].mean(1),[.025,.975]).tolist(),
            'screen_passed':bool(name!='retained768_dense' and accuracy.mean()>=.02 and compute.mean()<=0)}
    io.write(args.output/'private-held-items.json',results);io.write(args.output/'fold-receipts.json',receipts)
    io.write(args.output/'summary.json',{'schema':SCHEMA,'reports':reports,'common_questions':768,
        'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,'fresh_validation':False,
        'uncertainty_scope':'Descriptive paired question intervals with fitted folds fixed'})
    io.verify(record['source_files']);io.verify(record['input_evidence'])
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'any_screen_passed':any(v['screen_passed'] for v in reports.values()),
        'digests':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'reports':reports},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','diagnose'))
    for name in ('expanded-fit','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');args=parser.parse_args()
    with fixed.training_only_reads(),fixed.guarded.no_model_calls(),threadpool_limits(limits=1):
        bundle=inputs(args);record=bundle[-1]
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json')});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Changed registered policy comparison')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json','.run.lock') for p in args.output.iterdir()):raise ValueError('Preserve prior outputs')
            diagnose(args,*bundle)


if __name__=='__main__':main()
