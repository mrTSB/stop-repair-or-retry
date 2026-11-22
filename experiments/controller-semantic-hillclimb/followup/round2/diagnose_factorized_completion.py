"""Separate nested TRAIN screen of factorized versus categorical beliefs.

Uses the completed768-question feature/fit archives, no DEV/TEST or neural call.
This diagnostic cannot promote a policy or report controller accuracy.
"""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import factorized_completion_forecast as forecast
import fit_auxiliary_completion as categorical

io,fixed,nested=categorical.io,categorical.fixed,categorical.nested
SCHEMA='expanded768-factorized-belief-TRAIN-diagnostic-v1'


def inputs(args):
    rows,trees,observer,data,_=categorical.inputs(SimpleNamespace(features=args.features,output=args.output))
    spec=io.read(args.categorical/'registration.json');done=io.read(args.categorical/'complete.json')
    if (spec['schema']!=categorical.SCHEMA or done['schema']!=categorical.SCHEMA
            or done['registration_sha256']!=io.sha(args.categorical/'registration.json') or done['training_questions']!=768
            or spec['paths']['features']!=str(args.features.resolve()) or done['development_or_test_read']
            or done['new_neural_calls'] or done['cloud_calls']
            or spec['records_sha256']!=fixed.fingerprint(data['records']) or spec['labels_sha256']!=fixed.array_sha(data['labels'])
            or spec['valid_sha256']!=fixed.array_sha(data['valid'])
            or spec['feature_sha256']!={k:fixed.array_sha(v) for k,v in data['features'].items()}):
        raise ValueError('Require exact completed categorical768 TRAIN comparison')
    io.verify(spec['source_files']);io.verify(spec['input_evidence'])
    io.verify({str(args.categorical/n):v for n,v in done['digests'].items()})
    with np.load(args.categorical/'private-oof.npz',allow_pickle=False) as archive:
        predictions={k:archive[k].copy() for k in nested.FAMILIES}
    evidence={**spec['input_evidence'],**{str((args.categorical/n).resolve()):v for n,v in done['digests'].items()},
        str((args.categorical/'complete.json').resolve()):io.sha(args.categorical/'complete.json')}
    record={'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{k:str(getattr(args,k).resolve()) for k in ('features','categorical','output')},
        'training_questions':768,'nodes':2304,'families':list(nested.FAMILIES),'regularization':list(forecast.CS),
        'outer_folds':3,'inner_folds':3,'selection_loss':'mean of four marginal Brier scores',
        'categorical_selection_loss':'joint log loss, retained without retuning',
        'conditional_independence_approximation':'Four binary correctness heads form a sixteen-state product belief',
        'screen':'Per family: lower nested mean-four Brier AND joint log loss at most110% of categorical; all comparisons retained',
        'bootstrap_draws':2000,'bootstrap_seed':238031,'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False}
    return data,predictions,record


def diagnose(args,data,control,record):
    with io.lock(args.output/'.run.lock'),threadpool_limits(limits=1):
        if any(p.name not in ('registration.json','.run.lock') for p in args.output.iterdir()):raise ValueError('Preserve partial/completed diagnostic')
        ids=sorted({r['id'] for r in data['records']});groups=[[i for i,r in enumerate(data['records']) if r['id']==qid] for qid in ids]
        if len(groups)!=768 or any(len(v)!=3 for v in groups):raise ValueError('All768 complete question groups required')
        draws=np.random.default_rng(238031).integers(0,768,size=(2000,768));reports={};receipts={};oof={}
        for family in nested.FAMILIES:
            p=np.full((2304,16),np.nan);receipts[family]=[]
            for fold in range(3):
                held,value,model,receipt=forecast.outer_prediction(data,family,fold);p[held]=value;receipts[family].append(receipt)
                io.atomic_bytes(args.output/(family+f'-outer{fold}.pkl'),pickle.dumps(model,protocol=5))
                print({'factorized_family':family,'outer_fold_complete':fold,'selected_C':receipt['selected_C']},flush=True)
            if not np.isfinite(p).all():raise ValueError('Incomplete excluded-question forecasts')
            oof[family]=p;old=fixed.losses(data['labels'],control[family]);new=fixed.losses(data['labels'],p)
            comparison={}
            for metric in old:
                delta=np.asarray([(new[metric][g]-old[metric][g]).mean() for g in groups]);boot=delta[draws].mean(1)
                comparison[metric]={'categorical':float(old[metric].mean()),'factorized':float(new[metric].mean()),
                    'delta':float(delta.mean()),'paired_question_CI95':np.quantile(boot,[.025,.975]).tolist()}
            reports[family]={'comparisons':comparison,'screen_passed':bool(comparison['mean_four_marginal_Brier']['delta']<0
                and comparison['joint_log_loss']['factorized']<=1.1*comparison['joint_log_loss']['categorical'])}
        io.npz(args.output/'private-oof.npz',oof);io.write(args.output/'fold-receipts.json',receipts)
        io.write(args.output/'summary.json',{'schema':SCHEMA,'reports':reports,'training_questions':768,
            'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,'controller_accuracy_claim':False,
            'conditional_independence_approximation':True,'paired_CI_scope':'Descriptive question bootstrap with fitted folds fixed'})
        io.verify(record['source_files']);io.verify(record['input_evidence'])
        io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
            'digests':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
            'any_screen_passed':any(v['screen_passed'] for v in reports.values()),'controller_accuracy_claim':False})
        print({'reports':reports,'controller_accuracy_claim':False},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','diagnose'))
    for n in ('features','categorical','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--registration-sha256');args=p.parse_args()
    with fixed.training_only_reads(),fixed.guarded.no_model_calls():
        data,control,record=inputs(args)
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'fitting_started':False});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Changed factorized TRAIN registration')
        diagnose(args,data,control,record)


if __name__=='__main__':main()
