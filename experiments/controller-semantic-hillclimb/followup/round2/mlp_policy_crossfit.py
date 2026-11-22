"""TRAIN whole-policy screen of a small MLP head, preserving the forest control.

The frozen434M encoder is never called or trained. This fits only a64-unit
CPU head on existing observations. Same held questions, finite POMDP, clustered
observations, horizon, costs and accuracy/compute screen as the prior study.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import policy_crossfit_diagnostic as source
import mlp_completion_forecast as head

io,fixed,nested=source.io,source.fixed,source.nested
SCHEMA='small-MLP-whole-POMDP-crossfit-TRAIN-v1'
VARIANTS=(('expanded1280_MLP_raw',nested.FAMILIES[0]),('expanded1280_MLP_dense',nested.FAMILIES[1]))


@contextmanager
def head_fitter():
    original=source.forecast.fit
    source.forecast.fit=head.fit
    try:yield
    finally:source.forecast.fit=original


def inputs(args):
    rows,trees,observer,data,record=source.inputs(args)
    control_spec=io.read(args.control/'registration.json');done=io.read(args.control/'complete.json')
    if (control_spec['schema']!=source.SCHEMA or done['schema']!=source.SCHEMA
            or done['registration_sha256']!=io.sha(args.control/'registration.json')
            or control_spec['paths']['expanded_fit']!=str(args.expanded_fit.resolve())):raise ValueError('Require completed identical forest comparison')
    for key in ('labels_sha256','feature_sha256','records_sha256','forest_setting','policy_setting','kernel','smoothing'):
        if control_spec[key]!=record[key]:raise ValueError('Changed common data or fixed policy')
    io.verify(control_spec['source_files']);io.verify(control_spec['input_evidence'])
    io.verify({str(args.control/n):h for n,h in done['digests'].items()})
    control=io.read(args.control/'private-held-items.json')['retained768_dense']
    if set(control)!={r['id'] for r in rows[:768]}:raise ValueError('Require all768 identical held control decisions')
    record.update(schema=SCHEMA,source_files=io.sources(sys.modules[__name__]),
        variants=[list(v) for v in VARIANTS],head_settings={**head.SETTINGS,'hidden_layer_sizes':list(head.SETTINGS['hidden_layer_sizes'])},
        head_prior_weight=head.PRIOR_WEIGHT,encoder_training=False,new_encoder_calls=0,
        small_head_fitting=True,CPU_head_latency_not_in_parameter_token_proxy=True)
    record.pop('new_neural_calls');record['paths']['control']=str(args.control.resolve())
    record['control_forest_trees']=record.pop('trees')
    record['control_forest_setting']=record.pop('forest_setting')
    record['scope']='Exploratory TRAIN comparison of a new small head against the completed fixed forest control; no fresh claim'
    record['input_evidence'].update({str((args.control/n).resolve()):h for n,h in done['digests'].items()})
    record['input_evidence'][str((args.control/'complete.json').resolve())]=io.sha(args.control/'complete.json')
    return rows,trees,observer,data,control,record


def diagnose(args,rows,trees,observer,data,control,record):
    ids=[r['id'] for r in rows[:768]];results={};receipts={};reports={}
    draws=np.random.default_rng(238031).integers(0,768,(2000,768))
    for name,family in VARIANTS:
        items={};receipts[name]=[]
        for fold in range(3):
            with head_fitter():held,receipt=source.fit_fold(rows,trees,observer,data,1280,family,fold)
            if set(held)&set(items):raise ValueError('Duplicate held decisions')
            items.update(held);receipts[name].append(receipt)
            print({'variant':name,'outer_fold_complete':fold,'held_correct':sum(v['correct'] for v in held.values()),'held_questions':len(held)},flush=True)
        if set(items)!=set(ids):raise ValueError('Missing held questions')
        accuracy=np.array([int(items[k]['correct'])-int(control[k]['correct']) for k in ids])
        compute=np.array([items[k]['compute']-control[k]['compute'] for k in ids])
        results[name]=items
        reports[name]={'metrics':source.statistics(list(items.values())),
            'accuracy_delta':float(accuracy.mean()),'accuracy_delta_CI95':np.quantile(accuracy[draws].mean(1),[.025,.975]).tolist(),
            'compute_delta':float(compute.mean()),'compute_delta_CI95':np.quantile(compute[draws].mean(1),[.025,.975]).tolist(),
            'screen_passed':bool(accuracy.mean()>=.02 and compute.mean()<=0)}
    io.write(args.output/'private-held-items.json',results);io.write(args.output/'fold-receipts.json',receipts)
    io.write(args.output/'summary.json',{'schema':SCHEMA,'control':source.statistics(list(control.values())),
        'reports':reports,'common_questions':768,'new_encoder_calls':0,'cloud_calls':0,
        'encoder_training':False,'small_head_fitting':True,'development_or_test_read':False,'fresh_validation':False,
        'uncertainty_scope':'Descriptive paired question intervals with fitted folds fixed',
        'compute_scope':'Same paid token proxy; additional tiny CPU head latency not represented'})
    io.verify(record['source_files']);io.verify(record['input_evidence'])
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'any_screen_passed':any(v['screen_passed'] for v in reports.values()),
        'digests':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'reports':reports},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','diagnose'))
    for name in ('expanded-fit','control','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');args=parser.parse_args()
    with fixed.training_only_reads(),fixed.guarded.no_model_calls(),threadpool_limits(limits=1):
        bundle=inputs(args);record=bundle[-1]
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json')});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Changed registered MLP comparison')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json','.run.lock') for p in args.output.iterdir()):raise ValueError('Preserve prior outputs')
            diagnose(args,*bundle)


if __name__=='__main__':main()
