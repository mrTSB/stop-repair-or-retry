"""TRAIN-only observation-kernel refit of the completed768-question POMDPs."""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np

import fit_auxiliary_completion as previous
import quantile_completion_pomdp as policy

io,fixed,nested=previous.io,previous.fixed,previous.nested
SCHEMA='expanded768-quantile-observation-Complete16-fit-v1'


def inputs(args):
    rows,trees,observer,data,_=previous.inputs(SimpleNamespace(features=args.features,output=args.output))
    spec=io.read(args.models/'registration.json');done=io.read(args.models/'complete.json')
    if (spec['schema']!=previous.SCHEMA or done['schema']!=previous.SCHEMA or done['training_questions']!=768
            or done['registration_sha256']!=io.sha(args.models/'registration.json')
            or spec['paths']['features']!=str(args.features.resolve()) or done['development_or_test_read']
            or done['new_neural_calls'] or done['cloud_calls']
            or spec['records_sha256']!=fixed.fingerprint(data['records']) or spec['labels_sha256']!=fixed.array_sha(data['labels'])
            or spec['valid_sha256']!=fixed.array_sha(data['valid'])
            or spec['feature_sha256']!={k:fixed.array_sha(v) for k,v in data['features'].items()}):
        raise ValueError('Require exact completed categorical768 TRAIN fit')
    io.verify(spec['source_files']);io.verify(spec['input_evidence']);io.verify({str(args.models/n):v for n,v in done['digests'].items()})
    models={}
    for family in nested.FAMILIES:
        path=args.models/(family+'.pkl')
        if io.sha(path)!=done['models'][family]['model_sha256']:raise ValueError('Base model changed')
        model=pickle.loads(path.read_bytes())
        if model.fit_receipt!=done['models'][family]['fit_receipt'] or model.feature_family!=family:raise ValueError('Base model receipt changed')
        models[family]=model
    with np.load(args.models/'private-oof.npz',allow_pickle=False) as a:forecasts={k:a[k].copy() for k in nested.FAMILIES}
    evidence={**spec['input_evidence'],**{str((args.models/n).resolve()):v for n,v in done['digests'].items()},
        str((args.models/'complete.json').resolve()):io.sha(args.models/'complete.json')}
    record={'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{k:str(getattr(args,k).resolve()) for k in ('features','models','output')},'training_questions':768,
        'families':list(nested.FAMILIES),'learned_parameters':'Three current-probability quartiles and three best-action gain medians',
        'current_partition':'Valid repair/retry child TRAIN forecasts only, pooled subjects',
        'gain_partition':'Median per best-action group; fewer than16 children falls back to that action over all children',
        'forecast_provenance':'Sealed nested outer-held question predictions from categorical fit',
        'labels_used_for_thresholds':False,'emissions':'Refit on TRAIN states with exactly the original support, pseudocount and smoothing',
        'unchanged':['forecaster','prior','transition','cost','16 states','30 observations','exact finite Bellman backup'],
        'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,'controller_accuracy_claim':False}
    return rows,trees,data,models,forecasts,record


def fit(args,rows,trees,data,models,forecasts,record):
    with io.lock(args.output/'.fit.lock'):
        if any(p.name not in ('registration.json','.fit.lock') for p in args.output.iterdir()):raise ValueError('Preserve prior fit outputs')
        results={}
        for family,base in models.items():
            model=policy.rebin(base,rows,trees,data['records'],forecasts[family])
            for subject in base.dynamics:
                for field in ('prior','transition','cost'):
                    np.testing.assert_array_equal(getattr(base.dynamics[subject],field),getattr(model.dynamics[subject],field))
            model.fit_receipt['kernel_registration_sha256']=args.registration_sha256
            path=args.output/(family+'.pkl');io.atomic_bytes(path,pickle.dumps(model,protocol=5))
            results[family]={'model_sha256':io.sha(path),'fit_receipt':model.fit_receipt}
            print({'fitted':family,'current_edges':model.kernel.current_edges.tolist(),'gain_edges':model.kernel.gain_edges.tolist()},flush=True)
        io.verify(record['source_files']);io.verify(record['input_evidence'])
        io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,'models':results,
            'training_questions':768,'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,
            'digests':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','fit'))
    for n in ('features','models','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--registration-sha256');args=p.parse_args()
    with fixed.training_only_reads(),fixed.guarded.no_model_calls():
        rows,trees,data,models,forecasts,record=inputs(args)
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'fitting_started':False});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Changed quantile TRAIN registration')
        fit(args,rows,trees,data,models,forecasts,record)


if __name__=='__main__':main()
