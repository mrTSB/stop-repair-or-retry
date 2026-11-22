"""Fit expanded math observations only after the nested TRAIN screen passes."""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import expanded_math_diagnostic as diagnostic
import fit_math_specialization as original

io, fixed, core = diagnostic.io, diagnostic.fixed, original.core
SCHEMA = 'expanded-math768-eight-state-fit-v1'


def inputs(args):
    spec = io.read(args.diagnostic/'registration.json');done = io.read(args.diagnostic/'complete.json')
    source = SimpleNamespace(**{k:Path(v) for k,v in spec['paths'].items()})
    bundle = diagnostic.inputs(source)
    if (bundle[-1] != spec or done['schema'] != diagnostic.SCHEMA or not done['screen_passed']
        or done['registration_sha256'] != io.sha(args.diagnostic/'registration.json')):
        raise ValueError('Require successful sealed expanded-math nested screen')
    io.verify(spec['source_files']);io.verify(spec['input_evidence'])
    io.verify({str(args.diagnostic/k):v for k,v in done['digests'].items()})
    feature_spec = io.read(source.features/'registration.json')
    old_spec = io.read(Path(feature_spec['paths']['math_diagnostic'])/'registration.json')
    base_spec = io.read(Path(old_spec['paths']['base_fit'])/'registration.json')
    path = Path(base_spec['paths']['base'])/'C0.1-smooth32.pkl';base = pickle.loads(path.read_bytes())
    if type(base) is not core.Planner or base.fit_receipt['paired_questions'] != 512: raise ValueError('Changed structural POMDP ancestry')
    with np.load(args.diagnostic/'private-oof.npz',allow_pickle=False) as values:oof=values['probability'].copy()
    data = bundle[2]
    if oof.shape != (2304,8) or not np.isfinite(oof).all():raise ValueError('Require all768 excluded-question math forecasts')
    record = {'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),
        'input_evidence':{**spec['input_evidence'],str(path.resolve()):io.sha(path),
            **{str((args.diagnostic/k).resolve()):v for k,v in done['digests'].items()},
            str((args.diagnostic/'complete.json').resolve()):io.sha(args.diagnostic/'complete.json')},
        'paths':{k:str(getattr(args,k).resolve()) for k in ('diagnostic','output')},
        'math_questions':768,'decision_nodes':2304,'state_count':8,
        'forecaster_selection':'Same inner whole-question search on all768 math TRAIN',
        'emissions':'Math-only counts of nested outer-held forecasts, .1 support smoothing',
        'structural_prior_transition_cost':'Exact retained pooled-model math arrays, unchanged',
        'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False}
    return data,base,oof,record


def fit(data,base,oof,registration):
    model = original.fit(data,base,np.ones(len(data['records']),dtype=bool),oof,registration)
    model.fit_receipt.update(schema=SCHEMA,observation_forecast_math_questions=768,
        original_math_questions=256,additional_paired_math_questions=512)
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','fit'))
    for key in ('diagnostic','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--registration-sha256');args=p.parse_args()
    with fixed.training_only_reads(),fixed.guarded.no_model_calls(),threadpool_limits(limits=1):
        data,base,oof,record=inputs(args)
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json')},flush=True);return
        if not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256 or io.read(args.output/'registration.json')!=record:
            raise ValueError('Changed expanded math fit registration')
        with io.lock(args.output/'.run.lock'):
            if any(v.name not in ('registration.json','.run.lock') for v in args.output.iterdir()):raise ValueError('Preserve prior fit outputs')
            model=fit(data,base,oof,args.registration_sha256);path=args.output/'math.pkl';io.atomic_bytes(path,pickle.dumps(model,protocol=5))
            io.verify(record['source_files']);io.verify(record['input_evidence'])
            io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
                'model_sha256':io.sha(path),'fit_receipt':model.fit_receipt,'new_neural_calls':0,'cloud_calls':0,'development_or_test_read':False,
                'digests':{v.name:io.sha(v) for v in args.output.iterdir() if v.is_file() and not v.name.startswith('.')}})
            print({'complete':True,'setting':model.fit_receipt['forecaster_settings']},flush=True)


if __name__=='__main__':main()
