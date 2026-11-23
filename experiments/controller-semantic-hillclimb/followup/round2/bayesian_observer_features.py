"""Complete missing frozen observer inputs locally before the full C search.

All public science root/first-continuation prefixes are included independently
of correctness. Existing entries remain untouched. Exact compatible aliases
are reused; every remaining distinct input receives one frozen CPU forward.
This is offline feature extraction, not training, and no test record is opened.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import torch

import development_bayesian_hybrid_v2 as adapter
from prospective_freeze import observer_slots

original,io = adapter.original,adapter.original.io
SCHEMA = 'frozen-science-observer-whole-DEV-prefix-coverage-v1'


def inputs(args):
    inherited = io.read(args.previous/'registration.json')
    inherited_args = SimpleNamespace(previous=args.previous,output=args.output)
    bundle,_,source,_ = original.inputs(inherited_args)
    bundle,aliases = adapter.augment(bundle,source)
    rows,trees,_,_,caches,*_ = bundle
    found = {}
    for name in original.PARENTS:
        model = pickle.loads((source/'models'/(name+'.pkl')).read_bytes())
        for _,observer in observer_slots(model):
            asset = original.canonical.previous.previous.share_key(observer)
            if asset[1:] == ('science_option_observer','ScienceOptionObserver'): found[asset] = observer
    plan = []
    for asset,observer in sorted(found.items()):
        observer._recipe = io.read(Path(observer.directory)/'recipe.json')
        observer._cache = dict(caches[asset])
        for row in rows:
            if row['subject'] not in ('chemistry','physics'):continue
            safe = {**row,'answer':''}
            for path in ('','repair','retry','guided'):
                if path not in trees[row['id']]:continue
                paid = original.policy.core.prefix(trees[row['id']],path)
                entry = observer.entry(safe,path,paid)
                fingerprint = hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest()
                key = ('solver',row['id'],path,fingerprint)
                if key not in observer._cache:
                    plan.append({'asset':list(asset),'key':list(key),'entry_sha256':fingerprint,
                                 'row':safe,'path':path,'paid':paid})
    public_plan = [{k:v for k,v in p.items() if k not in ('row','paid')} for p in plan]
    record = {'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),
        'prior_registration_sha256':io.sha(args.previous/'registration.json'),
        'prior_complete_sha256':io.sha(args.previous/'complete.json'),'exact_aliases':aliases,
        'plan_sha256':hashlib.sha256(json.dumps(public_plan,sort_keys=True).encode()).hexdigest(),
        'missing_keys':len(plan),'distinct_forwards':len({(tuple(v['asset']),v['key'][1],v['entry_sha256']) for v in plan}),
        'parameters':434306836,'device':'cpu','threads':2,'batch':1,
        'selection':'Every available science root/repair/retry/guided public input, without label filtering',
        'model_fitting':False,'references_supplied_to_encoder':False,'test_used':False,
        'development_used':True,'cloud_calls':0,'cloud_training':False,
        'all_old_entries_unchanged':True,'all_used_query_slots_charged_at_policy_inference':True,
        'previous':str(args.previous.resolve()),'output':str(args.output.resolve())}
    return bundle,found,plan,record


def build(args,bundle,observers,plan,record):
    caches = dict(bundle[4]);computed,receipts = {},[]
    for asset,observer in observers.items():
        observer.device = 'cpu'
        observer.load()
        torch.set_num_threads(2)
        parameters = sum(p.numel() for p in observer._model.parameters())
        if parameters != record['parameters']: raise ValueError('Changed frozen observer architecture')
        caches[asset] = observer._cache
    for item in plan:
        asset,key = tuple(item['asset']),tuple(item['key'])
        identity = asset,key[1],key[3]
        observer = observers[asset]
        if identity not in computed:
            before = set(observer._cache)
            value = observer.values(item['row'],item['path'],item['paid'])
            if set(observer._cache)-before != {key}: raise ValueError('Unexpected observer forward/cache coverage')
            computed[identity] = value
            receipts.append({'asset':list(asset),'key':list(key),'input_sha256':key[3],
                'actual_forward_token_slots':int(value[3]),'parameters':record['parameters'],
                'compute_if_used_at_inference':record['parameters']*int(value[3])/3.09e9})
            if len(receipts)%16 == 0:print({'local_forwards':len(receipts),'total':record['distinct_forwards']},flush=True)
        else:
            observer._cache[key] = computed[identity]
    if len(receipts) != record['distinct_forwards']: raise ValueError('Incomplete local frozen forward plan')
    # Source bundle is rebuilt independently to verify every inherited key.
    old_bundle,*_ = inputs(args)
    for asset,entries in old_bundle[4].items():
        if asset not in caches:caches[asset] = entries
        for key,value in entries.items():
            if any(not np.array_equal(a,b) for a,b in zip(caches[asset][key],value)):
                raise ValueError('An inherited observation changed')
    if inputs(args)[-1] != record: raise ValueError('Changed registered sources or prefix selection')
    io.atomic_bytes(args.output/'private-neural-caches.pkl',pickle.dumps(caches,protocol=5))
    io.write(args.output/'forward-receipts.json',receipts)
    summary = {'schema':SCHEMA,'new_local_neural_forwards':len(receipts),'new_keys':len(plan),
        'actual_forward_token_slots':sum(v['actual_forward_token_slots'] for v in receipts),
        'parameters':record['parameters'],'cloud_calls':0,'model_fitting':False,
        'all_old_entries_verified_unchanged':True,'test_used':False,'references_supplied_to_encoder':False}
    io.write(args.output/'summary.json',summary)
    names = ('registration.json','private-neural-caches.pkl','forward-receipts.json','summary.json')
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'digests':{name:io.sha(args.output/name) for name in names},'test_used':False,'cloud_calls':0})
    print(summary,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','build'))
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    with original.prior.guard.no_test_reads():
        bundle,observers,plan,record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True,exist_ok=False)
            io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),
                   'missing_keys':record['missing_keys'],'distinct_forwards':record['distinct_forwards']},flush=True)
        else:
            if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                    or io.read(args.output/'registration.json') != record):raise ValueError('Changed local feature-bank registration')
            if {p.name for p in args.output.iterdir()} != {'registration.json'}:raise ValueError('Preserve previous feature extraction')
            build(args,bundle,observers,plan,record)


if __name__ == '__main__':main()
