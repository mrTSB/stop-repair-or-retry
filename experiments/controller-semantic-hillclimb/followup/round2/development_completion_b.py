"""Register/replay the fixed24 Complete-enabled B rules on sealed DEV only.

No network/NN acquisition path exists. The exact original winning B is first
reproduced on all256 questions; its128 math outcomes are retained unchanged.
All old B winners and five baselines remain in the final comparison.
"""
import argparse
import copy
from pathlib import Path
import sys

import completion_matched_b as policy
import completion_development_inputs as joined
import expanded_mentor_io as io

SCHEMA='complete-enabled-LLM-B-development-v1'


def inputs(args):
    rows,trees,judges,receipt=joined.load(args.data,args.retained,args.complete)
    if (str(args.b_reference.resolve())!=receipt['retained']['B_reference']
            or io.sha(args.b_reference)!=receipt['retained']['B_summary_sha256']):
        raise ValueError('Retain the exact strongest original B bound to the full review grid')
    reference=io.read(args.b_reference)
    if (reference['development_questions']!=256 or reference['test_used'] is not False
            or reference['B'].get('protocol')!='matched-retained-1024-512-128'):
        raise ValueError('This registered comparison retains the completed strongest matched B math route')
    evidence={str(args.b_reference.resolve()):io.sha(args.b_reference),
        str((args.b_reference.parent/'identity.json').resolve()):io.sha(args.b_reference.parent/'identity.json')}
    return rows,trees,judges,reference,receipt,evidence


def contract(args,bundle):
    rows,_,_,reference,receipt,evidence=bundle
    return {'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),
        'paths':{k:str(getattr(args,k).resolve()) for k in ('data','retained','complete','b_reference','output')},
        'input_receipt':receipt,'reference_evidence':evidence,'settings':policy.SETTINGS,
        'retained_math_B':reference['B'],'candidate_count':24,'development_questions':len(rows),
        'new_cloud_calls':0,'new_neural_calls':0,'test_used':False,
        'selection':'Highest full-panel accuracy across every old B winner and24 new rules; lower compute breaks ties',
        'reference_before_all_seals':False,'original_five_baselines_unchanged':True,
        'same_original_B_math_route_for_every_new_rule':True}


def seal_payload(directory):
    names=['registration.json','identity.json','search.json','summary.json','private-original-B.jsonl',
           *[f'private-candidate-{i:02d}.jsonl' for i in range(24)]]
    return {'schema':SCHEMA,'files':{n:io.sha(directory/n) for n in names}}


def persist(path,value):
    if path.exists():
        if io.read(path)!=value: raise ValueError('Preserve an inconsistent earlier replay artifact')
    else:io.write(path,value)


def persist_jsonl(path,rows):
    expected=''.join(io.json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows).encode()
    if path.exists():
        if path.read_bytes()!=expected: raise ValueError('Preserve an inconsistent or partial earlier JSONL replay artifact')
    else:io.atomic_bytes(path,expected)


def run(args):
    bundle=inputs(args);record=contract(args,bundle)
    if args.mode=='register':
        if args.output.exists(): raise ValueError('Use a new Complete-enabled B output directory')
        args.output.mkdir(parents=True);io.write(args.output/'registration.json',record)
        print({'registered':True,'registration_sha256':io.sha(args.output/'registration.json')},flush=True);return
    if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
            or io.read(args.output/'registration.json')!=record):
        raise ValueError('Require the unchanged explicit registered rule grid and full inputs')
    with io.lock(args.output/'.replay.lock'):
        marker=args.output/'complete.json'
        if marker.exists():
            if io.read(marker)!=seal_payload(args.output): raise ValueError('Completed B replay changed')
            return
        rows,trees,judges,reference,receipt,evidence=bundle
        original=[policy.original.policy(r,trees[r['id']],judges,reference['B']['settings']) for r in rows]
        if policy.original.metrics(original)!=reference['B']['dev']:
            raise ValueError('Original winning B failed exact full-panel reproduction')
        math={r['id']:r for r in original if r['subject']=='math'}
        baselines={name:policy.original.statistics([policy.original.fixed(r,trees[r['id']],*settings) for r in rows])
                   for name,settings in policy.original.BASELINES.items()}
        if baselines!=reference['baselines']: raise ValueError('An original baseline changed')
        persist_jsonl(args.output/'private-original-B.jsonl',original)
        candidates=[]
        for i,settings in enumerate(policy.SETTINGS):
            items=policy.evaluate_panel(rows,trees,judges,math,settings)
            dev=policy.original.metrics(items)
            dev['mean_completion_calls']=sum(r.get('completion_calls',0) for r in items)/256
            candidate={'family':'B','protocol':SCHEMA,'policy_module':'completion_matched_b',
                'settings':settings,'retained_math_B':reference['B'],'dev':dev}
            candidates.append(candidate);persist_jsonl(args.output/f'private-candidate-{i:02d}.jsonl',items)
        old=[]
        for name in ('ordinary_B','matched_B','B'):
            if name in reference and reference[name] not in [v['candidate'] for v in old]:
                old.append({'name':name,'candidate':copy.deepcopy(reference[name])})
        retained=[{**r['candidate'],'original_name':r['name'],'source_summary':str(args.b_reference.resolve()),
                   'source_summary_sha256':io.sha(args.b_reference)} for r in old]
        result=policy.compare(candidates,retained)
        result.update(schema=SCHEMA,baselines=baselines,development_questions=256,
            original_B_exactly_reproduced=True,new_cloud_calls=0,new_neural_calls=0,
            registration_sha256=args.registration_sha256,source_summary_sha256=io.sha(args.b_reference))
        result['B_above_highest_original_baseline']=result['B']['dev']['accuracy']-max(r['accuracy'] for r in baselines.values())
        identity={'schema':SCHEMA,'source_sha256':io.sha(__file__),'policy_sha256':io.sha(policy.__file__),
            'dev_questions_sha256':receipt['dev_questions_sha256'],'registration_sha256':args.registration_sha256,
            'test_used':False,'all_original_B_retained':True,'new_action_protocol':True}
        io.verify(record['source_files']);io.verify(evidence)
        if inputs(args)[4]!=receipt: raise ValueError('Sealed action input changed during replay')
        persist(args.output/'identity.json',identity);persist(args.output/'search.json',candidates)
        persist(args.output/'summary.json',result);io.write(marker,seal_payload(args.output))
        print({'complete':True,'B':result['B'],'new_best_B':result['new_best_B'],'fresh_validation':False},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','run'))
    for name in ('data','retained','complete','b_reference','output'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--registration-sha256')
    run(parser.parse_args())
