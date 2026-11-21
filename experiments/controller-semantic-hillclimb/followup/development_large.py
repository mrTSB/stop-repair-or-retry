"""Select semantic POMDP candidates on retired development data, filling needed paths only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from itertools import product
import json
from pathlib import Path
import pickle
import threading

import numpy as np

import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from semantic_data import read, records, write
from late_pairs import merged_records, judge_prompt, sanitized
import extended_acquire as old
from review_pilot import inference
from study_common import statistics, BASELINES, sha
from context_pomdp import evaluate_llm
from late_pairs import llm_policy
from study import fixed
from transport import resilient
from difficulty_pomdp import MissingObservation
import dynamic_grounded_pomdp as semantic


def run(args):
    train,tt,tj_original=records(args.data,'train',False)
    _,tt,tj_late=merged_records(args.data,'train',False)
    dev,dt,dj_original=records(args.data,'dev',False)
    _,dt,dj_late=merged_records(args.data,'dev',False)
    metadata=read(args.examples/'train.jsonl')
    outputs=np.load(args.verifier/'oof.npz')
    training_summary=json.loads((args.verifier/'training-summary.json').read_bytes())
    if not {r['id'] for r in metadata}<={r['id'] for r in train}:
        raise ValueError('OOF nodes contain non-training questions')
    contract=json.loads((args.verifier/'contract.json').read_bytes())
    if contract['data_sha256']!=sha(args.examples/'train.jsonl'):
        raise ValueError('Changed training examples')
    args.output.mkdir(parents=True,exist_ok=True)
    identity={'train':sha(args.data/'train-observations.jsonl'),
              'late_train':sha(args.data/'train-late-observations.jsonl'),
              'examples':sha(args.examples/'train.jsonl'), 'oof':sha(args.verifier/'oof.npz'),
              'semantic_code':sha(Path(semantic.__file__)), 'training_contract':sha(args.verifier/'contract.json')}
    if args.precomputed is not None:
        identity['precomputed_predictions']=sha(args.precomputed)
        identity['precomputed_inputs']=sha(args.precomputed.with_suffix('.jsonl'))
    receipt=args.output/'identity.json'
    if receipt.exists() and json.loads(receipt.read_bytes())!=identity:
        raise ValueError('Changed development study; choose a new output directory')
    receipt.write_text(json.dumps(identity,indent=2)+'\n')
    observer=semantic.Observer(str((args.verifier/'full').resolve()),
                               training_summary['temperature_selected_on_training_oof'],device=args.device)
    from transformers.utils import logging
    logging.set_verbosity_error()
    if args.precomputed is not None:
        descriptions=read(args.precomputed.with_suffix('.jsonl'))
        predictions=np.load(args.precomputed)
        for i,item in enumerate(descriptions):
            row=next(r for r in dev if r['id']==item['id'])
            tree=dt[row['id']];path=item['path']
            if item['kind']=='solver':node=tree[path];kind='solver'
            else:
                node=(dj_late if item['kind']=='late-review' else dj_original)[row['id'],path];kind='review'
            actual=observer.entry(row,path,tree,None if kind=='solver' else node)
            digest=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
            if digest!=item['input_sha256']:raise ValueError('Changed precomputed observation input')
            observer._cache[observer.key(row,path,node,kind)]=(
                predictions['state'][i],predictions['plain'][i],predictions['guided'][i],int(predictions['tokens'][i]))
        print(f'Loaded {len(descriptions)} reference-free GPU observations; policy passes remain charged',flush=True)
    else:
        observer.warm(dev,dt,dj_late)
        observer.warm(dev,dt,dj_original)
    observer.load()
    request=None
    if args.collect:
        import modal
        request=resilient(modal.Cls.from_name('srr-followup-solver-20261120','Ollama')(),
                          args.data/'development-transport-errors.jsonl')
        old.request=request
        tags={r['name']:r['digest'] for r in request('/api/tags')['models']}
        if tags.get(old.MODEL)!=old.MODEL_DIGEST or tags.get(old.JUDGE)!=old.JUDGE_DIGEST:
            raise ValueError('Changed solver/reviewer weights')
    lock=threading.Lock(); new_nodes=0
    with (args.data/'dev-observations.jsonl').open('a') as base_stream, (args.data/'dev-late-observations.jsonl').open('a') as late_stream:
        fcntl.flock(base_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(late_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def acquire(row,missing,late):
            nonlocal new_nodes
            if request is None:
                raise RuntimeError('Missing development observation; rerun with --collect') from missing
            tree=dt[row['id']]; kind,path=missing.kind,missing.path
            # Some LLM policies ask for the next review before dereferencing
            # its current solver response. Materialize that dependency first.
            if kind=='judge' and path not in tree:
                acquire(row,MissingObservation('solver',path),late)
            if kind=='solver' and path:
                parent=path.rsplit('/',1)[0] if '/' in path else ''
                if parent not in tree:
                    acquire(row,MissingObservation('solver',parent),late)
                if path=='guided' and (row['id'],'') not in dj_original:
                    acquire(row,MissingObservation('judge',''),False)
                if path.endswith('/reviewed') and (row['id'],parent) not in dj_late:
                    acquire(row,MissingObservation('judge',parent),True)
            # Acquisition prompts receive observable task text and paid responses only.
            safe={**row,'answer':''}
            storage='base'
            if kind=='judge' and path and late:
                node=sanitized(inference(request,safe,judge_prompt(safe,path,tree),'brief',True,path))
                node['protocol']='late-reviewed-repair-v1';storage='late'
                dj_late[row['id'],path]=node
            elif kind=='solver' and path.endswith('/reviewed'):
                parent=path.rsplit('/',1)[0]; review=dj_late[row['id'],parent]
                prompt=old.repair_prompt(safe,tree[parent],old.feedback(review))
                node=inference(request,safe,prompt,'brief',False,parent+'/repair')
                node['protocol']='late-reviewed-repair-v1';storage='late';tree[path]=node
            else:
                if kind=='solver' and path not in old.PATHS:
                    raise ValueError('Unexpected solver path')
                if kind=='judge' and path not in old.CHECKPOINTS:
                    raise ValueError('Unexpected review checkpoint')
                node=old.call(safe,path,tree,dj_original,kind=='judge')
                if kind=='solver':tree[path]=node
                else:
                    dj_original[row['id'],path]=node
                    if not path:dj_late[row['id'],path]=node
            with lock:
                if node.get('backend',{}).get('app')!='srr-followup-solver-20261120':
                    raise ValueError('Unexpected acquisition backend')
                stream=late_stream if storage=='late' else base_stream
                stream.write(json.dumps({'id':row['id'],'kind':kind,'path':path,'response':node})+'\n');stream.flush()
                new_nodes+=1
                if new_nodes%20==0:print(f'Acquired {new_nodes} missing development nodes',flush=True)
        def batch(function,late):
            def one(row):
                while True:
                    try:return function(row,dt[row['id']],dj_late if late else dj_original)
                    except MissingObservation as missing:acquire(row,missing,late)
                    except KeyError as error:
                        value=error.args[0]
                        if isinstance(value,tuple) and len(value)==2 and value[0]==row['id']:
                            acquire(row,MissingObservation('judge',value[1]),late)
                        elif isinstance(value,str) and value in old.PATHS:
                            acquire(row,MissingObservation('solver',value),late)
                        else:raise
            with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(one,dev))
        baselines={name:statistics([fixed(r,dt[r['id']],*parameters) for r in dev])
                   for name,parameters in BASELINES.items()}
        candidates=[]
        for late,stop,reuse in product((False,True),('action','agreement',.8,.95,.995,1.),(False,True)):
            settings={'judge_stop':stop,'reuse_review':reuse}
            policy=llm_policy if late else evaluate_llm
            items=batch(lambda r,t,j:policy(r,t,j,settings),late)
            candidates.append({'family':'B','model':None,'settings':settings,'late':late,'dev':statistics(items)})
        best_b=max(candidates,key=lambda v:(v['dev']['accuracy'],-v['dev']['mean_compute']))
        (args.output/'baselines-b.json').write_text(json.dumps({'baselines':baselines,'B':best_b},indent=2)+'\n')
        print('Strong independent B:',json.dumps(best_b),flush=True)
        for smoothing,semantic_review in product(args.smoothing,(False,True)):
            late=True;key=f'smooth{smoothing:g}-review{int(semantic_review)}-late1'
            p=args.output/(key+'.pkl')
            if p.exists():model=pickle.loads(p.read_bytes())
            else:
                print('Fitting',key,flush=True)
                model=semantic.fit(train,tt,tj_late,args.verifier/'full',metadata,outputs,
                                   smoothing=smoothing,late=late,semantic_review=semantic_review)
                p.write_bytes(pickle.dumps(model,protocol=5))
            # Share immutable network weights and raw prediction caches while
            # retaining each fitted model's own state calibration.
            for attribute in ('_model','_tokenizer','_recipe','_cache','device','_retriever','_backgrounds'):
                setattr(model.classifier,attribute,getattr(observer,attribute))
            model.review_observer=model.classifier if semantic_review else None
            for weight,penalty,continuous in product(args.weights,(0.,.04,.10),(False,True)):
                settings={'context_weight':weight,'penalty':penalty,'science_weight':1.,
                          'reuse_review':True,'late_feedback':True,'plan_reviews':False,
                          'continuous_filter':continuous}
                items=batch(lambda r,t,j:semantic.evaluate(r,t,j,model,settings,'A'),late)
                candidates.append({'family':'A','model':key,'settings':settings,'late':late,'dev':statistics(items)})
                for threshold in (0.,.03,.08):
                    current={**settings,'threshold':threshold}
                    items=batch(lambda r,t,j:semantic.evaluate(r,t,j,model,current,'C'),late)
                    candidates.append({'family':'C','model':key,'settings':current,'late':late,'dev':statistics(items)})
                for dynamics in model.dynamics.values():
                    dynamics._values.clear();dynamics._information.clear();dynamics._alphas.clear()
            (args.output/'search.json').write_text(json.dumps(candidates,indent=2)+'\n')
            current=[r for r in candidates if r['family']=='C' and r['dev']['mean_judge_calls']>0
                     and r['dev']['mean_compute']<best_b['dev']['mean_compute']]
            print('Completed',key,'best cheaper C',json.dumps(max(current,key=lambda v:(v['dev']['accuracy'],-v['dev']['mean_compute'])) if current else None),flush=True)
        chosen={family:max((r for r in candidates if r['family']==family and
                            (family!='C' or r['dev']['mean_judge_calls']>0)),
                           key=lambda v:(v['dev']['accuracy'],-v['dev']['mean_compute'])) for family in ('A','B','C')}
        cheap=[r for r in candidates if r['family']=='C' and r['dev']['mean_judge_calls']>0
               and r['dev']['mean_compute']<best_b['dev']['mean_compute']]
        chosen['C_cheaper_than_B']=max(cheap,key=lambda v:(v['dev']['accuracy'],-v['dev']['mean_compute'])) if cheap else None
        result={'selected':chosen,'baselines':baselines,'fresh_data_used':False,
                'train_questions':len(train),'dev_questions':len(dev),'new_development_nodes':new_nodes,
                'targets_percentage_points':{'A_B_minimum':max(r['accuracy'] for r in baselines.values())+.1,
                    'C_minimum':max(chosen[k]['dev']['accuracy'] for k in ('A','B'))+.1}}
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--examples',type=Path,required=True)
    parser.add_argument('--verifier',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--collect',action='store_true');parser.add_argument('--smoothing',type=float,nargs='+',default=[32.])
    parser.add_argument('--precomputed',type=Path)
    parser.add_argument('--device',choices=('cpu','mps'),default='cpu')
    parser.add_argument('--weights',type=float,nargs='+',default=[0.])
    run(parser.parse_args())
