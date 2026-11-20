"""Development-only search of the four-state POMDP using complete paired branches."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import pickle
import time

import numpy as np

from margin_data import HERE, read,wait_complete
from difficulty_pomdp import evaluate, fit, observation, probabilities, visible
from extended_study import records, statistics
from context_pomdp import evaluate_llm
from study import fixed


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def warm(model,rows,trees):
    examples=[(r,p) for r in rows for p in trees[r['id']]]
    x=np.array([visible(r,p,trees[r['id']]) for r,p in examples])
    scores=probabilities(model.classifier,x)
    for (r,p),v in zip(examples,scores):
        node=trees[r['id']][p]
        model._observed[r['id'],p,node['prompt_sha256'],node['text']]=observation(v,bool(node['answer']))


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    train,trees,judges=wait_complete(args.data,'train',args.wait)
    train_ids={r['id'] for r in train}
    dev,dt,dj=wait_complete(args.data,'dev',args.wait)
    llm=evaluate_llm
    if args.late_data:
        from late_pairs import merged_records,llm_policy
        deadline=time.monotonic()+args.wait
        for split,n in (('train',len(train)),('dev',len(dev))):
            while True:
                try:
                    path=args.data/f'{split}-late-observations.jsonl'
                    if path.exists() and len(read(path))==6*n:break
                except json.JSONDecodeError:pass
                if time.monotonic()>deadline:raise TimeoutError('Late paired data incomplete')
                time.sleep(5)
        train,trees,judges=merged_records(args.data,'train')
        dev,dt,dj=merged_records(args.data,'dev')
        llm=llm_policy
    assert not train_ids & {r['id'] for r in dev}
    fit_identity={'data_sha256':hashlib.sha256((args.data/'train-observations.jsonl').read_bytes()).hexdigest(),
                  'model_source_sha256':hashlib.sha256((HERE/'difficulty_pomdp.py').read_bytes()).hexdigest(),
                  'questions_sha256':hashlib.sha256((args.data/'splits/train-questions.jsonl').read_bytes()).hexdigest(),
                  'stage_review':True,'late_feedback':args.late_data,
                  'numpy':'2.5.4','scikit_learn':'1.8.0'}
    if args.late_data:
        fit_identity['late_data_sha256']=hashlib.sha256((args.data/'train-late-observations.jsonl').read_bytes()).hexdigest()
    fixed_rows={name:[fixed(r,dt[r['id']],action,depth) for r in dev] for name,action,depth in
                [('Stop','stop',0),('Repair once','repair',1),('Repair twice','repair',2),
                 ('Retry once','retry',1),('Retry twice','retry',2)]}
    baseline={k:statistics(v) for k,v in fixed_rows.items()}
    candidates=[]
    for threshold,reuse in product(('action','agreement',.5,.8,.95,.98,.995,1.),(False,True)):
        settings={'judge_stop':threshold,'reuse_review':reuse}
        items=[llm(r,dt[r['id']],dj,settings) for r in dev]
        candidates.append({'family':'B','model':None,'settings':settings,'dev':statistics(items)})
    b=max(candidates,key=lambda r:(r['dev']['accuracy'],-r['dev']['mean_compute']))
    save(args.output/'baseline-and-b.json',{'baselines':baseline,'B':b})
    for leaf,smoothing in product(args.leaves,args.smoothing):
        key=f'leaf{leaf}-smooth{smoothing:g}-late{int(args.late_data)}'
        target=args.output/(key+'.pkl')
        receipt=args.output/(key+'-fit.json')
        if target.exists():
            if json.loads(receipt.read_bytes())!=fit_identity:
                raise ValueError('Training evidence changed; use a new probe output directory')
            model=pickle.loads(target.read_bytes())
        else:
            print('Fitting',key,flush=True)
            model=fit(train,trees,judges,leaf,smoothing,late_feedback=args.late_data,stage_review=True)
            target.write_bytes(pickle.dumps(model,protocol=5))
            save(receipt,fit_identity)
        warm(model,dev,dt)
        for penalty,weight in product((0.,.025,.06,.12,.22),(.7,1.,1.4)):
            settings={'penalty':penalty,'science_weight':weight,'reuse_review':True,'late_feedback':args.late_data}
            items=[evaluate(r,dt[r['id']],dj,model,settings,'A') for r in dev]
            candidates.append({'family':'A','model':key,'settings':settings,'dev':statistics(items)})
            for threshold,reuse,nonmyopic in product((-.03,0.,.03,.08,.15),(False,True),(False,True)):
                current={**settings,'threshold':threshold,'reuse_review':reuse,'plan_reviews':nonmyopic}
                items=[evaluate(r,dt[r['id']],dj,model,current,'C') for r in dev]
                candidates.append({'family':'C','model':key,'settings':current,'dev':statistics(items)})
            for d in model.dynamics.values():
                d._values.clear();d._information.clear();d._alphas.clear()
        save(args.output/'development-search.json',candidates)
        members=[r for r in candidates if r['family']=='C' and r['dev']['mean_judge_calls']>0
                 and r['dev']['mean_compute']<b['dev']['mean_compute']]
        best=max(members,key=lambda r:(r['dev']['accuracy'],-r['dev']['mean_compute'])) if members else None
        print('Completed',key,'best cheaper C',json.dumps(best),flush=True)
    chosen={family:max((r for r in candidates if r['family']==family and
                       (family!='C' or r['dev']['mean_judge_calls']>0)),
                       key=lambda r:(r['dev']['accuracy'],-r['dev']['mean_compute'])) for family in ('A','B','C')}
    cheaper=[r for r in candidates if r['family']=='C' and r['dev']['mean_judge_calls']>0
             and r['dev']['mean_compute']<chosen['B']['dev']['mean_compute']]
    chosen['C cheaper than B']=max(cheaper,key=lambda r:(r['dev']['accuracy'],-r['dev']['mean_compute'])) if cheaper else None
    best_baseline=max(v['accuracy'] for v in baseline.values())
    result={'selected':chosen,'baselines':baseline,
            'targets_percentage_points':{'A_B_minimum':best_baseline+.1,
                'C_minimum_if_A_B_qualify':max(best_baseline+.1,chosen['A']['dev']['accuracy'],chosen['B']['dev']['accuracy'])+.1},
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
            'fresh_holdout_used':False,'note':'Development-only architecture probe; not a final freeze or fresh result.'}
    save(args.output/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--leaves',nargs='+',type=int,default=[8,20,40])
    parser.add_argument('--smoothing',nargs='+',type=float,default=[16.,64.])
    parser.add_argument('--wait',type=int,default=21600)
    parser.add_argument('--late-data',action='store_true')
    run(parser.parse_args())
