"""Frozen evidence, policy dispatch, and paired margin measurements."""

import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from margin_data import ROOT,HERE,read
from extended_study import records
from context_pomdp import evaluate_llm
from difficulty_pomdp import evaluate
from late_pairs import merged_records,llm_policy
from study import fixed
from extended_study import statistics as ordinary_statistics

BASELINES={'Stop':('stop',0),'Repair once':('repair',1),'Repair twice':('repair',2),
           'Retry once':('retry',1),'Retry twice':('retry',2)}


def locate(path):
    if path.exists():return path
    return path.with_suffix(path.suffix+'.gz')


def raw(path):
    path=locate(path)
    return gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()


def sha(path):return hashlib.sha256(raw(path)).hexdigest()


def evidence(data):
    result={'code/'+str(p.relative_to(ROOT)):sha(p) for p in sorted(
            (ROOT/'experiments/controller-hillclimb').rglob('*.py')) if '__pycache__' not in p.parts}
    for p in sorted((data/'splits').glob('*.json*')):
        result['splits/'+p.name]=sha(p)
    for p in sorted((data/'provenance').glob('*.json')):
        result['provenance/'+p.name]=sha(p)
    for split in ('train','dev'):
        for suffix in ('observations','late-observations','neural-observations'):
            p=data/f'{split}-{suffix}.jsonl'
            if locate(p).exists():result[f'data/{split}-{suffix}.jsonl']=sha(p)
    return result


def models(frozen,output):
    for descriptor in frozen.get('observer_assets',{}).values():
        for name,receipt in descriptor['files'].items():
            if sha(output/descriptor['directory']/name)!=receipt:
                raise ValueError('Frozen observation-model asset changed')
    result={}
    for family,receipt in frozen['models'].items():
        p=output/receipt['file']
        if sha(p)!=receipt['sha256']:raise ValueError('Frozen model changed')
        model=pickle.loads(p.read_bytes())
        if 'observer_directory' in receipt:
            model.classifier.directory=str(output/receipt['observer_directory'])
        model.classifier.set_params(n_jobs=1)
        result[family]=model
    return result


def run_policy(family,row,tree,base_judges,late_judges,model,configuration):
    protocol=configuration['protocol'];settings=configuration['settings']
    if family=='B':
        if model is not None:raise ValueError('B must not receive a POMDP')
        return (llm_policy if protocol=='late' else evaluate_llm)(
            row,tree,late_judges if protocol=='late' else base_judges,settings)
    planner=evaluate
    neural=configuration['settings'].get('neural_observer',False)
    if hasattr(model.classifier,'row_probabilities') and not neural:
        raise ValueError('Neural perception must be explicitly enabled and charged')
    if neural:
        from neural_pomdp import evaluate as neural_planner
        planner=neural_planner
    elif configuration['settings'].get('contextual_transitions',False):
        from contextual_pomdp import evaluate as contextual
        planner=contextual
    return planner(row,tree,late_judges if protocol=='late' else base_judges,model,settings,family)


def statistics(items):
    result=ordinary_statistics(items)
    for key in ('verifier_tokens','verifier_token_slots','verifier_calls'):
        result['mean_'+key]=float(np.mean([r.get(key,0.) for r in items]))
    return result


def load_test(data):
    rows,trees,judges=records(data,'test',False)
    p=data/'test-late-observations.jsonl'
    if locate(p).exists():
        _,late_trees,late_judges=merged_records(data,'test',False)
        trees=late_trees
    else:late_judges={k:v for k,v in judges.items() if k[1]==''}
    return rows,trees,judges,late_judges


def paired_margins(rows,results,repetitions=5000):
    names=list(results);n=len(rows)
    accuracy=np.array([[r['correct'] for r in results[k]] for k in names],dtype=float)
    cost=np.array([[r['compute'] for r in results[k]] for k in names],dtype=float)
    baseline=[names.index(k) for k in BASELINES]
    a,b,c=[names.index(k) for k in ('A','B','C')]
    def quantities(acc,work):
        best=acc[baseline].max(axis=0)
        return {'A_above_highest_baseline':acc[a]-best,
                'B_above_highest_baseline':acc[b]-best,
                'C_above_highest_A_B':acc[c]-np.maximum(acc[a],acc[b]),
                'C_above_B':acc[c]-acc[b],
                'C_compute_saved_vs_B':work[b]-work[c]}
    point=quantities(accuracy.mean(axis=1),cost.mean(axis=1))
    rng=np.random.default_rng(2026112009)
    strata=[[i for i,r in enumerate(rows) if r['subject']==s] for s in sorted({r['subject'] for r in rows})]
    draws=np.concatenate([rng.choice(g,(repetitions,len(g)),replace=True) for g in strata],axis=1)
    boot=quantities(accuracy[:,draws].mean(axis=2),cost[:,draws].mean(axis=2))
    return {k:{'point':float(v),'ci95':np.quantile(boot[k],[.025,.975]).tolist()} for k,v in point.items()}
