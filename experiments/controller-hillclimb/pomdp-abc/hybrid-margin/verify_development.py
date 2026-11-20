"""Refit selected models and replay development decisions before opening the holdout."""

import argparse
import json
from pathlib import Path

import numpy as np

from margin_data import read
from extended_study import records
from late_pairs import merged_records
from study_common import evidence,models,run_policy,sha,statistics


def clear(model):
    if model is None:return
    for name in ('_observed','_current_context','_review_context'):
        getattr(model,name,{}).clear()
    getattr(model.classifier,'_probability_cache',{}).clear()
    getattr(getattr(model,'review_observer',None),'_cache',{}).clear()


def without_grade(record):
    return {k:v for k,v in record.items() if k!='correct'}


def verify(args):
    frozen=json.loads((args.output/'frozen.json').read_bytes())
    if evidence(args.data)!=frozen['evidence']:raise ValueError('Changed freeze evidence')
    planners=models(frozen,args.output);refits={};counts={'refitted_outcomes':0,'blind_paid_replays':0}
    measured={}
    for family in ('A','B','C'):
        configuration=frozen['selected'][family];settings=configuration['settings']
        loader=merged_records if configuration['protocol']=='late' else records
        rows,trees,judges=loader(args.data,'dev')
        model=planners.get(family)
        if family!='B':
            train,tt,tj=loader(args.data,'train');receipt=model.fit_receipt
            parameters={'leaf':receipt['leaf'],'smoothing':receipt['smoothing'],
                        'late_feedback':settings.get('late_feedback',False),'stage_review':True}
            if settings.get('review_calibration',False):
                from review_calibration import fit
                refit=fit(train,tt,tj,**parameters,review_leaf=receipt['review_leaf'],
                          transition_leaf=receipt['transition_leaf'],context_weight=receipt['context_weight'])
            elif settings.get('neural_observer',False):
                from neural_pomdp import fit
                predictions={(r['id'],r['kind'],r['path']):np.array(r['probabilities'])
                             for r in read(args.data/'train-neural-observations.jsonl')}
                refit=fit(train,tt,tj,model.classifier.directory,predictions,**parameters,
                          contextual=settings.get('contextual_transitions',False),
                          transition_leaf=receipt.get('transition_leaf',24),
                          context_weight=settings.get('context_weight',.7))
            elif settings.get('contextual_transitions',False):
                from contextual_pomdp import fit
                refit=fit(train,tt,tj,**parameters,transition_leaf=receipt['transition_leaf'],
                          context_weight=receipt['context_weight'])
            else:
                from difficulty_pomdp import fit
                refit=fit(train,tt,tj,**parameters)
            refit.classifier.set_params(n_jobs=1);refits[family]=refit
        outcomes=[]
        for row in rows:
            tree=trees[row['id']]
            expected=run_policy(family,row,tree,judges,judges,model,configuration)
            outcomes.append(expected)
            if family in refits:
                reproduced=run_policy(family,row,tree,judges,judges,refits[family],configuration)
                assert reproduced==expected,(family,row['id'],'refit mismatch')
                counts['refitted_outcomes']+=1
            parts=expected['path'].split('/') if expected['path'] else []
            visited=['']+['/'.join(parts[:i]) for i in range(1,len(parts)+1)]
            acquired=[d['path'] for d in expected['decisions'] if d['queried']]
            paid_tree={p:tree[p] for p in visited}
            paid_judges={(row['id'],p):judges[row['id'],p] for p in acquired}
            clear(model)
            actual=run_policy(family,{**row,'answer':''},paid_tree,paid_judges,paid_judges,model,configuration)
            assert without_grade(actual)==without_grade(expected),(family,row['id'],'blinded replay mismatch')
            counts['blind_paid_replays']+=1
        metrics=statistics(outcomes)
        for k,v in configuration['dev'].items():
            assert np.isclose(v,metrics[k],rtol=0,atol=1e-9),(family,k,v,metrics[k])
        measured[family]=metrics
        print(f'{family}: complete development replay and refit checks passed',flush=True)
    result={'passed':True,'fresh_holdout_used':False,'freeze_sha256':sha(args.output/'frozen.json'),
            **counts,'methods':measured}
    (args.output/'development-reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    verify(parser.parse_args())
