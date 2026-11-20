"""Independently reconcile paid paths and repeat decisions with references removed."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from margin_data import read,stem
from study_common import BASELINES,evidence,load_test,models,run_policy,sha
from scoring import correct


def audit(args):
    frozen=json.loads((args.output/'frozen.json').read_bytes())
    if frozen['evidence']!=evidence(args.data):raise ValueError('Changed frozen evidence')
    rows,trees,base,late=load_test(args.data)
    if len(rows)!=800:raise ValueError('Incomplete fresh panel')
    history=json.loads((args.data/'provenance/historical-exclusions.json').read_bytes())
    excluded_ids=set(history['ids']);excluded_stems=set(history['normalized_stem_sha256'])
    assert not excluded_ids & {r['id'] for r in rows}
    assert not excluded_stems & {hashlib.sha256(stem(r['question']).encode()).hexdigest() for r in rows}
    assert len({r['id'] for r in rows})==len({stem(r['question']) for r in rows})==800
    q={r['id']:r for r in rows}
    saved=read(args.output/'test/per_question.jsonl')
    by_key={(r['policy'],r['id']):r for r in saved}
    if len(by_key)!=len(saved) or len(saved)!=6400:raise ValueError('Wrong policy outcome coverage')
    planners=models(frozen,args.output)
    checked=0;blinded=0
    for family in (*BASELINES,'A','B','C'):
        for row in rows:
            record=by_key[family,row['id']];path=record['path']
            parts=path.split('/') if path else []
            visited=['']+['/'.join(parts[:i]) for i in range(1,len(parts)+1)]
            assert len(visited)<=3
            if family in BASELINES:
                action,depth=BASELINES[family]
                assert path==('/'.join([action]*depth) if depth else '')
                acquired=[]
            else:
                configuration=frozen['selected'][family]
                acquired=[d['path'] for d in record['decisions'] if d['queried']]
                assert len(acquired)==len(set(acquired))<=2
                assert all(p in visited for p in acquired)
                if family=='A':assert not acquired
                if parts and parts[0]=='guided':assert '' in acquired
                if parts and parts[-1]=='reviewed':
                    assert configuration['protocol']=='late'
                    assert '/'.join(parts[:-1]) in acquired
                for decision in record['decisions']:
                    assert decision['action'] in ('stop','repair','retry')
                    if 'belief' in decision:
                        b=np.asarray(decision['belief'])
                        assert b.shape==(4,) and np.isfinite(b).all() and (b>=0).all()
                        assert np.isclose(b.sum(),1.)
            active=late if family not in BASELINES and frozen['selected'][family]['protocol']=='late' else base
            solver=sum(trees[row['id']][p]['input_tokens']+trees[row['id']][p]['output_tokens'] for p in visited)
            review=sum(active[row['id'],p]['input_tokens']+active[row['id'],p]['output_tokens'] for p in acquired)
            assert (record['solver_tokens'],record['judge_tokens'],record['solver_calls'],record['judge_calls'])==(solver,review,len(visited),len(acquired))
            verifier=0.
            if family not in BASELINES and frozen['selected'][family]['settings'].get('neural_observer',False):
                tokens,calls,parameters=planners[family].classifier.charges(row,trees[row['id']],record['decisions'][-1]['path'])
                slots=calls*planners[family].classifier._recipe['max_length']
                assert (record['verifier_tokens'],record['verifier_calls'],record['verifier_token_slots'])==(tokens,calls,slots)
                verifier=parameters*slots
            assert np.isclose(record['compute'],(4*solver+3*review+verifier)/3.09)
            if 'total_parameter_compute' in record:
                assert np.isclose(record['total_parameter_compute'],(4*solver+35*review+verifier)/3.09)
            assert record['correct']==int(correct(trees[row['id']][path],row['answer']))
            checked+=1
            if family in BASELINES:continue
            model=planners.get(family)
            expected={k:v for k,v in record.items() if k not in ('policy','correct')}
            paid_tree={p:trees[row['id']][p] for p in visited}
            paid_reviews={(row['id'],p):active[row['id'],p] for p in acquired}
            for reference in ('','918273645546372819'):
                if model is not None:
                    for attr in ('_observed','_current_context','_review_context'):
                        cache=getattr(model,attr,None)
                        if cache is not None:cache.clear()
                    cache=getattr(model.classifier,'_probability_cache',None)
                    if cache is not None:cache.clear()
                    getattr(getattr(model,'review_observer',None),'_cache',{}).clear()
                result=run_policy(family,{**row,'answer':reference},paid_tree,paid_reviews,paid_reviews,
                                  model,frozen['selected'][family])
                result.pop('correct')
                assert result==expected,(family,row['id'])
                blinded+=1
    receipt={'questions':800,'independent_graded_charge_checks':checked,'blinded_pruned_policy_replays':blinded,
             'feature_caches_cleared':True,'historical_disjointness':True,'passed':True,
             'freeze_sha256':sha(args.output/'frozen.json')}
    (args.output/'audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args())
