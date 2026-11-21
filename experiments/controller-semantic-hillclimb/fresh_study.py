"""Freeze audited semantic policies and score the entire reserved fresh192 panel once."""

import argparse
import json
from pathlib import Path
import shutil
import pickle

from semantic_data import read
from semantic_study_common import (BASELINES,evidence,load_test,models,paired_margins,run_policy,sha,statistics)
from study import fixed
from hillclimb import new_run,save


def freeze(args):
    candidates=[]
    for source in args.probe:
        summary=json.loads((source/'summary.json').read_bytes())
        if summary['fresh_data_used']:raise ValueError('Probe used holdout data')
        identity=json.loads((source/'identity.json').read_bytes())
        audit=json.loads((source/'audit.json').read_bytes())
        if not audit['passed'] or audit['summary_sha256']!=sha(source/'summary.json'):
            raise ValueError('Audit every selected development source first')
        available=('stable_pomdp','grounded_pomdp','tree_context_pomdp','calibrated_review_pomdp','stacked_pomdp','efficient_stacked_pomdp','dynamic_grounded_pomdp')
        module=next((m for m in available if sha(Path(__file__).with_name(m+'.py'))==identity['semantic_code']),None)
        if module is None:raise ValueError('Changed development POMDP source')
        protocol='late' if summary['selected']['C']['settings'].get('late_feedback',False) else 'original'
        for candidate in json.loads((source/'search.json').read_bytes()):
            candidates.append({**candidate,'probe':str(source),'protocol':('late' if candidate['late'] else 'original'),'module':module})
    best=lambda group:max(group,key=lambda r:(r['dev']['accuracy'],-r['dev']['mean_compute']))
    a=best([r for r in candidates if r['family']=='A'])
    b=best([r for r in candidates if r['family']=='B'])
    c_members=[r for r in candidates if r['family']=='C' and r['dev']['mean_judge_calls']>0
               and r['dev']['mean_compute']<b['dev']['mean_compute']]
    if not c_members:raise ValueError('No hybrid is cheaper than the strongest development-selected B')
    c=best(c_members)
    if c['dev']['accuracy']<max(a['dev']['accuracy'],b['dev']['accuracy'])+args.minimum_dev_lead:
        raise ValueError('Development C lead is below the declared validation threshold; continue hillclimbing')
    for name in ('test-observations.jsonl','test-late-observations.jsonl'):
        p=args.data/name
        if p.exists() and p.stat().st_size:raise ValueError('Fresh inference already started; cannot refreeze')
    test=read(args.data/'splits/test-questions.jsonl')
    train=read(args.data/'splits/train-questions.jsonl');dev=read(args.data/'splits/dev-questions.jsonl')
    if len(test)!=192:raise ValueError('Fresh panel must contain all 192 reserved questions')
    if {r['id'] for r in test}&{r['id'] for r in train+dev}:raise ValueError('Fit or selection IDs entered test')
    selected={'A':a,'B':b,'C':c}
    development_baseline=max(v['accuracy'] for v in json.loads((Path(a['probe'])/'summary.json').read_bytes())['baselines'].values())
    development_targets={'A_plus_10_points':a['dev']['accuracy']>=development_baseline+.1-1e-12,
                         'B_plus_10_points':b['dev']['accuracy']>=development_baseline+.1-1e-12,
                         'C_plus_10_points':c['dev']['accuracy']>=max(a['dev']['accuracy'],b['dev']['accuracy'])+.1-1e-12}
    with new_run(args.output) as stage:
        (stage/'models').mkdir()
        model_receipts={};observer_assets={}
        for family in ('A','C'):
            candidate=selected[family]
            source=Path(candidate['probe'])/(candidate['model']+'.pkl')
            destination=stage/'models'/(family+'.pkl')
            model=pickle.loads(source.read_bytes());extra={}
            folder=Path(model.classifier.directory)
            recipe=json.loads((folder/'recipe.json').read_bytes())
            key=sha(folder/'recipe.json')[:16];relative='observers/'+key
            if key not in observer_assets:
                shutil.copytree(folder,stage/relative)
                observer_assets[key]={'directory':relative,
                    'files':{p.name:sha(p) for p in (stage/relative).iterdir() if p.is_file()},
                    'training_contract':recipe,
                    'private_volume':'srr-semantic-training-20261120',
                    'remote':'/verifier-grounded-v1/full' if recipe.get('input_profile') else '/verifier-v2/full'}
            model.classifier.directory=relative;extra['observer_directory']=relative
            destination.write_bytes(pickle.dumps(model,protocol=5))
            model_receipts[family]={'file':'models/'+family+'.pkl','sha256':sha(destination),
                                   'probe_model_sha256':sha(source),**extra}
        payload={'selected':selected,'models':model_receipts,'evidence':evidence(args.data),
                 'observer_assets':observer_assets,
                 'baselines':BASELINES,'test_questions':192,'train_questions':len(train),'dev_questions':len(dev),
                 'targets':{'interpretation':'percentage points','A_B_above_highest_baseline':.10,
                            'C_above_highest_A_B':.10,'C_stretch_above_highest_A_B':.15,
                            'C_compute_below_B':True,'clear_C_B_accuracy_and_compute_ci95':True},
                 'development_validation_lead':args.minimum_dev_lead,
                 'development_targets':development_targets,
                 'validation_context':'Validate the best available development-selected policies once on all 192 questions. Lowering the optional development lead guard does not lower any final target or turn a missed target into a success.',
                 'selection':'A and B independently maximize development accuracy, then minimize compute. C maximizes development accuracy among hybrids cheaper than the strongest selected B.',
                 'roles':{'A':'Four-state POMDP with a trained discriminative observation encoder; no LLM reviewer acquired.',
                          'B':'LLM policy; no POMDP supplied.',
                          'C':'POMDP plus acquired LLM observations and finite-horizon review planning.'},
                 'observation_regeneration':'All inference-time raw neural logits are archived by model recipe and visible-response key; every use is charged even when cached. No ground-truth labels enter the collector.',
                 'budget':'Same frozen compact solver, original initial prompt/cap, at most two additional solver calls, and at most one review at each decision checkpoint.',
                 'compute':'Full input/output tokens: (4*solver + 3*reviewer + verifier_parameter_billions*verifier_padded_token_slots)/3.09. Proxy only; also report raw tokens, calls, and total-parameter sensitivity.',
                 'bootstrap':'5000 paired subject-stratified resamples; highest baseline/controller recomputed within each resample.'}
        save(stage/'frozen.json',payload)
    print(json.dumps({'selected':selected,'test_questions':192,'frozen':str(args.output/'frozen.json')}),flush=True)


def score(args):
    frozen=json.loads((args.output/'frozen.json').read_bytes())
    if frozen['evidence']!=evidence(args.data):raise ValueError('Source or development evidence changed after freeze')
    metadata=json.loads((args.data/'test-collection.json').read_bytes())
    if metadata['freeze_sha256']!=sha(args.output/'frozen.json') or metadata['completed']!=192:
        raise ValueError('Score only after all 192 questions complete under the same freeze')
    if metadata['neural_cache_sha256']!=sha(args.output/'test-neural-cache.jsonl'):
        raise ValueError('Changed reference-free neural prediction receipts')
    rows,trees,base_judges,late_judges=load_test(args.data)
    if len(rows)!=192:raise ValueError('Incomplete fresh question panel')
    planners=models(frozen,args.output)
    results={name:[fixed(r,trees[r['id']],*configuration) for r in rows]
             for name,configuration in BASELINES.items()}
    for family in ('A','B','C'):
        results[family]=[run_policy(family,r,trees[r['id']],base_judges,late_judges,
                                   planners.get(family),frozen['selected'][family]) for r in rows]
    stats={k:statistics(v) for k,v in results.items()}
    margins=paired_margins(rows,results)
    criteria={'A_plus_10_points':margins['A_above_highest_baseline']['point']>=.10-1e-12,
              'B_plus_10_points':margins['B_above_highest_baseline']['point']>=.10-1e-12,
              'C_plus_10_points':margins['C_above_highest_A_B']['point']>=.10-1e-12,
              'C_plus_15_points_stretch':margins['C_above_highest_A_B']['point']>=.15-1e-12,
              'C_accuracy_clearly_above_B':margins['C_above_B']['ci95'][0]>0,
              'C_compute_clearly_below_B':margins['C_compute_saved_vs_B']['ci95'][0]>0}
    highest=max(BASELINES,key=lambda k:(stats[k]['accuracy'],-stats[k]['mean_compute']))
    with new_run(args.output/'test') as stage:
        save(stage/'summary.json',{'methods':stats,'margins':margins,'criteria':criteria,
                                   'highest_baseline':highest,'all_required_passed':all(
                                       value for key,value in criteria.items() if key!='C_plus_15_points_stretch')})
        with (stage/'per_question.jsonl').open('w') as stream:
            for family,items in results.items():
                for item in items:stream.write(json.dumps({'policy':family,**item})+'\n')
    print(json.dumps({'methods':stats,'margins':margins,'criteria':criteria},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','score'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',type=Path,nargs='+')
    parser.add_argument('--minimum-dev-lead',type=float,default=.10)
    args=parser.parse_args()
    (freeze if args.phase=='freeze' else score)(args)
