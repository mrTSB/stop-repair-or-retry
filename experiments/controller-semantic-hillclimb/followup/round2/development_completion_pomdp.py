"""Select the new Complete-action A on all256 DEV; never collect or read TEST.

Requires the independently sealed384 science action grid and a fixed TRAIN
fit. Evaluates seven fixed penalties plus one subject-specific selection.
All six new matched controls share the same retained math component; science
panels also show their direct fixed-action behavior. Original five unchanged.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import completion_pomdp as core
import completion_subject_pomdp as routed
import development_nonlinear as cached
import development_stronger_metadata as reference
import expanded_mentor_io as io
import fit_completion_pomdp as fitted
import scoring_completion_availability as parsing
import stronger_base_hybrid as base_policy
import training_sparse_text as guarded
from prepare_mixed_mentor import plain_data
from study import fixed
from study_common import BASELINES, statistics

SCHEMA = 'science-terminal-completion-development-A-v1'
PENALTIES = (0., .01, .02, .04, .08, .16, .24)
MATCHED = (('Complete','',''), ('Repair then Complete','repair',''), ('Retry then Complete','retry',''),
           ('Stop if valid else Complete','','valid'), ('Repair then Complete if invalid','repair','valid'),
           ('Retry then Complete if invalid','retry','valid'))


def validate_collection_binding(collection, complete, model, models_directory, registration_sha256):
    import development_complete_actions as actions
    if (collection['schema'] != actions.SCHEMA or complete['schema'] != actions.SCHEMA
        or collection['evaluation_penalties'] != list(PENALTIES)
        or collection['model_sha256'] != model['model_sha256']
        or complete['registration_sha256'] != registration_sha256
        or complete['coverage'] != {'expected_cells':384,'physical_calls':384,'logical_actions':384,
                                    'reused_physical_calls':0,'new_physical_calls':384}
        or collection['test_used'] or complete['test_used']
        or Path(collection['paths']['model']).resolve() != models_directory.resolve()):
        raise ValueError('Development grid changed its fixed model, complete coverage, or registered penalties')
    io.verify(collection['source_files'])


def registered_inputs(args):
    """Bind sources and completed ledgers before evaluating any new outcome."""
    model = fitted.verify_complete(args.models)
    collection = io.read(args.complete/'registration.json')
    complete = io.read(args.complete/'complete.json')
    validate_collection_binding(collection,complete,model,args.models,io.sha(args.complete/'registration.json'))
    panel = io.sha(args.data/'splits/dev-questions.jsonl')
    chosen, references, pending = reference.references(args.registry, panel, False)
    evidence_paths = [args.models/'registration.json',args.models/'complete.json',args.models/'completion-A.pkl',
        args.complete/'registration.json',args.complete/'complete.json',args.data/'splits/dev-questions.jsonl',
        args.data/'dev-plain-complete.json',args.data/'dev-observations.jsonl',
        args.features/'complete.json',args.features/'metadata.jsonl',args.features/'neural.npz',
        args.base/'identity.json',args.base/'C1-smooth8.pkl',Path(chosen['summary']),Path(chosen['model_path'])]
    evidence = {str(path.resolve()):io.sha(path) for path in evidence_paths}
    evidence.update({str((args.complete/name).resolve()):value for name,value in complete['digests'].items()})
    for item in references:
        evidence[item['summary']] = item['summary_sha256']
    identity = {'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{name:str(getattr(args,name).resolve()) for name in ('data','complete','models','base','features','registry','output')},
        'development_questions':256,'science_questions':128,'dev_questions_sha256':panel,
        'fitted_model_sha256':model['model_sha256'],'base_selection':chosen,'A_references':references,
        'pending_optional_A':pending,'penalties':list(PENALTIES),'candidate_count':8,
        'selection':'Highest DEV accuracy then lowest compute; seven global penalties and one subject-specific policy',
        'all_five_original_baselines_unchanged':True,'matched_controls':[item[0] for item in MATCHED],
        'matched_full_panel_math':'Identical retained math A for all six added controls and all new A candidates',
        'science_control_panels':True,'new_neural_calls':0,'new_cloud_calls':0,
        'test_used':False,'fresh_validation':False,'runtime_policy_module':'completion_subject_pomdp',
        'strongest_previous_A_retained_if_better':True,'new_answer_changing_action':True}
    io.verify(evidence)
    return identity


def cached_observer(args, rows, trees, original):
    receipt = io.read(args.features/'complete.json')
    if (receipt['dev_questions_sha256'] != io.sha(args.data/'splits/dev-questions.jsonl')
        or receipt['encoder_recipe_sha256'] != io.sha(Path(original.directory)/'recipe.json')
        or receipt['neural_sha256'] != io.sha(args.features/'neural.npz')
        or receipt['metadata_sha256'] != io.sha(args.features/'metadata.jsonl')
        or receipt['nodes'] != 768 or receipt['test_used']):
        raise ValueError('Require the exact existing full768 paid old-observer DEV cache')
    original._recipe=io.read(Path(original.directory)/'recipe.json')
    metadata=[json.loads(line) for line in (args.features/'metadata.jsonl').read_text().splitlines()]
    with np.load(args.features/'neural.npz',allow_pickle=False) as arrays:
        lookup,checked=cached.verify_features(rows,trees,original,metadata,arrays,True)
    observer=cached.CachedObserver(original.directory,original.temperature,calibration=original.calibration)
    observer._recipe,observer._retriever,observer._backgrounds=original._recipe,original._retriever,original._backgrounds
    observer.saved=lookup; observer.input_sha256={(r['id'],r['path']):r['input_sha256'] for r in checked}
    return observer


def matched_item(row, tree, prefix, conditional):
    offered=parsing.offered_letters({**row,'answer':''})
    paths=['']+([prefix] if prefix else [])
    if not conditional or tree[prefix]['answer'] not in offered:
        paths.append((prefix+'/' if prefix else '')+'complete')
    item=core.plain.outcome(row,tree,{},paths[-1],paths,[],[])
    item.update(verifier_tokens=0,verifier_calls=0,verifier_token_slots=0,
                completion_calls=int(paths[-1].endswith('complete')),new_answer_changing_action=True)
    return item


def fit_rank(item):
    return item['dev']['accuracy'],-item['dev']['mean_compute']


def evaluate(args, identity):
    import development_complete_actions as actions
    science_rows,science_trees,collection_receipt=actions.development_data(args.data,args.complete)
    rows,trees=plain_data(args.data,'dev')
    if len(rows)!=256 or len(science_rows)!=128:
        raise ValueError('Use the entire registered256 development panel')
    by_id={r['id']:r for r in rows}
    if {r['id'] for r in science_rows}!={r['id'] for r in rows if r['subject']!='math'}:
        raise ValueError('Science action grid replaced or omitted development questions')
    for row in science_rows:
        if row!=by_id[row['id']]:raise ValueError('Changed development question/reference')
        for path,node in trees[row['id']].items():
            if node!=science_trees[row['id']][path]:raise ValueError('A plain control changed')
        trees[row['id']]=science_trees[row['id']]
    original_science=pickle.loads((args.models/'completion-A.pkl').read_bytes())
    original_science.classifier._recipe=io.read(Path(original_science.classifier.directory)/'recipe.json')
    chosen=identity['base_selection']; saved=pickle.loads(Path(chosen['model_path']).read_bytes())
    original_math,math_settings=base_policy.resolve_base(saved,chosen['module'],chosen['candidate'].get('settings',{}),'math')
    if base_policy.asset_id(original_math.classifier)!=base_policy.asset_id(original_science.classifier):
        raise ValueError('Retained math observer needs a separate validated cache; never silently substitute an asset')
    observer=cached_observer(args,rows,trees,copy.deepcopy(original_science.classifier))
    science,math=copy.deepcopy(original_science),copy.deepcopy(original_math)
    science.classifier=observer; math.classifier=observer
    math_fn=importlib.import_module('continuous_repairability' if 'posterior_weight' in math_settings else 'repairability_pomdp').evaluate
    math_items={r['id']:math_fn(r,trees[r['id']],{},math,math_settings,'A') for r in rows if r['subject']=='math'}
    fixed_items={name:[fixed(r,trees[r['id']],*setting) for r in rows] for name,setting in BASELINES.items()}
    fixed_metrics={name:statistics(values) for name,values in fixed_items.items()}
    fixed_science={name:{s:statistics([v for r,v in zip(rows,values)
                    if (r['subject']!='math' if s=='all' else r['subject']==s)])
                    for s in ('all','chemistry','physics')} for name,values in fixed_items.items()}
    matched={}; matched_private={}
    for name,prefix,conditional in MATCHED:
        values=[math_items[r['id']] if r['subject']=='math' else matched_item(r,trees[r['id']],prefix,conditional) for r in rows]
        matched[name]={'full256_with_retained_math_A':statistics(values),
            'science':{s:statistics([v for r,v in zip(rows,values) if (r['subject']!='math' if s=='all' else r['subject']==s)])
                       for s in ('all','chemistry','physics')}}
        matched_private[name]=values
    candidates=[]; paired={}; by_subject={s:[] for s in ('chemistry','physics')}
    for penalty in PENALTIES:
        settings={'penalty':penalty}
        values=[math_items[r['id']] if r['subject']=='math' else core.evaluate(r,trees[r['id']],{},science,settings,'A') for r in rows]
        key=f'penalty-{penalty:g}'
        candidate={'key':key,'science_settings':settings,'dev':statistics(values)}
        candidates.append(candidate); paired[key]=values
        for subject in by_subject:
            by_subject[subject].append({'settings':settings,'dev':statistics([v for r,v in zip(rows,values) if r['subject']==subject])})
    settings={'by_subject':{s:max(values,key=fit_rank)['settings'] for s,values in by_subject.items()}}
    values=[math_items[r['id']] if r['subject']=='math' else core.evaluate(r,trees[r['id']],{},science,settings,'A') for r in rows]
    candidate={'key':'subject-specific','science_settings':settings,'dev':statistics(values)}
    candidates.append(candidate);paired[candidate['key']]=values
    best=max(candidates,key=fit_rank)
    final,refs,pending=reference.references(args.registry,identity['dev_questions_sha256'],False)
    deploy=routed.Planner(original_math,original_science,math_settings,best['science_settings'],
        {'schema':SCHEMA,'new_science_state_count':16,'math_state_count':8,
         'math_source':chosen,'science_source_sha256':identity['fitted_model_sha256'],
         'registration_sha256':io.sha(args.output/'registration.json'),'new_action_protocol':True,
         'no_reference_or_counterfactual_observations':True})
    model_bytes=pickle.dumps(deploy,protocol=5)
    if io.structural_digest(pickle.loads(model_bytes))!=io.structural_digest(deploy):
        raise ValueError('Saved new A changed learned state')
    io.atomic_bytes(args.output/'completion-subject-A.pkl',model_bytes)
    completion_spec={'source':'new-completion-POMDP',**best,
        'model_path':str((args.output/'completion-subject-A.pkl').resolve()),
        'model_sha256':io.sha(args.output/'completion-subject-A.pkl'),
        'policy_module':'completion_subject_pomdp','settings':{},
        'embedded_settings':{'math':math_settings,'science':best['science_settings']}}
    prior_spec={'source':'previous-A',**final['candidate'],
        'model_path':final['model_path'],'model_sha256':final.get('model_sha256',io.sha(final['model_path'])),
        'policy_module':final['module'],'source_summary':final['summary'],
        'source_summary_sha256':final.get('summary_sha256',io.sha(final['summary']))}
    retained=max([completion_spec,prior_spec],key=fit_rank)
    io.write(args.output/'identity.json',{'schema':SCHEMA,'source_sha256':io.sha(__file__),
        'policy_module':'completion_subject_pomdp','policy_sha256':io.sha(routed.__file__),
        'dev_questions_sha256':identity['dev_questions_sha256'],
        'registration_sha256':io.sha(args.output/'registration.json'),
        'new_answer_changing_action':True,'test_used':False,
        'selection_adapter_required':'New action-aware resolver; old eight-state hybrid routes stay unchanged',
        'selected_A':retained})
    io.write(args.output/'search.json',candidates)
    io.write(args.output/'private-candidate-items.json',paired)
    io.write(args.output/'private-matched-control-items.json',matched_private)
    io.verify(identity['source_files']);io.verify(identity['input_evidence'])
    highest=max(v['accuracy'] for v in fixed_metrics.values())
    result={'schema':SCHEMA,'development_questions':256,'baselines':fixed_metrics,
        'baseline_science_panels':fixed_science,
        'matched_complete_controls':matched,'matched_control_math_rule':identity['matched_full_panel_math'],
        'completion_A':completion_spec,
        'selected_A':retained,'strongest_previous_A':final,'A_references':refs,'pending_optional_A':pending,
        'A_above_highest_baseline':retained['dev']['accuracy']-highest,
        'A_plus_10_points':retained['dev']['accuracy']>=highest+.1-1e-12,
        'best_matched_complete_control_accuracy':max(v['full256_with_retained_math_A']['accuracy'] for v in matched.values()),
        'completion_A_above_best_matched_control':best['dev']['accuracy']-max(v['full256_with_retained_math_A']['accuracy'] for v in matched.values()),
        'fitting_questions':256,'new_neural_calls':0,'new_cloud_calls':0,
        'actual_cached_observations_billed':True,'test_used':False,'fresh_validation':False,
        'collection_receipt':collection_receipt,'math_metrics':statistics(list(math_items.values())),
        'registration_sha256':io.sha(args.output/'registration.json')}
    io.write(args.output/'summary.json',result)
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':io.sha(args.output/'registration.json'),
        'artifacts':{p.name:io.sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and not p.name.startswith('.')},
        'test_used':False,'new_neural_calls':0,'new_cloud_calls':0})
    print({'complete':True,'new_A':best['dev'],'selected_A':retained['dev'],
           'A_plus_10_points':result['A_plus_10_points'],'fresh_validation':False},flush=True)


def run(args):
    with guarded.no_model_calls(),threadpool_limits(limits=1):
        identity=registered_inputs(args)
        if args.mode=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',identity)
            print({'registered':True,'registration_sha256':io.sha(args.output/'registration.json')},flush=True);return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
            or io.read(args.output/'registration.json')!=identity):raise ValueError('Changed registered development evaluation')
        with io.lock(args.output/'.evaluation.lock'):
            if any(p.name not in ('registration.json','.evaluation.lock') for p in args.output.iterdir()):
                raise ValueError('Preserve prior or partial evaluation; do not overwrite results')
            evaluate(args,identity)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','evaluate'))
    for name in ('data','complete','models','base','features','registry','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256')
    run(parser.parse_args())
