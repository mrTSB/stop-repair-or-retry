"""One preregistered fixed-half Complete-A DEV replay; no hyperparameter search."""
import argparse
import builtins
from collections import Counter
from contextlib import contextmanager
import copy
import io as python_io
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

from threadpoolctl import threadpool_limits

import completion_blend_pomdp as policy
import completion_filter_diagnostic as diagnostic
import development_complete_actions as actions
import development_completion_pomdp as original
import expanded_mentor_io as io

SCHEMA='fixed-half-completion-A-development-v1'


@contextmanager
def no_test_reads():
    ordinary_open=builtins.open;path_open=python_io.open
    def guarded(function):
        def call(file,*args,**kwargs):
            if isinstance(file,(str,bytes,Path)):
                name=Path(file.decode() if isinstance(file,bytes) else file).name
                if name.startswith('test-') and name.endswith(('.jsonl','.json','.npz')):
                    raise AssertionError('Fresh TEST inputs are forbidden in fixed DEV replay')
            return function(file,*args,**kwargs)
        return call
    with patch.object(builtins,'open',guarded(ordinary_open)),patch.object(python_io,'open',guarded(path_open)):
        yield


def verified_artifacts(directory,schema):
    done=io.read(directory/'complete.json')
    if done['schema']!=schema:raise ValueError('Wrong completed parent/diagnostic schema')
    io.verify({str(directory/name):value for name,value in done['artifacts'].items()})
    registered=io.read(directory/'registration.json')
    if done['registration_sha256']!=io.sha(directory/'registration.json'):
        raise ValueError('Completed input registration changed')
    io.verify(registered['source_files']);io.verify(registered['input_evidence'])
    return registered,done


def load_inputs(args):
    registered,done=verified_artifacts(args.parent,original.SCHEMA)
    source=SimpleNamespace(**{k:Path(v) for k,v in registered['paths'].items()})
    diagnostic_spec,diagnostic_done=verified_artifacts(args.diagnostic,diagnostic.SCHEMA)
    diagnostic_summary=io.read(args.diagnostic/'summary.json')
    if (diagnostic_spec['posterior_blend_weight']!=policy.WEIGHT or diagnostic_spec['regularization']!=.1
        or diagnostic_spec['smoothing']!=32. or diagnostic_summary['reports']['screen_passed'] is not True
        or diagnostic_summary['development_or_test_read'] is not False
        or Path(diagnostic_spec['paths']['base_fit']).resolve()!=source.models.resolve()):
        raise ValueError('Require the completed fixed nested TRAIN screen on this exact parent fit')
    summary=io.read(args.parent/'summary.json');identity=io.read(args.parent/'identity.json')
    chosen=summary['completion_A'];model_path=args.parent/'completion-subject-A.pkl'
    if (summary['development_questions']!=256 or summary['test_used'] or summary['fresh_validation']
        or chosen['model_sha256']!=io.sha(model_path) or Path(chosen['model_path']).resolve()!=model_path.resolve()
        or chosen['science_settings']!={'penalty':policy.PENALTY}
        or identity['source_sha256']!=io.sha(original.__file__)
        or identity['policy_sha256']!=io.sha(policy.parent_policy.__file__)
        or identity['registration_sha256']!=io.sha(args.parent/'registration.json')):
        raise ValueError('Require unchanged selected finite Complete-A parent')
    parent=pickle.loads(model_path.read_bytes())
    deploy=policy.build(parent,{'parent_directory':str(args.parent.resolve()),'parent_model_sha256':io.sha(model_path),
        'TRAIN_diagnostic_directory':str(args.diagnostic.resolve()),'TRAIN_diagnostic_complete_sha256':io.sha(args.diagnostic/'complete.json')})
    science_rows,science_trees,collection=actions.development_data(source.data,source.complete)
    rows,trees=original.plain_data(source.data,'dev')
    if (len(rows)!=256 or len({r['id'] for r in rows})!=256
        or Counter(r['subject'] for r in rows)!={'math':128,'chemistry':64,'physics':64}
        or len(science_rows)!=128 or io.sha(source.data/'splits/dev-questions.jsonl')!=identity['dev_questions_sha256']):
        raise ValueError('Require unchanged complete256 DEV panel')
    by_id={r['id']:r for r in rows}
    for row in science_rows:
        qid=row['id']
        if row!=by_id[qid] or any(science_trees[qid][p]!=v for p,v in trees[qid].items()):
            raise ValueError('Plain prefixes or science question changed')
        trees[qid]=science_trees[qid]
    if (type(parent.math.classifier) is not type(parent.science.classifier)
        or original.base_policy.asset_id(parent.math.classifier)!=original.base_policy.asset_id(parent.science.classifier)):
        raise ValueError('Do not substitute incompatible math/science observer caches')
    observer=original.cached_observer(source,rows,trees,copy.deepcopy(parent.science.classifier))
    reference,refs,pending=original.reference.references(source.registry,identity['dev_questions_sha256'],False)
    prior=summary['selected_A']
    legacy={**reference['candidate'],'model_path':reference['model_path'],'model_sha256':reference['model_sha256'],
        'policy_module':reference['module'],'source_summary':reference['summary'],'source_summary_sha256':reference['summary_sha256']}
    evidence={**registered['input_evidence'],
        **{str((args.parent/name).resolve()):value for name,value in done['artifacts'].items()},
        str((args.parent/'complete.json').resolve()):io.sha(args.parent/'complete.json'),
        str((args.diagnostic/'registration.json').resolve()):io.sha(args.diagnostic/'registration.json'),
        str((args.diagnostic/'complete.json').resolve()):io.sha(args.diagnostic/'complete.json'),
        str((args.diagnostic/'summary.json').resolve()):io.sha(args.diagnostic/'summary.json')}
    for ref in refs:
        evidence[ref['summary']]=ref['summary_sha256']
        resolved=ref['resolved'];evidence[resolved['model_path']]=resolved['model_sha256']
        evidence[resolved['summary']]=resolved['summary_sha256']
        evidence[str(Path(resolved['summary']).parent/'identity.json')]=resolved['identity_sha256']
    io.verify(evidence)
    receipt={'paths':{k:str(getattr(args,k).resolve()) for k in ('parent','diagnostic','output')},
        'input_evidence':evidence,'parent_model_sha256':io.sha(model_path),
        'parent_identity_sha256':io.sha(args.parent/'identity.json'),'parent_summary_sha256':io.sha(args.parent/'summary.json'),
        'parent_structural_sha256':io.structural_digest(parent),
        'math_structural_sha256':io.structural_digest(parent.math),
        'science_structural_sha256':io.structural_digest(parent.science),
        'deploy_structural_sha256':io.structural_digest(deploy),
        'dev_questions_sha256':identity['dev_questions_sha256'],'collection_receipt':collection,
        'previous_A_candidates':[prior,legacy],'A_references':refs,'pending_optional_A':pending,
        'parent_settings':{'math':parent.math_settings,'science':parent.science_settings}}
    return rows,trees,parent,deploy,observer,summary,receipt


def specification(receipt):
    return {**receipt,'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'candidate_count':1,
        'science_penalty':policy.PENALTY,'posterior_weight':policy.WEIGHT,'development_questions':256,
        'new_neural_calls':0,'new_cloud_calls':0,'cached_raw_observations':768,'test_used':False,'fresh_validation':False,
        'parent_full_TRAIN_model_unchanged':True,'math_route_and_root_forecasts_unchanged':True,
        'original_five_and_six_matched_controls_retained':True,'approximate_belief_estimator':True,
        'finite_Bellman_backup_unchanged':True,'limitation':policy.LIMITATION,
        'selection':'Retain strongest previous A if higher accuracy; lower existing compute proxy breaks accuracy ties',
        'no_penalty_or_threshold_search':True,'online_cached_observation_billing_unchanged':True}


def select(candidate,previous,baselines,matched):
    if len(baselines)!=5 or len(matched)!=6 or not previous:
        raise ValueError('Retain all five original and six matched controls plus prior A')
    if any(r['dev']['n']!=256 for r in [candidate,*previous]):raise ValueError('Full same256 comparisons only')
    selected=max([*previous,candidate],key=original.fit_rank)
    highest=max(v['accuracy'] for v in baselines.values())
    matched_best=max(v['full256_with_retained_math_A']['accuracy'] for v in matched.values())
    return {'selected_A':selected,'blend_A':candidate,'previous_A_candidates':previous,
        'retained_previous_A':selected is not candidate,'baselines':baselines,'matched_complete_controls':matched,
        'A_above_highest_baseline':selected['dev']['accuracy']-highest,
        'A_plus_10_points':selected['dev']['accuracy']>=highest+.1-1e-12,
        'blend_above_best_matched_control':candidate['dev']['accuracy']-matched_best,
        'best_matched_complete_control_accuracy':matched_best}


def replay(args,bundle,registered):
    rows,trees,parent,deploy,observer,summary,receipt=bundle
    live_parent,live_new=copy.deepcopy(parent),copy.deepcopy(deploy)
    for model in (live_parent,live_new):model.math.classifier=model.science.classifier=observer
    original_items=[policy.parent_policy.evaluate(r,trees[r['id']],{},live_parent,{},'A') for r in rows]
    saved=io.read(args.parent/'private-candidate-items.json')[summary['completion_A']['key']]
    if original_items!=saved or original.statistics(original_items)!=summary['completion_A']['dev']:
        raise ValueError('Exact original full256 decisions/charges/result failed reproduction')
    new_items=[policy.evaluate(r,trees[r['id']],{},live_new,{},'A') for r in rows]
    math={r['id']:v for r,v in zip(rows,original_items) if r['subject']=='math'}
    if (any(new!=old for row,new,old in zip(rows,new_items,original_items) if row['subject']=='math')
        or original.statistics(list(math.values()))!=summary['math_metrics']):
        raise ValueError('Retained math route changed')
    if any(new['decisions'][0]!=old['decisions'][0] for new,old in zip(new_items,original_items)):
        raise ValueError('Root forecast/first action changed before a child update')
    baselines={name:original.statistics([original.fixed(r,trees[r['id']],*setting) for r in rows])
        for name,setting in original.BASELINES.items()}
    matched={}
    for name,prefix,conditional in original.MATCHED:
        values=[math[r['id']] if r['subject']=='math' else original.matched_item(r,trees[r['id']],prefix,conditional) for r in rows]
        matched[name]={'full256_with_retained_math_A':original.statistics(values),
            'science':{s:original.statistics([v for r,v in zip(rows,values) if (r['subject']!='math' if s=='all' else r['subject']==s)])
                for s in ('all','chemistry','physics')}}
    if baselines!=summary['baselines'] or matched!=summary['matched_complete_controls']:
        raise ValueError('An original baseline or matched completion control changed')
    for name,component in (('math',deploy.math),('science',deploy.science)):
        if io.structural_digest(component)!=receipt[name+'_structural_sha256']:
            raise ValueError('Deployment changed a fitted parent component')
    raw=pickle.dumps(deploy,protocol=5)
    if io.structural_digest(pickle.loads(raw))!=receipt['deploy_structural_sha256']:
        raise ValueError('Deployable wrapper changed on serialization')
    path=args.output/'completion-blend-A.pkl';io.atomic_bytes(path,raw)
    candidate={'model':'fixed-half-completion-A','model_path':str(path.resolve()),'model_sha256':io.sha(path),
        'policy_module':'completion_blend_pomdp','settings':{},'embedded_settings':{'science_penalty':policy.PENALTY,
        'posterior_weight':policy.WEIGHT,'math':deploy.math_settings},'dev':original.statistics(new_items)}
    result=select(candidate,registered['previous_A_candidates'],baselines,matched)
    result.update(schema=SCHEMA,development_questions=256,math_metrics=summary['math_metrics'],
        parent_A=summary['completion_A'],parent_exactly_reproduced=True,
        new_neural_calls=0,new_cloud_calls=0,cached_raw_observations=768,test_used=False,fresh_validation=False,
        candidate_count=1,approximate_belief_estimator=True,finite_Bellman_backup_unchanged=True,
        limitation=policy.LIMITATION,registration_sha256=args.registration_sha256)
    io.jsonl(args.output/'private-parent-items.jsonl',original_items);io.jsonl(args.output/'private-blend-items.jsonl',new_items)
    io.write(args.output/'summary.json',result)
    io.write(args.output/'identity.json',{'schema':SCHEMA,'source_sha256':io.sha(__file__),
        'policy_module':'completion_blend_pomdp','policy_sha256':io.sha(policy.__file__),
        'dev_questions_sha256':registered['dev_questions_sha256'],'registration_sha256':args.registration_sha256,
        'selected_A':result['selected_A'],'test_used':False,'approximate_belief_estimator':True})
    io.verify(registered['source_files']);io.verify(registered['input_evidence'])
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'artifacts':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
        'new_neural_calls':0,'new_cloud_calls':0,'test_used':False})
    print({'complete':True,'blend_A':candidate['dev'],'selected_A':result['selected_A']['dev'],
        'A_plus_10_points':result['A_plus_10_points'],'fresh_validation':False},flush=True)


def run(args):
    with no_test_reads(),original.guarded.no_model_calls(),threadpool_limits(limits=1):
        bundle=load_inputs(args);registered=specification(bundle[-1])
        if args.mode=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',registered)
            print({'registered':True,'registration_sha256':io.sha(args.output/'registration.json'),'scored':False},flush=True);return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
            or io.read(args.output/'registration.json')!=registered):raise ValueError('Changed explicit fixed comparison registration')
        with io.lock(args.output/'.replay.lock'):
            if any(p.name not in ('registration.json','.replay.lock') for p in args.output.iterdir()):
                raise ValueError('Preserve completed or partial comparison')
            replay(args,bundle,registered)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','replay'))
    for name in ('parent','diagnostic','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');run(parser.parse_args())
