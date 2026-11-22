"""Registered16-candidate DEV comparison of TRAIN-clustered observation models."""
import argparse
import copy
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import development_nonlinear_completion as previous
import fit_clustered_completion as fitted
import clustered_completion_pomdp as policy

io,original,nested=previous.io,previous.original,previous.nested
SCHEMA='expanded768-clustered-Complete16-full-development-v1'


def inputs(args):
    old_spec=io.read(args.previous/'registration.json');old_done=io.read(args.previous/'complete.json')
    old_args=SimpleNamespace(**{k:Path(v) for k,v in old_spec['paths'].items()},registration_sha256=io.sha(args.previous/'registration.json'))
    if (old_done['schema']!=previous.SCHEMA or old_done['registration_sha256']!=old_args.registration_sha256
            or old_args.output.resolve()!=args.previous.resolve() or old_done['test_used']):raise ValueError('Require completed expanded development comparison')
    io.verify({str(args.previous/n):v for n,v in old_done['artifacts'].items()})
    data,old_models,values,parent,expected=previous.inputs(old_args)
    if expected!=old_spec:raise ValueError('Previous study inputs changed')
    io.verify(old_spec['source_files']);io.verify(old_spec['input_evidence'])
    spec=io.read(args.models/'registration.json');done=io.read(args.models/'complete.json')
    if (spec['schema']!=fitted.SCHEMA or done['schema']!=fitted.SCHEMA or done['training_questions']!=768
            or done['registration_sha256']!=io.sha(args.models/'registration.json')
            or spec['paths']['models']!=str(old_args.models.resolve())
            or done['new_neural_calls'] or done['cloud_calls'] or done['development_or_test_read']):
        raise ValueError('Require completed TRAIN-only clustered refit of the exact previous model')
    io.verify(spec['source_files']);io.verify(spec['input_evidence']);io.verify({str(args.models/n):v for n,v in done['digests'].items()})
    models={}
    for family in nested.FAMILIES:
        path=args.models/(family+'.pkl')
        if io.sha(path)!=done['models'][family]['model_sha256']:raise ValueError('Kernel model changed')
        model=pickle.loads(path.read_bytes())
        if model.fit_receipt!=done['models'][family]['fit_receipt'] or model.feature_family!=family:raise ValueError('Kernel fit receipt changed')
        if io.read(Path(model.classifier.directory)/'recipe.json')!=data[3]._recipe:raise ValueError('Different observation encoder')
        for subject,d in model.dynamics.items():
            for field in ('prior','transition','cost'):np.testing.assert_array_equal(getattr(d,field),getattr(old_models[family].dynamics[subject],field))
        models[family]=model
    evidence={**old_spec['input_evidence'],**spec['input_evidence'],
        **{str((args.previous/n).resolve()):v for n,v in old_done['artifacts'].items()},
        **{str((args.models/n).resolve()):v for n,v in done['digests'].items()},
        str((args.previous/'complete.json').resolve()):io.sha(args.previous/'complete.json'),
        str((args.models/'complete.json').resolve()):io.sha(args.models/'complete.json')}
    old_summary=io.read(args.previous/'summary.json')
    record={'schema':SCHEMA,'source_files':io.sources(sys.modules[__name__]),'input_evidence':evidence,
        'paths':{k:str(getattr(args,k).resolve()) for k in ('previous','models','output')},
        'previous_A_candidates':[*old_spec['previous_A_candidates'],old_summary['selected_A']],
        'dev_questions_sha256':old_spec['dev_questions_sha256'],'families':list(nested.FAMILIES),'penalties':list(original.PENALTIES),
        'candidate_count':16,'development_questions':256,'training_questions':768,
        'new_neural_calls':0,'new_cloud_calls':0,'test_used':False,'fresh_validation':False,
        'partition_selection':'TRAIN child forecasts only; fixed24+6 clusters and no DEV partition tuning',
        'policy_selection':'Accuracy then compute; seven fixed penalties plus one subject-specific candidate per family',
        'exact_finite_Bellman':True,'all_original_and_matched_controls_retained':True,'math_unchanged':True}
    return data,models,values,parent,record


def replay(args,data,models,values,parent_path,record):
    rows,trees,parent,old_observer,parent_summary,_,bundle=data
    live=copy.deepcopy(parent);live.math.classifier=live.science.classifier=old_observer
    old_items=[previous.routed.evaluate(r,trees[r['id']],{},live,{},'A') for r in rows]
    if old_items!=io.read(parent_path/'private-candidate-items.json')[parent_summary['completion_A']['key']]:raise ValueError('Original decisions/costs changed')
    math={r['id']:v for r,v in zip(rows,old_items) if r['subject']=='math'}
    if original.statistics(list(math.values()))!=parent_summary['math_metrics']:raise ValueError('Original math changed')
    baselines,matched=previous.previous.controls(rows,trees,math,parent_summary)
    observer=previous.previous.CachedDense(old_observer.directory,old_observer.temperature)
    observer._recipe=old_observer._recipe;observer._retriever=old_observer._retriever;observer._backgrounds=old_observer._backgrounds
    observer.expected={(v['id'],v['path']):v['input_sha256'] for v in bundle['metadata']}
    observer.saved={(v['id'],v['path']):(*[values[k][i] for k in ('state','plain','guided')],int(values['token_slots'][i]))
        for i,v in enumerate(bundle['metadata'])}
    observer.joint={(v['id'],v['path']):values['joint256'][i] for i,v in enumerate(bundle['metadata'])}
    candidates=[];all_items={}
    for family,model in models.items():
        deploy_model=copy.deepcopy(model);model.classifier=observer;grid=[]
        for penalty in original.PENALTIES:
            grid.append([math[r['id']] if r['subject']=='math' else policy.evaluate_science(r,trees[r['id']],{},model,{'penalty':penalty},'A') for r in rows])
        chosen={s:max(range(7),key=lambda i:original.fit_rank({'dev':original.statistics([v for r,v in zip(rows,grid[i]) if r['subject']==s])})) for s in ('chemistry','physics')}
        grid.append([math[r['id']] if r['subject']=='math' else grid[chosen[r['subject']]][i] for i,r in enumerate(rows)])
        for index,items in enumerate(grid):
            settings={'penalty':original.PENALTIES[index]} if index<7 else {'by_subject':{s:{'penalty':original.PENALTIES[i]} for s,i in chosen.items()}}
            deploy=policy.SubjectPlanner(copy.deepcopy(parent.math),copy.deepcopy(deploy_model),parent.math_settings,settings,
                {'schema':SCHEMA,'training_model_sha256':io.sha(args.models/(family+'.pkl')),'math_unchanged':True})
            name=family+'-'+str(index);path=args.output/(name+'.pkl');io.atomic_bytes(path,pickle.dumps(deploy,protocol=5))
            candidates.append({'key':name,'family':family,'model_path':str(path.resolve()),'model_sha256':io.sha(path),
                'policy_module':'clustered_completion_pomdp','settings':{},'science_settings':settings,'dev':original.statistics(items)})
            all_items[name]=items
    selected=max([*record['previous_A_candidates'],*candidates],key=original.fit_rank)
    summary={'schema':SCHEMA,'selected_A':selected,'candidates':candidates,'previous_A_candidates':record['previous_A_candidates'],
        'baselines':baselines,'matched_complete_controls':matched,'parent_exactly_reproduced':True,'development_questions':256,
        'A_above_highest_baseline':selected['dev']['accuracy']-max(v['accuracy'] for v in baselines.values()),
        'new_neural_calls':0,'new_cloud_calls':0,'test_used':False,'fresh_validation':False,'registration_sha256':args.registration_sha256}
    io.write(args.output/'private-candidate-items.json',all_items);io.write(args.output/'summary.json',summary)
    io.write(args.output/'identity.json',{'schema':SCHEMA,'selected_A':selected,'dev_questions_sha256':record['dev_questions_sha256'],
        'source_sha256':io.sha(__file__),'policy_sha256':io.sha(policy.__file__),'policy_module':'clustered_completion_pomdp',
        'registration_sha256':args.registration_sha256,'test_used':False})
    io.verify(record['source_files']);io.verify(record['input_evidence'])
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'artifacts':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
        'new_neural_calls':0,'new_cloud_calls':0,'test_used':False})
    print({'selected_A':selected['dev'],'new_candidates':[(v['key'],v['dev']['correct']) for v in candidates]},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','replay'))
    for n in ('previous','models','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--registration-sha256');args=p.parse_args()
    with previous.previous.parent_io.no_test_reads(),original.guarded.no_model_calls(),threadpool_limits(limits=1):
        data,models,values,parent,record=inputs(args)
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'scored':False});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Changed quantile development registration')
        with io.lock(args.output/'.run.lock'):
            previous.previous.preserve_partial(args.output);replay(args,data,models,values,parent,record)


if __name__=='__main__':main()
