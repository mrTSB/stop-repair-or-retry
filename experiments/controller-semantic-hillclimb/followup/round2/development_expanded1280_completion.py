"""Full256 DEV replay of only the separately screened expanded1280 forests."""
import argparse
import copy
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from threadpoolctl import threadpool_limits

import development_nonlinear_completion as parent
import fit_expanded1280_completion as trained
import development_clustered_completion as retained_cluster
import development_continuous_completion as retained_continuous
import development_math_specialization as retained_math
import expanded_completion_variants as variants

io, original, routed, runtime, nested = parent.io, parent.original, parent.routed, parent.runtime, parent.nested
previous = parent.previous
SCHEMA = 'expanded1280-forest-Complete16-full-development-v1'


def inputs(args):
    spec = io.read(args.previous/'registration.json'); done = io.read(args.previous/'complete.json')
    old_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()}, registration_sha256=io.sha(args.previous/'registration.json'))
    if done['schema'] != parent.SCHEMA or done['registration_sha256'] != old_args.registration_sha256 or done['test_used']:
        raise ValueError('Require completed previous expanded A comparison')
    io.verify({str(args.previous/n): h for n, h in done['artifacts'].items()})
    data, _, values, parent_path, expected = parent.inputs(old_args)
    if expected != spec: raise ValueError('Previous development inputs changed')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    fit_spec = io.read(args.models/'registration.json'); fit_done = io.read(args.models/'complete.json')
    if (fit_spec['schema'] != trained.SCHEMA or fit_done['schema'] != trained.SCHEMA
            or fit_done['registration_sha256'] != io.sha(args.models/'registration.json')
            or fit_done['training_questions'] != 1280 or fit_done['new_neural_calls']
            or fit_done['cloud_calls'] or fit_done['development_or_test_read']): raise ValueError('Require sealed TRAIN-only nonlinear fits')
    io.verify(fit_spec['source_files']); io.verify(fit_spec['input_evidence'])
    io.verify({str(args.models/n): h for n, h in fit_done['digests'].items()})
    models = {}
    if not fit_done['any_screen_passed'] or not fit_done['models']: raise ValueError('No training-screened model to promote')
    for family in fit_done['models']:
        path = args.models/(family+'.pkl')
        if io.sha(path) != fit_done['models'][family]['model_sha256']: raise ValueError('Model changed')
        model = pickle.loads(path.read_bytes())
        if (model.feature_family != family or model.fit_receipt != fit_done['models'][family]['fit_receipt']
                or io.read(Path(model.classifier.directory)/'recipe.json') != data[3]._recipe): raise ValueError('Fit identity changed')
        models[family] = model
        entry = fit_done['clustered_models'][family]
        path = args.models/(family+'-clustered.pkl')
        if io.sha(path) != entry['model_sha256']: raise ValueError('Clustered model changed')
        variant = pickle.loads(path.read_bytes())
        if variant.fit_receipt != entry['fit_receipt']: raise ValueError('Clustered fit identity changed')
        models[family+'-clustered'] = variant
    old = io.read(args.previous/'summary.json')
    extra_spec = io.read(args.retain_a/'registration.json'); extra_done = io.read(args.retain_a/'complete.json')
    if (extra_spec['schema'] != retained_math.SCHEMA
            or extra_done['schema'] != extra_spec['schema']
            or extra_done['registration_sha256'] != io.sha(args.retain_a/'registration.json')
            or extra_spec['dev_questions_sha256'] != spec['dev_questions_sha256']):
        raise ValueError('Require the stronger completed math-specialized A on the identical panel')
    io.verify(extra_spec['source_files']); io.verify(extra_spec['input_evidence'])
    io.verify({str(args.retain_a/n): h for n, h in extra_done['artifacts'].items()})
    extra = io.read(args.retain_a/'summary.json')
    if extra['baselines'] != old['baselines'] or extra['test_used']: raise ValueError('Changed retained A controls/freshness')
    selected = extra['selected_A']
    if selected['policy_module'] != 'continuous_completion_pomdp' or io.sha(selected['model_path']) != selected['model_sha256']:
        raise ValueError('Retained math-specialized full policy changed')
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], **fit_spec['input_evidence'],
            **{str((args.previous/n).resolve()): h for n, h in done['artifacts'].items()},
            **{str((args.models/n).resolve()): h for n, h in fit_done['digests'].items()},
            str((args.previous/'complete.json').resolve()): io.sha(args.previous/'complete.json'),
            str((args.models/'complete.json').resolve()): io.sha(args.models/'complete.json'),
            **{str((args.retain_a/n).resolve()): h for n, h in extra_done['artifacts'].items()},
            str((args.retain_a/'complete.json').resolve()): io.sha(args.retain_a/'complete.json')},
        'paths': {k: str(getattr(args, k).resolve()) for k in ('previous', 'models', 'retain_a', 'output')},
        'previous_A_candidates': [*spec['previous_A_candidates'], old['selected_A'], extra['selected_A']],
        'dev_questions_sha256': spec['dev_questions_sha256'], 'families': list(models),
        'penalties': list(original.PENALTIES), 'posterior_weights': [0., .5],
        'candidate_count': 16*len(models), 'development_questions': 256,
        'training_questions': 1280, 'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False,
        'fresh_validation': False, 'all_original_and_matched_controls_retained': True,
        'same_math_route': 'Retain selected A153 math policy and its exact cached decisions/costs', 'exact_finite_Bellman': True,
        'selection': 'Original and clustered partitions, two posterior weights, seven global penalties plus subject combination; accuracy then compute',
        'matched_controls': 'Recompute full-panel matched controls with the identical retained math route; preserve previous controls',
        'replay_implementation': 'Original parent and retained stronger A must both reproduce exactly'}
    return data, models, values, parent_path, record


def replay(args,data,models,values,parent_path,record):
    rows,trees,parent,old_observer,parent_summary,_,bundle=data
    live=copy.deepcopy(parent);live.math.classifier=live.science.classifier=old_observer
    old_items=[routed.evaluate(r,trees[r['id']],{},live,{},'A') for r in rows]
    if old_items!=io.read(parent_path/'private-candidate-items.json')[parent_summary['completion_A']['key']]:
        raise ValueError('Original decisions/costs changed')
    math={r['id']:v for r,v in zip(rows,old_items) if r['subject']=='math'}
    if original.statistics(list(math.values()))!=parent_summary['math_metrics']:raise ValueError('Original math route changed')
    baselines,matched=previous.controls(rows,trees,math,parent_summary)
    observer=previous.CachedDense(old_observer.directory,old_observer.temperature)
    observer._recipe=old_observer._recipe;observer._retriever=old_observer._retriever;observer._backgrounds=old_observer._backgrounds
    observer.expected={(v['id'],v['path']):v['input_sha256'] for v in bundle['metadata']}
    observer.saved={(v['id'],v['path']):(*[values[k][i] for k in ('state','plain','guided')],int(values['token_slots'][i]))
        for i,v in enumerate(bundle['metadata'])}
    observer.joint={(v['id'],v['path']):values['joint256'][i] for i,v in enumerate(bundle['metadata'])}
    retained_summary=io.read(args.retain_a/'summary.json'); retained=retained_summary['selected_A']
    retained_model=pickle.loads(Path(retained['model_path']).read_bytes()); deploy_math=copy.deepcopy(retained_model.math)
    retained_model.math.classifier=old_observer;retained_model.science.classifier=observer
    retained_items=[retained_math.policy.evaluate(r,trees[r['id']],{},retained_model,{},'A') for r in rows]
    if retained_items!=io.read(args.retain_a/'private-candidate-items.json')[retained['key']]:
        raise ValueError('Retained best A decisions or costs changed')
    math={r['id']:v for r,v in zip(rows,retained_items) if r['subject']=='math'}
    if original.statistics(list(math.values()))!=retained_summary['math_metrics']:raise ValueError('Retained math changed')
    old_matched=matched;matched={}
    for name,prefix,conditional in original.MATCHED:
        items=[math[r['id']] if r['subject']=='math' else original.matched_item(r,trees[r['id']],prefix,conditional) for r in rows]
        matched[name]={'full256_with_retained_math_A':original.statistics(items),
            'science':{s:original.statistics([v for r,v in zip(rows,items) if (r['subject']!='math' if s=='all' else r['subject']==s)])
                for s in ('all','chemistry','physics')}}
        if matched[name]['science']!=old_matched[name]['science']:raise ValueError('Matched science control changed')
    if matched!=retained_summary['matched_complete_controls']:raise ValueError('Retained matched controls changed')
    candidates=[];all_items={}
    for family,model in models.items():
        deploy_model=variants.wrap(model);model=variants.wrap(model);model.classifier=observer
        for weight in (0.,.5):
            items_by_penalty=[]
            for penalty in original.PENALTIES:
                settings={'penalty':penalty,'posterior_weight':weight}
                items_by_penalty.append([math[r['id']] if r['subject']=='math' else variants.evaluate_science(r,trees[r['id']],{},model,settings,'A') for r in rows])
            chosen={s:max(range(7),key=lambda i:original.fit_rank({'dev':original.statistics([v for r,v in zip(rows,items_by_penalty[i]) if r['subject']==s])}))
                for s in ('chemistry','physics')}
            items_by_penalty.append([math[r['id']] if r['subject']=='math' else items_by_penalty[chosen[r['subject']]][i] for i,r in enumerate(rows)])
            for index,items in enumerate(items_by_penalty):
                settings={'penalty':original.PENALTIES[index],'posterior_weight':weight} if index<7 else {'by_subject':{s:{'penalty':original.PENALTIES[i],'posterior_weight':weight} for s,i in chosen.items()}}
                deploy=variants.SubjectPlanner(copy.deepcopy(deploy_math),copy.deepcopy(deploy_model),retained_model.math_settings,settings,
                    {'schema':SCHEMA,'training_model_sha256':io.sha(args.models/(family+'.pkl')),'retained_A_sha256':retained['model_sha256'],'math_unchanged':True})
                name=family+'-w'+str(weight)+'-'+str(index);path=args.output/(name+'.pkl');io.atomic_bytes(path,pickle.dumps(deploy,protocol=5))
                candidate={'key':name,'family':family,'model_path':str(path.resolve()),'model_sha256':io.sha(path),
                    'policy_module':'expanded_completion_variants','settings':{},'math_settings':retained_model.math_settings,
                    'science_settings':settings,'dev':original.statistics(items)}
                candidates.append(candidate);all_items[name]=items
    selected=max([*record['previous_A_candidates'],*candidates],key=original.fit_rank)
    result={'schema':SCHEMA,'selected_A':selected,'candidates':candidates,'previous_A_candidates':record['previous_A_candidates'],
        'baselines':baselines,'matched_complete_controls':matched,'previous_matched_complete_controls':old_matched,
        'parent_exactly_reproduced':True,'retained_A_exactly_reproduced':True,
        'A_above_highest_baseline':selected['dev']['accuracy']-max(v['accuracy'] for v in baselines.values()),
        'development_questions':256,'training_questions':1280,'new_neural_calls':0,'new_cloud_calls':0,
        'test_used':False,'fresh_validation':False,'registration_sha256':args.registration_sha256,'candidate_count':16*len(models),'exact_finite_Bellman':True}
    io.write(args.output/'private-candidate-items.json',all_items);io.write(args.output/'summary.json',result)
    io.write(args.output/'identity.json',{'schema':SCHEMA,'selected_A':selected,'dev_questions_sha256':record['dev_questions_sha256'],
        'source_sha256':io.sha(__file__),'policy_sha256':io.sha(variants.__file__),'policy_module':selected['policy_module'],
        'registration_sha256':args.registration_sha256,'test_used':False})
    io.verify(record['source_files']);io.verify(record['input_evidence'])
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'artifacts':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
        'new_neural_calls':0,'new_cloud_calls':0,'test_used':False})
    print({'selected_A':selected['dev'],'new_candidates':[(v['key'],v['dev']['correct']) for v in candidates]},flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'replay'))
    for n in ('previous', 'models', 'retain_a', 'output'): p.add_argument('--'+n.replace('_', '-'), type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with previous.parent_io.no_test_reads(), original.guarded.no_model_calls(), threadpool_limits(limits=1):
        data, models, values, parent_path, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered development comparison')
        with io.lock(args.output/'.run.lock'):
            previous.preserve_partial(args.output); replay(args, data, models, values, parent_path, record)


if __name__ == '__main__': main()
