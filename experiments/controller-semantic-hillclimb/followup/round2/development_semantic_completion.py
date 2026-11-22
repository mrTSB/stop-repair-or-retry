"""Charge and evaluate a training-screened semantic sensor on the full DEV panel."""
import argparse
import copy
import gc
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import development_nonlinear_completion as parent
import development_critic_review_extended as receipts
import fit_semantic_completion as trained
import science_option_observer as option
import semantic_completion_pomdp as policy

io, original, previous, routed = parent.io, parent.original, parent.previous, parent.routed
fingerprint = receipts.diagnostic.inputs.fingerprint
SCHEMA = 'semantic-current-Complete16-full-development-v1'


class CachedSemantic(option.ScienceOptionObserver):
    def load(self):
        if not self._recipe: raise ValueError('Missing sealed recipe')

    def values(self, row, path, tree, judge=None):
        if judge is not None or row['answer'] or set(tree) != {'', path}:
            raise ValueError('Only a blank-reference paid solver prefix is allowed')
        key = (row['id'], path)
        if fingerprint(self.entry(row, path, tree)) != self.expected[key]:
            raise ValueError('Changed sensor input')
        return self.saved[key]


def inputs(args):
    spec = io.read(args.previous/'registration.json'); done = io.read(args.previous/'complete.json')
    old_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    data, _, values, parent_path, expected = parent.inputs(old_args)
    if expected != spec or done['schema'] != parent.SCHEMA or done['registration_sha256'] != io.sha(args.previous/'registration.json'):
        raise ValueError('Changed previous nonlinear comparison')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.previous/n): h for n, h in done['artifacts'].items()})
    fit_spec = io.read(args.models/'registration.json'); fit_done = io.read(args.models/'complete.json')
    if (fit_spec['schema'] != trained.SCHEMA or fit_done['schema'] != trained.SCHEMA
            or fit_done['registration_sha256'] != io.sha(args.models/'registration.json')
            or fit_done['development_or_test_used'] or fit_done['new_neural_calls'] or fit_done['cloud_calls']):
        raise ValueError('Require sealed TRAIN-only sensor fit')
    io.verify(fit_spec['source_files']); io.verify(fit_spec['input_evidence'])
    io.verify({str(args.models/n): h for n, h in fit_done['digests'].items()})
    path = args.models/'semantic.pkl'
    if io.sha(path) != fit_done['model_sha256']: raise ValueError('Model changed')
    model = pickle.loads(path.read_bytes()); rows, trees = data[:2]; plan = []
    for row in rows:
        if row['subject'] == 'math': continue
        for path in ('', 'repair', 'retry'):
            if not trees[row['id']][path]['answer']: continue
            entry = model.semantic.entry({**row, 'answer': ''}, path, {k: trees[row['id']][k] for k in ('', path)})
            plan.append({'id': row['id'], 'kind': 'solver', 'path': path, 'input_sha256': fingerprint(entry)})
    old = io.read(args.previous/'summary.json')
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], **fit_spec['input_evidence'],
            **{str((args.previous/n).resolve()): h for n, h in done['artifacts'].items()},
            **{str((args.models/n).resolve()): h for n, h in fit_done['digests'].items()},
            str((args.previous/'complete.json').resolve()): io.sha(args.previous/'complete.json'),
            str((args.models/'complete.json').resolve()): io.sha(args.models/'complete.json')},
        'paths': {k: str(getattr(args, k).resolve()) for k in ('previous', 'models', 'output')},
        'previous_A_candidates': [*spec['previous_A_candidates'], old['selected_A']],
        'dev_questions_sha256': spec['dev_questions_sha256'], 'penalties': list(original.PENALTIES),
        'candidate_count': 8, 'development_questions': 256, 'training_questions': 768,
        'feature_plan': plan, 'new_neural_calls': len(plan), 'new_cloud_calls': 0,
        'test_used': False, 'fresh_validation': False, 'all_original_and_matched_controls_retained': True,
        'same_math_route': True, 'exact_finite_Bellman': True,
        'selection': 'Seven global science penalties plus subject combination; accuracy then compute',
        'sensor_cost': 'Every valid visited science decision prefix receives one charged V3 forward',
        'feature_execution': 'CPU float32, batch1, actual trimmed input; TRAIN archive used MPS batch8'}
    return data, {'semantic': model}, values, parent_path, record


def extract(args, data, models, record):
    rows, trees = data[:2]; by_id = {r['id']: r for r in rows}
    folder = args.output/'features'; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists(): raise ValueError('Preserve completed features')
    known = receipts.restored(args, record); observer = None
    try:
        for item in record['feature_plan']:
            key = fingerprint(item)
            if key in known: continue
            receipts.diagnostic.extractor.memory_guard(folder, observer is not None)
            if observer is None:
                observer = copy.deepcopy(models['semantic'].semantic); observer.load(); torch.set_num_threads(2)
            row = {**by_id[item['id']], 'answer': ''}; path = item['path']; tree = {k: trees[row['id']][k] for k in ('', path)}
            if fingerprint(observer.entry(row, path, tree)) != item['input_sha256']: raise ValueError('Feature input changed')
            io.append(folder/'attempts.jsonl', {'key': key, 'registration_sha256': args.registration_sha256})
            start = time.perf_counter(); values = observer.values(row, path, tree)
            result = {'key': key, 'outputs': [v.tolist() for v in values[:3]], 'slots': int(values[3]),
                      'elapsed_seconds': time.perf_counter()-start, 'registration_sha256': args.registration_sha256}
            io.append(folder/'observations.jsonl', result); known[key] = result; observer._cache.clear()
            if len(known) % 32 == 0: print({'completed': len(known), 'total': len(record['feature_plan'])}, flush=True)
    finally: observer = None; gc.collect()
    receipts.restored(args, record, True)
    io.write(folder/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'encoder_passes': len(known), 'token_slots': sum(v['slots'] for v in known.values()),
        'digests': {n: io.sha(folder/n) for n in ('attempts.jsonl', 'observations.jsonl', 'memory.jsonl')}, 'new_cloud_calls': 0})


def cached_sensor(args, model, record):
    folder = args.output/'features'; done = io.read(folder/'complete.json')
    if done['registration_sha256'] != args.registration_sha256: raise ValueError('Changed feature completion')
    io.verify({str(folder/n): h for n, h in done['digests'].items()})
    saved = receipts.restored(args, record, True)
    sensor = CachedSemantic(model.semantic.directory, model.semantic.temperature)
    sensor._recipe = model.semantic._recipe; sensor.expected = {}; sensor.saved = {}
    for item in record['feature_plan']:
        key = (item['id'], item['path']); value = saved[fingerprint(item)]
        sensor.expected[key] = item['input_sha256']
        sensor.saved[key] = (*[np.asarray(v) for v in value['outputs']], value['slots'])
    return sensor


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'extract', 'replay'))
    for n in ('previous', 'models', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with previous.parent_io.no_test_reads():
        with original.guarded.no_model_calls(), threadpool_limits(limits=1): data, models, values, parent_path, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json'), 'sensor_passes': len(record['feature_plan'])}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered sensor comparison')
        with io.lock(args.output/'.run.lock'):
            if args.phase == 'extract': extract(args, data, models, record)
            else:
                if (args.output/'complete.json').exists(): raise ValueError('Preserve completed replay')
                with original.guarded.no_model_calls(), threadpool_limits(limits=1): replay(args, data, models, values, parent_path, record)



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
    candidates=[];all_items={}
    for family,model in models.items():
        deploy_model=copy.deepcopy(model);model.classifier=observer;model.semantic=cached_sensor(args,model,record);items_by_penalty=[]
        for penalty in original.PENALTIES:
            items_by_penalty.append([math[r['id']] if r['subject']=='math' else policy.evaluate_science(r,trees[r['id']],{},model,{'penalty':penalty},'A') for r in rows])
        chosen={s:max(range(7),key=lambda i:original.fit_rank({'dev':original.statistics([v for r,v in zip(rows,items_by_penalty[i]) if r['subject']==s])}))
            for s in ('chemistry','physics')}
        items_by_penalty.append([math[r['id']] if r['subject']=='math' else items_by_penalty[chosen[r['subject']]][i] for i,r in enumerate(rows)])
        for index,items in enumerate(items_by_penalty):
            settings={'penalty':original.PENALTIES[index]} if index<7 else {'by_subject':{s:{'penalty':original.PENALTIES[i]} for s,i in chosen.items()}}
            deploy=policy.SubjectPlanner(copy.deepcopy(parent.math),copy.deepcopy(deploy_model),parent.math_settings,settings,
                {'schema':SCHEMA,'training_model_sha256':io.sha(args.models/(family+'.pkl')),'math_unchanged':True})
            name=family+'-'+str(index);path=args.output/(name+'.pkl');io.atomic_bytes(path,pickle.dumps(deploy,protocol=5))
            candidate={'key':name,'family':family,'model_path':str(path.resolve()),'model_sha256':io.sha(path),
                'policy_module':'semantic_completion_pomdp','settings':{},'science_settings':settings,'dev':original.statistics(items)}
            candidates.append(candidate);all_items[name]=items
    selected=max([*record['previous_A_candidates'],*candidates],key=original.fit_rank)
    result={'schema':SCHEMA,'selected_A':selected,'candidates':candidates,'previous_A_candidates':record['previous_A_candidates'],
        'baselines':baselines,'matched_complete_controls':matched,'parent_exactly_reproduced':True,
        'A_above_highest_baseline':selected['dev']['accuracy']-max(v['accuracy'] for v in baselines.values()),
        'development_questions':256,'training_questions':768,'new_neural_calls':record['new_neural_calls'],'new_cloud_calls':0,
        'test_used':False,'fresh_validation':False,'registration_sha256':args.registration_sha256,'candidate_count':8,'exact_finite_Bellman':True}
    io.write(args.output/'private-candidate-items.json',all_items);io.write(args.output/'summary.json',result)
    io.write(args.output/'identity.json',{'schema':SCHEMA,'selected_A':selected,'dev_questions_sha256':record['dev_questions_sha256'],
        'source_sha256':io.sha(__file__),'policy_sha256':io.sha(policy.__file__),'policy_module':'semantic_completion_pomdp',
        'registration_sha256':args.registration_sha256,'test_used':False})
    io.verify(record['source_files']);io.verify(record['input_evidence'])
    io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'artifacts':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
        'new_neural_calls':record['new_neural_calls'],'new_cloud_calls':0,'test_used':False})
    print({'selected_A':selected['dev'],'new_candidates':[(v['key'],v['dev']['correct']) for v in candidates]},flush=True)



if __name__ == '__main__': main()
