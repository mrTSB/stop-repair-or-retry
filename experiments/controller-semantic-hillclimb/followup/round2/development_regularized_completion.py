"""Registered full256 DEV comparison of two nested-regularized Complete POMDPs.

All384 science prefix features are sealed before scoring. No cloud or TEST.
The original A, five baselines and six matched action controls remain intact.
"""
import argparse
import copy
import gc
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import development_completion_blend as parent_io
import completion_subject_pomdp as routed
import fit_regularized_completion as trained
import regularized_completion_pomdp as runtime

io, original, extraction, probe, nested = trained.io, parent_io.original, trained.extraction, trained.probe, trained.nested
SCHEMA = 'regularized-Complete16-full-development-v1'


def inputs(args):
    # Reuse the audited parent/data/feature loader; its blend model is discarded.
    old_args = SimpleNamespace(parent=args.parent, diagnostic=args.parent.parent/'completion-filter-diagnostic-v1', output=args.output)
    rows, trees, parent, unused_blend, observer, summary, receipt = parent_io.load_inputs(old_args)
    spec = io.read(args.models/'registration.json'); done = io.read(args.models/'complete.json')
    if (done['schema'] != trained.SCHEMA or spec['schema'] != trained.SCHEMA
            or done['registration_sha256'] != io.sha(args.models/'registration.json')
            or done['new_neural_calls'] or done['cloud_calls'] or done['development_or_test_read']):
        raise ValueError('Require the separate completed TRAIN-only model fit')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    models = {}
    for family in nested.FAMILIES:
        path = args.models/(family+'.pkl')
        if io.sha(path) != done['models'][family]['model_sha256']: raise ValueError('Fitted policy changed')
        model = pickle.loads(path.read_bytes())
        if model.feature_family != family or model.fit_receipt != done['models'][family]['fit_receipt']:
            raise ValueError('Wrong registered feature policy')
        model.classifier._recipe = io.read(Path(model.classifier.directory)/'recipe.json')
        if model.classifier._recipe != observer._recipe: raise ValueError('Different observation model weights/profile')
        models[family] = model
    science = [r for r in rows if r['subject'] != 'math']; metadata, entries, reference = [], [], []
    for row in science:
        for path in ('', 'repair', 'retry'):
            public = {**row, 'answer': ''}; paid = {p: trees[row['id']][p] for p in ('', path)}
            entry = observer.entry(public, path, paid); value = observer.values(public, path, paid)
            entries.append(entry); reference.append(np.concatenate(value[:3]))
            metadata.append({'id':row['id'], 'path':path, 'input_sha256':nested.fixed.fingerprint(entry)})
    if len(entries) != 384: raise ValueError('All128 science questions and all three prefixes required')
    evidence = {**receipt['input_evidence'], **spec['input_evidence'],
        str((args.models/'registration.json').resolve()):io.sha(args.models/'registration.json'),
        str((args.models/'complete.json').resolve()):io.sha(args.models/'complete.json'),
        **{str((args.models/(f+'.pkl')).resolve()):done['models'][f]['model_sha256'] for f in nested.FAMILIES}}
    registration = {'schema':SCHEMA, 'source_files':io.sources(sys.modules[__name__]), 'input_evidence':evidence,
        'paths':{n:str(getattr(args,n).resolve()) for n in ('parent','models','output')},
        'families':list(nested.FAMILIES), 'penalties':list(original.PENALTIES), 'candidate_count':16,
        'selection':'Seven global penalties plus one subject-specific choice per feature family; accuracy then compute',
        'previous_A_candidates':receipt['previous_A_candidates'], 'A_references':receipt['A_references'],
        'dev_questions_sha256':receipt['dev_questions_sha256'], 'development_questions':256,
        'neural_observations':384, 'new_cloud_calls':0, 'test_used':False, 'fresh_validation':False,
        'entry_sha256':nested.fixed.fingerprint(entries), 'metadata_sha256':nested.fixed.fingerprint(metadata),
        'old_raw20_sha256':nested.fixed.array_sha(np.asarray(reference)), 'parity':probe.PARITY,
        'device':'cpu', 'batch':4, 'chunk':32, 'threads':2, 'memory_guard':[35,25],
        'all_features_before_scoring':True, 'five_original_baselines_and_six_matched_controls_retained':True,
        'same_retained_math_A':True, 'exact_finite_Bellman':True, 'runtime_module':'completion_subject_pomdp'}
    bundle = {'entries':entries,'metadata':metadata,'old_raw20':np.asarray(reference)}
    return (rows,trees,parent,observer,summary,models,bundle), registration


def extract(args, data, record):
    folder=args.output/'features'; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists(): verify_features(args,data,record); return
    if any((folder/n).exists() for n in ('neural.npz','differences.npz','parity.json','metadata.jsonl')):
        raise ValueError('Preserve unsealed aggregate output for inspection')
    model=tokenizer=None
    def guard(): extraction.resource.memory_guard(folder,model is not None)
    def forward(entries):
        nonlocal model,tokenizer
        if model is None:
            torch.set_num_threads(2)
            model,tokenizer,recipe=extraction.network.load(data[3].directory)
            if (recipe != data[3]._recipe or sum(p.numel() for p in model.parameters())!=recipe['parameters']
                    or any(p.device.type!='cpu' for p in model.parameters())):
                raise ValueError('Observation weights/profile/device changed')
            for p in model.parameters():p.requires_grad_(False)
        parts=[probe.capture(model,extraction.network.encode(tokenizer,entries[i:i+4],512)) for i in range(0,len(entries),4)]
        return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    try: values,seals=extraction.extract_arrays(folder,data[-1],args.registration_sha256,forward,guard)
    finally: model=tokenizer=None; gc.collect()
    parity,difference=probe.parity(values,data[-1]['old_raw20'])
    io.npz(folder/'neural.npz',values);io.npz(folder/'differences.npz',{'difference':difference})
    io.write(folder/'parity.json',parity);io.jsonl(folder/'metadata.jsonl',data[-1]['metadata'])
    names=['neural.npz','differences.npz','parity.json','metadata.jsonl','attempts.jsonl','memory.jsonl']
    seals.update({n:io.sha(folder/n) for n in names})
    io.verify(record['source_files']);io.verify(record['input_evidence'])
    io.write(folder/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
        'nodes':384,'physical_batched_forwards':96,'padded_token_slots':384*512,
        'unpadded_token_slots':int(values['token_slots'].sum()),'digests':seals,
        'scored':False,'new_cloud_calls':0,'test_used':False})


def verify_features(args,data,record):
    folder=args.output/'features';done=io.read(folder/'complete.json')
    if (done['schema']!=SCHEMA or done['registration_sha256']!=args.registration_sha256 or done['nodes']!=384
            or done['scored'] or done['test_used'] or done['physical_batched_forwards']!=96
            or done['new_cloud_calls'] or done['padded_token_slots']!=384*512):
        raise ValueError('Require complete ungraded384-observation archive')
    expected={'neural.npz','differences.npz','parity.json','metadata.jsonl','attempts.jsonl','memory.jsonl'}
    expected|={f'chunks/{i:03d}{s}' for i in range(12) for s in ('.json','.npz')}
    if set(done['digests'])!=expected:raise ValueError('Incomplete feature archive manifest')
    io.verify({str(folder/n):v for n,v in done['digests'].items()})
    with np.load(folder/'neural.npz',allow_pickle=False) as archive:values={k:archive[k].copy() for k in archive.files}
    probe.validate(values,384);parity,difference=probe.parity(values,data[-1]['old_raw20'])
    if not parity['all_logits_within_tolerance'] or io.read(folder/'parity.json')!=parity:
        raise ValueError('Registered whole-grid parity failed')
    if io.sha(folder/'metadata.jsonl')!=done['digests']['metadata.jsonl']:raise ValueError('Metadata changed')
    with np.load(folder/'differences.npz',allow_pickle=False) as archive:
        if archive.files!=['difference'] or not np.array_equal(archive['difference'],difference):
            raise ValueError('Parity difference archive changed')
    def forbidden(*args):raise AssertionError('No new forward during archive verification')
    combined,_=extraction.extract_arrays(folder,data[-1],args.registration_sha256,forbidden,forbidden)
    if any(not np.array_equal(values[k],combined[k]) for k in values):
        raise ValueError('Aggregate differs from sealed chunks')
    # The file digest is sealed; independently match the exact registered row order.
    import json
    metadata=[json.loads(s) for s in (folder/'metadata.jsonl').read_text().splitlines()]
    if (metadata!=data[-1]['metadata'] or nested.fixed.fingerprint(metadata)!=record['metadata_sha256']
            or done['unpadded_token_slots']!=int(values['token_slots'].sum())):
        raise ValueError('Changed paid-input alignment or costs')
    return values


class CachedDense(runtime.DenseObserver):
    def load(self): return None
    def values(self,row,path,tree,judge=None):
        if judge is not None:raise ValueError('A cannot acquire review observations')
        public={**row,'answer':''};paid={p:tree[p] for p in ('',path)};key=row['id'],path
        if nested.fixed.fingerprint(self.entry(public,path,paid))!=self.expected[key]:
            raise ValueError('Cached observation accessed under another paid input')
        return self.saved[key]
    def features(self,row,path,tree):
        value=self.values(row,path,tree)
        paid={p:tree[p] for p in ('',path)}
        return np.r_[runtime.core.plain.observables({**row,'answer':''},path,paid,value),self.joint[row['id'],path]]


def controls(rows,trees,math,summary):
    baseline={n:original.statistics([original.fixed(r,trees[r['id']],*s) for r in rows]) for n,s in original.BASELINES.items()}
    matched={}
    for name,prefix,conditional in original.MATCHED:
        values=[math[r['id']] if r['subject']=='math' else original.matched_item(r,trees[r['id']],prefix,conditional) for r in rows]
        matched[name]={'full256_with_retained_math_A':original.statistics(values),
            'science':{s:original.statistics([v for r,v in zip(rows,values) if (r['subject']!='math' if s=='all' else r['subject']==s)])
                for s in ('all','chemistry','physics')}}
    if baseline!=summary['baselines'] or matched!=summary['matched_complete_controls']:
        raise ValueError('Original baseline or matched action control changed')
    return baseline,matched


def preserve_partial(output):
    if any(p.is_file() and p.name not in ('registration.json','.run.lock') for p in output.iterdir()):
        raise ValueError('Preserve partial replay outputs for inspection')


def replay(args,data,record):
    rows,trees,parent,old_observer,summary,models,bundle=data
    values=verify_features(args,data,record)
    with original.guarded.no_model_calls(),threadpool_limits(limits=1):
        live=copy.deepcopy(parent);live.math.classifier=live.science.classifier=old_observer
        original_items=[routed.evaluate(r,trees[r['id']],{},live,{},'A') for r in rows]
        saved=io.read(args.parent/'private-candidate-items.json')[summary['completion_A']['key']]
        if original_items!=saved:raise ValueError('Original full256 paid decisions or charges changed')
        math={r['id']:v for r,v in zip(rows,original_items) if r['subject']=='math'}
        if original.statistics(list(math.values()))!=summary['math_metrics']:raise ValueError('Retained math changed')
        baseline,matched=controls(rows,trees,math,summary)
        observer=CachedDense(old_observer.directory,old_observer.temperature)
        observer._recipe=old_observer._recipe;observer._retriever=old_observer._retriever;observer._backgrounds=old_observer._backgrounds
        observer.expected={(v['id'],v['path']):v['input_sha256'] for v in bundle['metadata']}
        observer.saved={(v['id'],v['path']):(*[values[k][i] for k in ('state','plain','guided')],int(values['token_slots'][i]))
            for i,v in enumerate(bundle['metadata'])}
        observer.joint={(v['id'],v['path']):values['joint256'][i] for i,v in enumerate(bundle['metadata'])}
        candidates=[];all_items={}
        for family,model in models.items():
            deploy_model=copy.deepcopy(model);model.classifier=observer
            family_items=[]
            for penalty in original.PENALTIES:
                items=[math[r['id']] if r['subject']=='math' else runtime.evaluate(r,trees[r['id']],{},model,{'penalty':penalty},'A') for r in rows]
                family_items.append(items)
            chosen={s:max(range(7),key=lambda i:original.fit_rank({'dev':original.statistics([v for r,v in zip(rows,family_items[i]) if r['subject']==s])}))
                for s in ('chemistry','physics')}
            family_items.append([math[r['id']] if r['subject']=='math' else family_items[chosen[r['subject']]][i] for i,r in enumerate(rows)])
            for index,items in enumerate(family_items):
                settings={'penalty':original.PENALTIES[index]} if index<7 else {'by_subject':{s:{'penalty':original.PENALTIES[i]} for s,i in chosen.items()}}
                deploy=routed.Planner(copy.deepcopy(parent.math),copy.deepcopy(deploy_model),parent.math_settings,settings,
                    {'schema':SCHEMA,'training_model_sha256':io.sha(args.models/(family+'.pkl')),'math_unchanged':True})
                name=family+'-'+str(index);path=args.output/(name+'.pkl');io.atomic_bytes(path,pickle.dumps(deploy,protocol=5))
                candidate={'key':name,'family':family,'model_path':str(path.resolve()),'model_sha256':io.sha(path),
                    'policy_module':'completion_subject_pomdp','settings':{},'science_settings':settings,'dev':original.statistics(items)}
                candidates.append(candidate);all_items[name]=items
        selected=max([*record['previous_A_candidates'],*candidates],key=original.fit_rank)
        result={'schema':SCHEMA,'selected_A':selected,'candidates':candidates,'previous_A_candidates':record['previous_A_candidates'],
            'baselines':baseline,'matched_complete_controls':matched,'parent_exactly_reproduced':True,
            'A_above_highest_baseline':selected['dev']['accuracy']-max(v['accuracy'] for v in baseline.values()),
            'development_questions':256,'new_neural_observations':384,'new_cloud_calls':0,'test_used':False,'fresh_validation':False,
            'registration_sha256':args.registration_sha256,'candidate_count':16,'exact_finite_Bellman':True}
        io.write(args.output/'private-candidate-items.json',all_items);io.write(args.output/'summary.json',result)
        io.write(args.output/'identity.json',{'schema':SCHEMA,'selected_A':selected,'dev_questions_sha256':record['dev_questions_sha256'],
            'source_sha256':io.sha(__file__),'policy_sha256':io.sha(routed.__file__),
            'policy_module':'completion_subject_pomdp','registration_sha256':args.registration_sha256,'test_used':False})
        io.verify(record['source_files']);io.verify(record['input_evidence'])
        io.write(args.output/'complete.json',{'schema':SCHEMA,'registration_sha256':args.registration_sha256,
            'artifacts':{p.name:io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
            'features_complete_sha256':io.sha(args.output/'features/complete.json'),'test_used':False})
        print({'selected_A':selected['dev'],'new_candidates':[(v['key'],v['dev']['correct']) for v in candidates]},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','register','extract','replay'))
    for n in ('parent','models','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--registration-sha256');args=p.parse_args()
    with parent_io.no_test_reads():
        with original.guarded.no_model_calls():data,record=inputs(args)
        if args.phase=='plan':print({'nodes':384,'candidates':16,'cloud_calls':0,'test_used':False});return
        if args.phase=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registration_sha256':io.sha(args.output/'registration.json'),'extraction_started':False});return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
                or io.read(args.output/'registration.json')!=record):raise ValueError('Registered comparison changed')
        with io.lock(args.output/'.run.lock'):
            if (args.output/'complete.json').exists():raise ValueError('Preserve completed results')
            if args.phase=='replay':preserve_partial(args.output)
            (extract if args.phase=='extract' else replay)(args,data,record)


if __name__=='__main__':main()
