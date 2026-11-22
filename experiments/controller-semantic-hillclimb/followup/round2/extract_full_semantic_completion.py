"""Prepare guarded CPU768 same-forward dense observations and fixed TRAIN CV.

Registration, extraction and diagnostic fitting are separate explicit phases.
This helper never acquires an LLM response, trains an encoder, selects a policy,
or opens DEV/TEST question records. All current sources/artifacts stay intact.
"""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import full_semantic_completion_probe as probe
import completion_v4_diagnostic as fixed
import fit_completion_pomdp as fitted
import expanded_mentor_io as io
import extract_expanded_mentor as resource
import network_large as network

SCHEMA='TRAIN-science-same-forward-full1024-extraction-v1'
CHUNK,BATCH=32,4


def training_bundle(base_fit):
    """Reconstruct the completed fitter's TRAIN bytes, avoiding its DEV gate."""
    fitted.verify_complete(base_fit);registered=io.read(base_fit/'registration.json')
    args=SimpleNamespace(**{k:Path(v) for k,v in registered['paths'].items()})
    import science_complete_actions as actions
    all_rows,plain=fitted.plain_data(args.data,'train');rows=[r for r in all_rows if r['subject']!='math']
    public=[actions.pilot.protocol.public_row(r) for r in rows]
    trees={r['id']:plain[r['id']].copy() for r in rows}
    grid=io.read(args.complete/'registration.json');sealed=io.read(args.complete/'complete.json')
    if (grid['selection']!=[{'id':r['id'],'subject':r['subject']} for r in rows]
        or grid['paths']['data']!=str(args.data.resolve()) or grid['paths']['output']!=str(args.complete.resolve())
        or registered['complete_grid_receipt']['complete_sha256']!=io.sha(args.complete/'complete.json')):
        raise ValueError('Completed fitter and exact TRAIN action grid differ')
    io.verify({str(args.complete/n):v for n,v in sealed['digests'].items()})
    reused=actions.pilot.protocol.strict_records(Path(grid['paths']['pilot'])/'raw-generations.jsonl')
    if len(reused)!=192 or io.sha(Path(grid['paths']['pilot'])/'raw-generations.jsonl')!=grid['raw_reuse']['sha256']:
        raise ValueError('Changed reused TRAIN Complete observations')
    known,coverage=actions.restore(public,trees,{r['request_sha256'] for r in reused},
        actions.pilot.protocol.strict_records(args.complete/'train.jsonl'),
        actions.pilot.protocol.strict_records(args.complete/'raw-generations.jsonl'),True)
    if coverage!=sealed['coverage']:raise ValueError('Changed sealed768 TRAIN Complete grid')
    for row in rows:
        for prefix in actions.PREFIXES:trees[row['id']][actions.action_path(prefix)]=known[row['id'],prefix]['response']
    observer=fitted.cached.observer_from(args.base)
    legacy_ids={json.loads(line)['id'] for line in args.legacy_examples.read_bytes().splitlines()}
    if legacy_ids & {r['id'] for r in all_rows}:raise ValueError('Old encoder saw a new training question')
    if observer._recipe['data_sha256']!=io.sha(args.legacy_examples):raise ValueError('Old observer training corpus changed')
    old_metadata=[json.loads(line) for line in (args.base/'training-neural.jsonl').read_bytes().splitlines()]
    with np.load(args.base/'training-neural.npz',allow_pickle=False) as values:
        old,alignment=fitted.cached.verify_features(all_rows,plain,observer,old_metadata,values)
    if fixed.fingerprint(alignment)!=registered['original_neural_alignment_sha256']:
        raise ValueError('Old neural paid-input alignment changed')
    metadata,entries,reference=[],[],[]
    for row in rows:
        for path in ('','repair','retry'):
            paid={p:trees[row['id']][p] for p in ('',path)}
            entry=observer.entry({**row,'answer':''},path,paid)
            entries.append(entry);reference.append(np.concatenate(old[row['id'],path][:3]))
            metadata.append({'id':row['id'],'subject':row['subject'],'path':path,
                'prompt_sha256':paid[path]['prompt_sha256'],'input_sha256':fixed.fingerprint(entry)})
    if len(rows)!=256 or len(entries)!=768 or len({r['id'] for r in rows})!=256:
        raise ValueError('Use all256 TRAIN science questions and768 prefixes')
    recipe=observer._recipe
    if recipe['max_length']!=512 or recipe['fold']!='full' or recipe['input_profile']!='full-options-openstax-v1':
        raise ValueError('Require unchanged old full512 observer profile')
    assets={str(p.resolve()):io.sha(p) for p in Path(observer.directory).iterdir() if p.is_file()}
    evidence={**registered['input_evidence'],**assets,
        str((base_fit/'registration.json').resolve()):io.sha(base_fit/'registration.json'),
        str((base_fit/'complete.json').resolve()):io.sha(base_fit/'complete.json')}
    return {'rows':rows,'trees':trees,'observer':observer,'recipe':recipe,'metadata':metadata,'entries':entries,
        'old_raw20':np.asarray(reference),'evidence':evidence,'old_alignment_sha256':fixed.fingerprint(alignment)}


def specification(args,bundle):
    return {'schema':SCHEMA,'profile':probe.PROFILE,'source_files':io.sources(sys.modules[__name__]),
        'input_evidence':bundle['evidence'],'base_fit':str(args.base_fit.resolve()),'output':str(args.output.resolve()),
        'metadata_sha256':fixed.fingerprint(bundle['metadata']),'input_sha256':fixed.fingerprint(bundle['entries']),
        'old_raw20_sha256':fixed.array_sha(bundle['old_raw20']),'old_alignment_sha256':bundle['old_alignment_sha256'],
        'encoder_directory':str(Path(bundle['observer'].directory).resolve()),
        'parameters':bundle['recipe']['parameters'],'encoder_max_length':512,'device':'cpu','threads':2,
        'questions':256,'nodes':768,'paths':['','repair','retry'],'batch_size':BATCH,'chunk_size':CHUNK,
        'memory_guard':{'startup_free_percent':35,'continue_free_percent':25,'one_encoder':True},
        'same_forward':'Capture existing Network.norm1024 and project256 outputs through detached hooks in one eval/inference forward',
        'parity':probe.PARITY,'features':{'control':386,'augmented':1410,'additional_dense':1024},
        'diagnostic':{'separate_registration_required':True,'temperature':1.,'folds':3,'scaler':'Fit only on training questions in each fold',
            'row_weight':1.,'question_mass':3.,'bootstrap_draws':fixed.BOOTSTRAPS,'bootstrap_seed':fixed.SEED,
            'screen':'Lower joint_log_loss AND lower mean_four_marginal_Brier; CI descriptive'},
        'no_encoder_training':True,'extra_online_encoder_passes':0,'no_online_adapter_promotion':True,
        'online_head_CPU_time':'Future runtime must separately report added projection-vector/logistic-head CPU overhead',
        'text_uploaded':False,'cloud_calls':0,'development_or_test_read':False,
        'environment':{'python':list(sys.version_info[:3]),'packages':{n:importlib.metadata.version(n)
            for n in ('torch','transformers','safetensors','numpy','scikit-learn')}}}


def chunk_identity(bundle,start,registration):
    if type(start) is not int or start<0 or start>=len(bundle['entries']) or start%CHUNK:
        raise ValueError('Out-of-grid dense chunk attempt')
    return {'registration_sha256':registration,'start':start,'count':min(CHUNK,len(bundle['entries'])-start),
        'metadata_sha256':fixed.fingerprint(bundle['metadata'][start:start+CHUNK]),
        'input_sha256':fixed.fingerprint(bundle['entries'][start:start+CHUNK])}


def extract_arrays(output,bundle,registration,forward,guard):
    chunks=output/'chunks';chunks.mkdir(exist_ok=True)
    if list(output.glob('*.tmp')) or list(chunks.glob('*.tmp')):raise ValueError('Preserve temporary output; no implicit re-forward')
    allowed={f'{i//CHUNK:03d}{suffix}' for i in range(0,len(bundle['entries']),CHUNK) for suffix in ('.npz','.json')}
    if any(p.name not in allowed for p in chunks.iterdir()):raise ValueError('Unexpected dense chunk output')
    attempts={}
    if (output/'attempts.jsonl').exists():
        raw=(output/'attempts.jsonl').read_bytes()
        if raw and not raw.endswith(b'\n'):raise ValueError('Partial attempted-forward journal')
        for line in raw.splitlines():
            record=json.loads(line);start=record['identity']['start']
            if (start in attempts or record['identity']!=chunk_identity(bundle,start,registration)
                or record['attempted_rows']!=record['identity']['count']):raise ValueError('Changed/duplicate forward attempt')
            attempts[start]=record
    pieces=[];seals={}
    for start in range(0,len(bundle['entries']),CHUNK):
        identity=chunk_identity(bundle,start,registration);path=chunks/f'{start//CHUNK:03d}.npz';receipt_path=path.with_suffix('.json')
        if path.exists() or receipt_path.exists():
            if not path.exists() or not receipt_path.exists() or start not in attempts:raise ValueError('Unsealed/orphan chunk requires inspection')
            receipt=io.read(receipt_path)
            if receipt['identity']!=identity or receipt['arrays_sha256']!=io.sha(path):raise ValueError('Changed dense chunk identity')
            with np.load(path,allow_pickle=False) as archive:values={n:archive[n].copy() for n in archive.files if n!='raw20_difference'}
        else:
            if start in attempts:raise ValueError('Earlier interrupted forward requires inspection; no automatic repeat')
            guard();io.append(output/'attempts.jsonl',{'identity':identity,'attempted_rows':identity['count']})
            values=forward(bundle['entries'][start:start+CHUNK]);probe.validate(values,identity['count'])
            check,difference=probe.parity(values,bundle['old_raw20'][start:start+CHUNK])
            io.npz(path,{**values,'raw20_difference':difference})
            receipt={'identity':identity,'arrays_sha256':io.sha(path),'parity':check,
                'physical_padded_token_slots':identity['count']*512,'actual_unpadded_token_slots':int(values['token_slots'].sum()),
                'extra_encoder_passes_for_project256':0}
            io.write(receipt_path,receipt)
        probe.validate(values,identity['count']);check,difference=probe.parity(values,bundle['old_raw20'][start:start+CHUNK])
        if (receipt['parity']!=check or receipt['physical_padded_token_slots']!=identity['count']*512
            or receipt['actual_unpadded_token_slots']!=int(values['token_slots'].sum())
            or receipt['extra_encoder_passes_for_project256']!=0):raise ValueError('Changed dense chunk semantics')
        with np.load(path,allow_pickle=False) as archive:
            if not np.array_equal(archive['raw20_difference'],difference):raise ValueError('Changed full parity differences')
        if not check['all_logits_within_tolerance']:raise ValueError('Raw20 parity failed; preserve every row and do not change tolerance')
        pieces.append(values);seals[str(path.relative_to(output))]=io.sha(path);seals[str(receipt_path.relative_to(output))]=io.sha(receipt_path)
        print({'completed_rows':start+identity['count'],'total':len(bundle['entries'])},flush=True)
    values={n:np.concatenate([p[n] for p in pieces]) for n in pieces[0]};probe.validate(values,len(bundle['entries']))
    return values,seals


def verify_complete(output,registration,bundle):
    complete=io.read(output/'neural-complete.json')
    registered=io.read(output/'registration.json')
    expected={'registration.json','attempts.jsonl','memory.jsonl','dense-neural.npz',
        'raw20-parity-differences.npz','metadata.jsonl','parity.json'}
    expected|={f'chunks/{i//CHUNK:03d}{suffix}' for i in range(0,768,CHUNK) for suffix in ('.npz','.json')}
    if (complete['schema']!=SCHEMA or complete['registration_sha256']!=registration or complete['nodes']!=768
        or io.sha(output/'registration.json')!=registration or set(complete['digests'])!=expected
        or registered['schema']!=SCHEMA or registered['output']!=str(output.resolve())
        or registered['device']!='cpu' or registered['encoder_max_length']!=512
        or registered['batch_size']!=BATCH or registered['chunk_size']!=CHUNK):
        raise ValueError('Changed dense completion contract')
    io.verify(registered['source_files']);io.verify(registered['input_evidence'])
    io.verify({str(output/n):v for n,v in complete['digests'].items()})
    with np.load(output/'dense-neural.npz',allow_pickle=False) as archive:values={n:archive[n].copy() for n in archive.files}
    probe.validate(values,768)
    def forbidden(*args):raise AssertionError('Completed extraction cannot acquire a missing neural chunk')
    combined,_=extract_arrays(output,bundle,registration,forbidden,forbidden)
    if any(not np.array_equal(values[n],combined[n]) for n in values):raise ValueError('Aggregate differs from sealed dense chunks')
    metadata=[json.loads(line) for line in (output/'metadata.jsonl').read_bytes().splitlines()]
    check,difference=probe.parity(values,bundle['old_raw20'])
    if (metadata!=bundle['metadata'] or registered['metadata_sha256']!=fixed.fingerprint(metadata)
        or registered['input_sha256']!=fixed.fingerprint(bundle['entries'])
        or registered['old_raw20_sha256']!=fixed.array_sha(bundle['old_raw20'])
        or io.read(output/'parity.json')!=check or not check['all_logits_within_tolerance']):
        raise ValueError('Changed dense completed input/parity alignment')
    with np.load(output/'raw20-parity-differences.npz',allow_pickle=False) as archive:
        if archive.files!=['difference'] or not np.array_equal(archive['difference'],difference):raise ValueError('Changed complete parity differences')
    semantic={'questions':256,'neural_question_passes':768,'physical_batched_forwards':192,
        'physical_padded_token_slots':768*512,'actual_unpadded_token_slots':int(values['token_slots'].sum()),
        'parameters':registered['parameters'],'offline_parameter_token_proxy':registered['parameters']/1e9*768*512/3.09,
        'extra_project_encoder_passes':0,'encoder_training_steps':0,'cloud_calls':0,'development_or_test_read':False}
    if any(complete[k]!=v for k,v in semantic.items()):raise ValueError('Changed dense completed cost/scope semantics')
    return values,complete


def extract(args,bundle,record):
    with io.lock(args.output/'.extract.lock'):
        if (args.output/'neural-complete.json').exists():verify_complete(args.output,args.registration_sha256,bundle);return
        model=tokenizer=None
        def guard():resource.memory_guard(args.output,model is not None)
        def forward(entries):
            nonlocal model,tokenizer
            if model is None:
                torch.set_num_threads(2);model,tokenizer,recipe=network.load(record['encoder_directory']);model.eval()
                for parameter in model.parameters():parameter.requires_grad_(False)
                if (recipe!=bundle['recipe'] or sum(p.numel() for p in model.parameters())!=record['parameters']
                    or any(p.device.type!='cpu' for p in model.parameters())):raise ValueError('Loaded observer recipe/device/parameters changed')
            parts=[]
            for start in range(0,len(entries),BATCH):
                tensors=network.encode(tokenizer,entries[start:start+BATCH],512)
                parts.append(probe.capture(model,tensors));del tensors
            return {n:np.concatenate([p[n] for p in parts]) for n in parts[0]}
        try:values,seals=extract_arrays(args.output,bundle,args.registration_sha256,forward,guard)
        finally:model=tokenizer=None;gc.collect()
        check,difference=probe.parity(values,bundle['old_raw20'])
        io.npz(args.output/'dense-neural.npz',values);io.npz(args.output/'raw20-parity-differences.npz',{'difference':difference})
        io.jsonl(args.output/'metadata.jsonl',bundle['metadata']);io.write(args.output/'parity.json',check)
        names=('registration.json','attempts.jsonl','memory.jsonl','dense-neural.npz','raw20-parity-differences.npz','metadata.jsonl','parity.json')
        seals.update({n:io.sha(args.output/n) for n in names});io.verify(record['source_files']);io.verify(record['input_evidence'])
        complete={'schema':SCHEMA,'registration_sha256':args.registration_sha256,'nodes':768,'questions':256,'digests':seals,
            'neural_question_passes':768,'physical_batched_forwards':192,'physical_padded_token_slots':768*512,
            'actual_unpadded_token_slots':int(values['token_slots'].sum()),'parameters':record['parameters'],
            'offline_parameter_token_proxy':record['parameters']/1e9*768*512/3.09,
            'runtime_per_observation_cost':'Same frozen encoder parameters times actual unpadded slots; no extra encoder forward for joint256',
            'extra_project_encoder_passes':0,'encoder_training_steps':0,'cloud_calls':0,'development_or_test_read':False}
        io.write(args.output/'neural-complete.json',complete)
        verify_complete(args.output,args.registration_sha256,bundle)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('register','extract'))
    for name in ('base-fit','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--registration-sha256');parser.add_argument('--diagnostic-output',type=Path);args=parser.parse_args()
    with fixed.training_only_reads():
        with fixed.guarded.no_model_calls():bundle=training_bundle(args.base_fit)
        record=specification(args,bundle)
        if args.mode=='register':
            args.output.mkdir(parents=True,exist_ok=False);io.write(args.output/'registration.json',record)
            print({'registered':True,'registration_sha256':io.sha(args.output/'registration.json'),'extracted':False},flush=True);return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json')!=args.registration_sha256
            or io.read(args.output/'registration.json')!=record):raise ValueError('Changed explicit dense extraction registration')
        io.verify(record['source_files']);io.verify(record['input_evidence'])
        extract(args,bundle,record)


if __name__=='__main__':main()
