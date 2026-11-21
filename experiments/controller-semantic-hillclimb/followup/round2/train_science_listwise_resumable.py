"""Same grouped science recipe, with exact recovery and bounded archive copies."""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

import train_science_listwise as original
import science_training_checkpoint as checkpoint

sha, write = original.sha, checkpoint.atomic_json
OPERATIONS = {'checkpoint_every_steps':100, 'checkpoint_generations_retained':2,
              'checkpoint_tensor_cpu_copy_bytes':checkpoint.CHUNK_BYTES,
              'telemetry_every_steps':10, 'pause_below_system_free_percent':20,
              'cleanup':'Delete batch/graph references and clear gradients after each completed update; empty unused MPS cache after recovery checkpoints.',
              'training_recipe_changed':False, 'batch_shapes_changed':False,
              'deployment_selection':'Final epoch3 at2160completed updates only; recovery checkpoints are never model-selection candidates'}


def source_files():
    return {str(Path(module.__file__).resolve()):sha(module.__file__) for module in (
        original, original.data, original.network, original.original, checkpoint)} | {str(Path(__file__).resolve()):sha(__file__)}


def verify_files(files):
    for path, expected in files.items():
        if sha(path)!=expected:raise ValueError('Registered source/input changed: '+str(path))


def memory_snapshot(device='cpu'):
    result={'time_unix':time.time(),'pid':os.getpid()}
    for name,command in [('vm_stat',['vm_stat']),('swap',['sysctl','vm.swapusage']),
                         ('memory_pressure',['memory_pressure','-Q']),('rss_kib',['ps','-p',str(os.getpid()),'-o','rss='])]:
        try:
            value=subprocess.run(command,capture_output=True,text=True,timeout=5,check=True).stdout.strip()
            result[name]=value
        except (OSError,subprocess.SubprocessError) as error:
            result[name+'_unavailable']=type(error).__name__
    match=re.search(r'System-wide memory free percentage:\s*(\d+)%',result.get('memory_pressure',''))
    result['system_free_percent']=int(match[1]) if match else None
    if device=='mps':
        result['mps_tensor_bytes']=torch.mps.current_allocated_memory()
        result['mps_driver_bytes']=torch.mps.driver_allocated_memory()
        result['mps_recommended_max_bytes']=torch.mps.recommended_max_memory()
    return result


def append(path,value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


@contextmanager
def stop_at_boundary():
    state={'requested':False,'signal':None}; previous={}
    def stop(signum,frame):
        if state['requested']:
            # A second interrupt can abort an unresponsive operation; the last
            # atomic checkpoint remains valid and partial files stay preserved.
            raise KeyboardInterrupt('Second stop request before safe boundary')
        state.update(requested=True,signal=signum)
    for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,stop)
    try:yield state
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def next_position(step,batches,grouped):
    epoch,batch=divmod(step,batches)
    order=np.random.default_rng(original.SEED+epoch).permutation(len(grouped))
    selected=[grouped[i] for i in order[2*batch:2*batch+2]]
    return {'step':step,'epoch':epoch,'batch_in_epoch':batch,
            'next_group_indexes_sha256':hashlib.sha256(json.dumps(selected).encode()).hexdigest()}


def prepare(args):
    initial=original.identity(args)
    rows=original.load_registered(args.metadata,args.source,'train')
    samples=original.data.examples(rows,initial['recipe']['numeric_width']);grouped=original.groups(samples)
    steps=original.RECIPE['epochs']*math.ceil(len(grouped)/original.RECIPE['questions_per_batch'])
    if (len(rows),steps)!=(1440,2160):raise ValueError('Keep the exact1440question2160update recipe')
    payload=''.join(json.dumps(row)+'\n' for row in samples).encode()
    registration={**original.RECIPE,'schema':'science-grouped-resumable-v3','operations':OPERATIONS,
        'training_questions':len(rows),'training_options':len(samples),'steps':steps,
        'warmup_steps':max(1,steps//10),'lr_schedule':'Linear warmup, then linear decay to zero',
        'selected_training_ids':[row['id'] for row in rows], 'initial_directory':str(args.initial.resolve()),
        'initial_recipe_sha256':initial['initial_files']['recipe.json'],
        'initial_completion_sha256':initial['initial_completion_sha256'],
        'initial_training_registration_sha256':initial['initial_training_registration_sha256'],
        'initial_files':initial['initial_files'],'initial_weights':initial['recipe']['weights'],
        'auxiliary_metadata_sha256':initial['metadata_sha256'],'source_sha256':initial['source_sha256'],
        'source_revision':initial['source_revision'],'protected_split_sha256':initial['protected_split_sha256'],
        'seal_receipt_sha256':initial['seal_receipt_sha256'],'historical_exclusions_sha256':initial['historical_exclusions_sha256'],
        'training_inputs_sha256':hashlib.sha256(payload).hexdigest(),'source_files':source_files(),
        'device':args.device,'local_only':True,'fold':'full','mode':'train',
        'initialization':'Restart from immutable completedv1; no v2 learned state survived its resource abort.',
        'auxiliary_calibration_authorized_after_final_archive':True,
        'no_project_development_test_question_response_values_read':True,
        'runtime':{name:original.version(name) for name in ('torch','transformers','numpy','safetensors')}}
    path=args.output/'training-registration.json'
    if path.exists():
        if json.loads(path.read_bytes())!=registration:raise ValueError('Changed resume registration')
        if sha(args.output/'training-inputs.jsonl')!=registration['training_inputs_sha256']:raise ValueError('Changed training inputs')
    else:
        if any(p.name!='.training.lock' for p in args.output.iterdir()):raise ValueError('Unregistered output requires inspection')
        (args.output/'training-inputs.jsonl').write_bytes(payload);write(path,registration)
    return initial,rows,samples,grouped,registration


def release_guard(args):
    if args.device!='mps':return
    if args.release is None:raise ValueError('MPS execution requires the serialized waiters release receipt')
    receipt=json.loads(args.release.read_bytes())
    if receipt['output']!=str(args.output.resolve()) or receipt['trainer_sha256']!=sha(__file__):raise ValueError('Changed scheduler release')
    verify_files(receipt['prerequisites'])
    try:os.kill(receipt['protected_v4_pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('Protected V4 process still exists; never overlap training')
    memory=memory_snapshot('cpu')
    if memory['system_free_percent'] is None or memory['system_free_percent']<35:
        raise ValueError('Wait for at least35%system memory headroom before MPS startup')


def final_calibration(args,registration,samples,checkpoint_receipt):
    full=args.output/'full';recipe=json.loads((full/'recipe.json').read_bytes())
    if recipe['exact_training_checkpoint_sha256']!=checkpoint_receipt['manifest_sha256']:
        raise ValueError('Deployment archive does not descend from the final recovery state')
    verify_files({str(full/name):digest for name,digest in recipe['weights'].items()})
    cache=args.output/'calibration-margins.npz';cal_receipt=args.output/'calibration-inference.json'
    if cal_receipt.exists():
        saved=json.loads(cal_receipt.read_bytes())
        if saved['weights']!=recipe['weights'] or saved['registration_sha256']!=sha(args.output/'training-registration.json') or saved['logits_sha256']!=sha(cache):
            raise ValueError('Changed completed auxiliary-calibration inference')
        with np.load(cache) as values:logits=values['margin']
        cal_rows=original.load_registered(args.metadata,args.source,'calibration')
        calibration=original.data.examples(cal_rows,recipe['numeric_width'])
    else:
        if cache.exists():raise ValueError('Unsealed calibration cache requires inspection')
        archived,tokenizer,reloaded=original.network.load(full)
        if reloaded['weights']!=recipe['weights']:raise ValueError('Final weight reload mismatch')
        archived=archived.to(args.device)
        cal_rows=original.load_registered(args.metadata,args.source,'calibration')
        calibration=original.data.examples(cal_rows,recipe['numeric_width'])
        tensors=original.network.encode(tokenizer,calibration,original.RECIPE['max_length'])
        logits=original.original.margin(torch.from_numpy(original.network.predict(archived,tensors,args.device,batch=8)[0])).numpy()
        temporary=cache.with_name('calibration-margins.tmp.npz');np.savez_compressed(temporary,margin=logits);temporary.replace(cache)
        write(cal_receipt,{'weights':recipe['weights'],'registration_sha256':sha(args.output/'training-registration.json'),
                          'logits_sha256':sha(cache),'question_count':160,'selected_checkpoint':'Final2160updates only'})
        del archived,tensors;gc.collect()
        if args.device=='mps':torch.mps.empty_cache()
    search=[original.scores(logits,calibration,t) for t in original.TEMPERATURES]
    chosen=min(search,key=lambda r:(r['balanced_bce'],r['temperature']))
    precalibration=recipe.get('precalibration_recipe_sha256',sha(full/'recipe.json'))
    recipe.update(temperature=chosen['temperature'],calibration_weight_sha256=recipe['weights'],precalibration_recipe_sha256=precalibration)
    write(full/'recipe.json',recipe)
    summary={'registration_sha256':sha(args.output/'training-registration.json'),'mode':'train',
        'training_steps':2160,'training_questions':1440,'training_options':len(samples),
        'history_sha256':sha(args.output/'optimization-history.jsonl'),
        'training_encoded_sha256':sha(args.output/'training-encoded.npz'),
        'all_encoder_head_gradients_finite_nonzero':True,'initial_checkpoint_unchanged':True,
        'project_dev_or_test_used':False,'cloud_calls':0,'calibration':chosen,'temperature_search':search,
        'temperature_selected_on_auxiliary_calibration':chosen['temperature'],'recipe_sha256':sha(full/'recipe.json'),
        'parameters':recipe['parameters'],'calibration_weight_sha256':recipe['weights'],
        'calibration_used_reloaded_archive':True,'calibration_margins_sha256':sha(cache),
        'checkpoint_selection':'Final epoch3 only','exact_training_checkpoint':checkpoint_receipt,
        'resumable_training':True,'operational_recipe':OPERATIONS}
    verify_files(registration['source_files'])
    verify_files({str(args.initial/name):digest for name,digest in registration['initial_files'].items()})
    write(args.output/'summary.json',summary);write(args.output/'training-summary.json',summary)
    print(json.dumps({'complete':True,'steps':2160,'calibration':chosen}),flush=True)


def execute(args,initial,samples,grouped,registration):
    release_guard(args)
    if args.device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    sources=registration['source_files'];history=args.output/'optimization-history.jsonl'
    binding={'registration_sha256':sha(args.output/'training-registration.json'),'training_inputs_sha256':registration['training_inputs_sha256'],
             'planned_steps':2160,'seed':original.SEED,'batch_questions':2}
    checkpoints=args.output/'recovery';batches=math.ceil(len(grouped)/2)
    if (args.output/'training-summary.json').exists():
        summary=json.loads((args.output/'training-summary.json').read_bytes())
        if summary['registration_sha256']!=binding['registration_sha256'] or summary['recipe_sha256']!=sha(args.output/'full/recipe.json'):
            raise ValueError('Completed training artifacts changed')
        recipe=json.loads((args.output/'full/recipe.json').read_bytes())
        verify_files({str(args.output/'full'/name):digest for name,digest in recipe['weights'].items()})
        print('Verified completed v3 training; no work repeated.',flush=True);return
    if (args.output/'full').exists():
        directory,manifest=checkpoint.load_manifest(checkpoints,binding,args.device)
        if manifest['progress']!=next_position(2160,batches,grouped):raise ValueError('Final archive before complete optimization')
        final_calibration(args,registration,samples,{'manifest_sha256':sha(directory/'manifest.json'),'step':2160})
        return
    copied=args.output/'initial-checkpoint-copy'
    if not copied.exists():shutil.copytree(args.initial,copied)
    verify_files({str(copied/name):digest for name,digest in initial['initial_files'].items()})
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(original.SEED)
    if args.device=='mps':torch.mps.manual_seed(original.SEED)
    np.random.seed(original.SEED);random.seed(original.SEED)
    model,tokenizer,_=original.network.load(copied);model=model.to(args.device)
    encoder,heads=original.configure(model)
    optimizer=torch.optim.AdamW([{'params':encoder,'lr':original.RECIPE['encoder_lr']},
        {'params':heads,'lr':original.RECIPE['head_lr']}],weight_decay=original.RECIPE['weight_decay'],foreach=False)
    scheduler=get_linear_schedule_with_warmup(optimizer,216,2160)
    encoded=args.output/'training-encoded.npz';encoded_receipt=args.output/'training-encoded.json'
    if encoded.exists():
        if (not encoded_receipt.exists() or json.loads(encoded_receipt.read_bytes())['sha256']!=sha(encoded)
                or json.loads(encoded_receipt.read_bytes()).get('training_inputs_sha256')!=registration['training_inputs_sha256']):
            raise ValueError('Unsealed or changed encoded training inputs')
        with np.load(encoded) as values:tensors={key:torch.from_numpy(values[key].copy()) for key in values.files}
    else:
        tensors=original.network.encode(tokenizer,samples,512)
        temporary=encoded.with_name('training-encoded.tmp.npz');np.savez_compressed(temporary,**{k:v.numpy() for k,v in tensors.items()});temporary.replace(encoded)
        write(encoded_receipt,{'sha256':sha(encoded),'training_inputs_sha256':registration['training_inputs_sha256']})
    labels=np.asarray([row['label'] for row in samples],dtype=np.int64)
    if (checkpoints/'LATEST.json').exists():
        position=checkpoint.restore(checkpoints,model,optimizer,scheduler,binding,args.device,history)
        if position!=next_position(position['step'],batches,grouped):raise ValueError('Resume batch order differs from original recipe')
        step=position['step']
    else:
        if history.exists() and history.stat().st_size:raise ValueError('History without a recovery checkpoint requires inspection')
        history.touch(exist_ok=True);step=0
        checkpoint.save(checkpoints,model,optimizer,scheduler,next_position(step,batches,grouped),binding,args.device,history)
    started=time.monotonic();last_checkpoint=None
    if step==2160:
        directory,_=checkpoint.load_manifest(checkpoints,binding,args.device)
        last_checkpoint={'manifest_sha256':sha(directory/'manifest.json'),'step':2160}
    with stop_at_boundary() as stopping:
        while step<2160:
            if stopping['requested']:
                last_checkpoint=checkpoint.save(checkpoints,model,optimizer,scheduler,next_position(step,batches,grouped),binding,args.device,history)
                write(args.output/'paused.json',{'reason':'signal','signal':stopping['signal'],'checkpoint':last_checkpoint});return
            epoch,batch_number=divmod(step,batches)
            order=np.random.default_rng(original.SEED+epoch).permutation(len(grouped))
            selected=[grouped[index] for index in order[2*batch_number:2*batch_number+2]]
            ids=[index for group in selected for index in group]
            model.train();optimizer.zero_grad(set_to_none=True)
            batch={key:value[ids].to(args.device) for key,value in tensors.items()}
            state,plain_unused,guided_unused=model(**batch)
            del plain_unused,guided_unused
            total,ce,bce=original.grouped_loss(original.original.margin(state),torch.tensor(labels[ids],device=args.device),[len(g) for g in selected])
            if not bool(torch.isfinite(total)):raise ValueError('Nonfinite grouped objective')
            total.backward();enc_norm,head_norm=original.norm(encoder),original.norm(heads)
            if not all(math.isfinite(v) and v>0 for v in (enc_norm,head_norm)):raise ValueError('Missing/nonfinite training gradient')
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.).cpu())
            optimizer.step();scheduler.step();step+=1
            append(history,{'step':step,'epoch':epoch+1,'questions':len(selected),'options':len(ids),
                'joint_loss':float(total.detach().cpu()),'listwise_ce':float(ce.detach().cpu()),'balanced_bce':float(bce.detach().cpu()),
                'encoder_gradient_norm':enc_norm,'head_gradient_norm':head_norm,'preclip_gradient_norm':norm})
            optimizer.zero_grad(set_to_none=True);del batch,state,total,ce,bce
            memory=None
            if step%10==0:
                memory=memory_snapshot(args.device);append(args.output/'memory-telemetry.jsonl',{'step':step,**memory})
            pressure=memory is not None and memory['system_free_percent'] is not None and memory['system_free_percent']<20
            if step%100==0 or step==2160 or stopping['requested'] or pressure:
                last_checkpoint=checkpoint.save(checkpoints,model,optimizer,scheduler,next_position(step,batches,grouped),binding,args.device,history)
                gc.collect()
                if args.device=='mps':torch.mps.empty_cache()
                print(json.dumps({'step':step,'steps':2160,'checkpoint':last_checkpoint,'elapsed_this_attempt_seconds':time.monotonic()-started}),flush=True)
                if stopping['requested'] or pressure:
                    write(args.output/'paused.json',{'reason':'signal' if stopping['requested'] else 'memory-pressure',
                        'signal':stopping['signal'],'memory':memory,'checkpoint':last_checkpoint});return
        if step!=2160 or scheduler.last_epoch!=2160:raise ValueError('Incomplete fixed schedule')
        verify_files(sources)
        checkpoint.save_deployable(model,tokenizer,args.output/'full',
            {**registration,'temperature':1.,'exact_training_checkpoint_sha256':last_checkpoint['manifest_sha256']})
    del model,optimizer,scheduler,tensors,encoder,heads;gc.collect()
    if args.device=='mps':torch.mps.empty_cache()
    if stopping['requested']:
        write(args.output/'paused.json',{'reason':'signal-after-final-archive','checkpoint':last_checkpoint});return
    final_calibration(args,registration,samples,last_checkpoint)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        initial,rows,samples,grouped,registration=prepare(args)
        if not args.execute:
            print(json.dumps({'registered':True,'training_started':False,'registration_sha256':sha(args.output/'training-registration.json')}));return
        execute(args,initial,samples,grouped,registration)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('metadata','source','initial','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','mps'),default='mps')
    parser.add_argument('--release',type=Path)
    parser.add_argument('--execute',action='store_true')
    run(parser.parse_args())
