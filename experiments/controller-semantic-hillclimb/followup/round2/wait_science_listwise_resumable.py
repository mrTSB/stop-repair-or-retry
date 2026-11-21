"""Serialize v3 science training after V4 archives/OOF and refined-A development."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import train_science_listwise_resumable as trainer
import refine_stream
import development_refined_observer

sha,write=trainer.sha,trainer.write


def alive(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    return True


def prerequisites(v4,development,pid):
    required=[v4/'training-summary.json',v4/'oof.npz',v4/'contract.json',
              development/'summary.json',development/'identity.json',development/'oof-provenance.json']
    required += [v4/name/'recipe.json' for name in ('0','1','2','full')]
    if any(not path.exists() for path in required) or alive(pid):return None
    contract=json.loads((v4/'contract.json').read_bytes())
    if contract.get('trainer_sha256')!=sha(refine_stream.__file__):raise ValueError('V4 source changed')
    summary=json.loads((v4/'training-summary.json').read_bytes())
    if (summary.get('fresh_data_used') is not False or summary.get('development_data_used') is not False
            or summary.get('question_grouped_folds')!=3):raise ValueError('Require completed training-only V4 OOF receipt')
    development_summary=json.loads((development/'summary.json').read_bytes())
    identity=json.loads((development/'identity.json').read_bytes())
    if (development_summary.get('test_used') is not False or identity.get('test_used') is not False
            or development_summary.get('development_questions')!=256
            or identity.get('source_sha256')!=sha(development_refined_observer.__file__)
            or identity.get('verifier')!=str(v4.resolve())):
        raise ValueError('Require the completed registered refined-A development run')
    provenance=json.loads((development/'oof-provenance.json').read_bytes())
    if (development_summary.get('oof_provenance_sha256')!=sha(development/'oof-provenance.json')
            or provenance.get('oof_sha256')!=sha(v4/'oof.npz')
            or provenance.get('training_contract_sha256')!=sha(v4/'contract.json')
            or provenance.get('training_summary_sha256')!=sha(v4/'training-summary.json')):
        raise ValueError('Refined development does not bind these completed V4 observations')
    for name in ('0','1','2','full'):
        folder=v4/name;recipe=json.loads((folder/'recipe.json').read_bytes())
        if recipe.get('fold')!=(int(name) if name!='full' else name):raise ValueError('V4 archived fold changed')
        for filename,expected in recipe['weights'].items():
            path=folder/filename
            if sha(path)!=expected:raise ValueError('V4 final archived weight changed')
            required.append(path)
    return {str(path.resolve()):sha(path) for path in required}


def run(args):
    args.queue.mkdir(parents=True,exist_ok=True)
    source_files=trainer.source_files() | {str(Path(__file__).resolve()):sha(__file__),
        str(Path(refine_stream.__file__).resolve()):sha(refine_stream.__file__),
        str(Path(development_refined_observer.__file__).resolve()):sha(development_refined_observer.__file__)}
    expected={'schema':'science-v3-serialized-waiter-v1','source_files':source_files,
        'metadata':str(args.metadata.resolve()),'source':str(args.source.resolve()),'initial':str(args.initial.resolve()),
        'output':str(args.output.resolve()),'v4_directory':str(args.v4.resolve()),
        'refined_development':str(args.refined_development.resolve()),'protected_v4_pid':args.v4_pid,
        'minimum_system_free_percent':35,'operations':trainer.OPERATIONS,'training_recipe':trainer.original.RECIPE,
        'no_algorithm_or_batch_change':True,'no_training_before_prerequisites':True}
    with (args.queue/'.waiter.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registration=args.queue/'registration.json'
        if registration.exists():
            if json.loads(registration.read_bytes())!=expected:raise ValueError('Changed queued training recipe')
        else:write(registration,expected)
        if not args.execute:
            print(json.dumps({'registered':True,'training_launched':False}));return
        while True:
            trainer.verify_files(source_files)
            ready=prerequisites(args.v4,args.refined_development,args.v4_pid)
            if ready is not None:
                memory=trainer.memory_snapshot('cpu')
                if memory['system_free_percent'] is not None and memory['system_free_percent']>=35:break
            time.sleep(30)
        # This lock covers the entire child process and serializes every new
        # training waiter using this scheduler. Frozen V4 is gated separately.
        with Path('/private/tmp/srr-training-mps.lock').open('a') as device_lock:
            while True:
                try:fcntl.flock(device_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(30)
            ready=prerequisites(args.v4,args.refined_development,args.v4_pid)
            if ready is None:raise ValueError('Prerequisites changed while awaiting training slot')
            trainer.verify_files(source_files)
            memory=trainer.memory_snapshot('cpu')
            if memory['system_free_percent'] is None or memory['system_free_percent']<35:
                raise ValueError('Memory headroom fell below35%; rerun waiter without changing recipe')
            receipt={'schema':'science-v3-resource-release-v1','output':str(args.output.resolve()),
                'protected_v4_pid':args.v4_pid,'v4_process_observed_gone':True,
                'trainer_sha256':sha(trainer.__file__),'waiter_sha256':sha(__file__),
                'registration_sha256':sha(registration),'prerequisites':ready,'memory_before_launch':memory,
                'training_recipe_changed':False,'batch_shapes_changed':False,'released_unix':time.time()}
            release=args.queue/'release.json';write(release,receipt)
            command=[sys.executable,str(Path(trainer.__file__).resolve()),'--metadata',str(args.metadata),
                '--source',str(args.source),'--initial',str(args.initial),'--output',str(args.output),
                '--device','mps','--release',str(release),'--execute']
            print('Prerequisites complete; launching serialized science-v3 training.',flush=True)
            completed=subprocess.run(command,check=False)
            write(args.queue/'process-result.json',{'returncode':completed.returncode,
                'training_summary_exists':(args.output/'training-summary.json').exists(),
                'paused_receipt_exists':(args.output/'paused.json').exists(),'release_sha256':sha(release)})
            if completed.returncode:raise SystemExit(completed.returncode)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('metadata','source','initial','output','queue','v4','refined-development'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--v4-pid',type=int,required=True)
    parser.add_argument('--execute',action='store_true')
    run(parser.parse_args())
