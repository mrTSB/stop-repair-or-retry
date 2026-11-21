"""Explicit operational override for a bounded, same-recipe science-v3 resume.

Only unused MPS allocator cache is released immediately before the frozen
trainer's ten-step telemetry. Arithmetic, data, batch order, RNG, optimizer,
scheduler, weight archive selection and the original pressure guard stay fixed.
The reviewed frozen trainer/checkpoint files are never edited.
"""
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import time

import train_science_listwise_resumable as trainer

SCHEMA = 'science-v3-pre-telemetry-cache-release-v1'
sha, write = trainer.sha, trainer.write


def read(path):
    return json.loads(Path(path).read_bytes())


def checkpoint_binding(output):
    record = read(output / 'training-registration.json')
    return {'registration_sha256': sha(output / 'training-registration.json'),
        'training_inputs_sha256': record['training_inputs_sha256'], 'planned_steps': 2160,
        'seed': trainer.original.SEED, 'batch_questions': 2}


def specification(args):
    if args.device != 'mps' or args.until_step != 100:
        raise ValueError('This first bounded operational check is MPS through completed step100 only')
    record = read(args.output / 'training-registration.json')
    trainer.verify_files(record['source_files'])
    path, manifest = trainer.checkpoint.load_manifest(args.output / 'recovery', checkpoint_binding(args.output), 'mps')
    if manifest['progress']['step'] != 20 or (args.output / 'training-summary.json').exists() or (args.output / 'full').exists():
        raise ValueError('Resume only the preserved incomplete step20 checkpoint')
    pause = read(args.output / 'paused.json')
    if pause['reason'] != 'memory-pressure' or pause['checkpoint']['manifest_sha256'] != sha(path / 'manifest.json'):
        raise ValueError('Require the exact reviewed memory-pressure pause and atomic checkpoint')
    release = read(args.release)
    if release['output'] != str(args.output.resolve()) or release['trainer_sha256'] != sha(trainer.__file__):
        raise ValueError('Changed serialized training release')
    trainer.verify_files(release['prerequisites'])
    return {'schema': SCHEMA, 'source_files': {**trainer.source_files(), str(Path(__file__).resolve()): sha(__file__)},
        'training_registration_sha256': sha(args.output / 'training-registration.json'),
        'checkpoint_manifest_sha256': sha(path / 'manifest.json'), 'checkpoint_files': len(manifest['files']),
        'starting_progress': manifest['progress'], 'until_step': args.until_step,
        'original_pause_sha256': sha(args.output / 'paused.json'), 'release_sha256': sha(args.release),
        'paths': {name: str(getattr(args, name).resolve()) for name in ('metadata', 'source', 'initial', 'output', 'release', 'operation')},
        'operations': ['Before each existing MPS ten-step telemetry: gc.collect, synchronize, empty_cache, synchronize, original measurement',
                       'After the original atomic step100 checkpoint: request the original graceful SIGTERM boundary handler'],
        'unchanged': ['training data', 'last6-layer/head LR', 'loss', 'batch shapes/order', 'RNG', 'optimizer', 'scheduler',
                      '2160-step final archive rule', '35percent startup guard', '20percent pressure pause guard'],
        'boundaries': 'Only fully completed optimizer+scheduler steps can be saved or stopped',
        'no_best_checkpoint_selection': True, 'no_deployment_or_calibration_before2160': True,
        'training_only': True, 'new_cloud_calls': 0, 'new_development_or_test_inference': False}


@contextmanager
def operational_hooks(until_step, telemetry_path):
    """Narrow, explicitly registered function overrides around frozen execute()."""
    original_memory, original_save = trainer.memory_snapshot, trainer.checkpoint.save
    def memory(device='cpu'):
        if device != 'mps':
            return original_memory(device)
        before = {'tensor_bytes': trainer.torch.mps.current_allocated_memory(),
                  'driver_bytes': trainer.torch.mps.driver_allocated_memory()}
        started = time.monotonic()
        gc.collect(); trainer.torch.mps.synchronize(); trainer.torch.mps.empty_cache(); trainer.torch.mps.synchronize()
        result = original_memory(device)
        trainer.append(telemetry_path, {'unix': time.time(), 'before_release': before,
            'after_release': result, 'cleanup_seconds': time.monotonic()-started,
            'method': 'gc+synchronize+empty_cache+synchronize', 'no_arithmetic_or_rng_change': True})
        return result
    def save(*args, **kwargs):
        progress = args[4] if len(args) > 4 else kwargs['progress']
        result = original_save(*args, **kwargs)
        if progress['step'] >= until_step:
            if progress['step'] != until_step or not callable(signal.getsignal(signal.SIGTERM)):
                raise ValueError('Bounded resume requires the original completed-boundary signal handler')
            # The frozen loop checks its stop flag after this atomic checkpoint,
            # before another optimizer step; never snapshot mid-backward/update.
            os.kill(os.getpid(), signal.SIGTERM)
        return result
    trainer.memory_snapshot, trainer.checkpoint.save = memory, save
    try:
        yield
    finally:
        trainer.memory_snapshot, trainer.checkpoint.save = original_memory, original_save


def register(args):
    record = specification(args)
    args.operation.mkdir(parents=True, exist_ok=False)
    write(args.operation / 'registration.json', record)
    trainer.checkpoint.atomic_bytes(args.operation / 'original-paused.json', (args.output / 'paused.json').read_bytes())
    print(json.dumps({'registered': True, 'operation_registration_sha256': sha(args.operation / 'registration.json'),
                      'starting_step': 20, 'bounded_until_step': 100, 'training_started': False}), flush=True)


def execute(args):
    with ExitStack() as stack:
        for path in (args.operation / '.operation.lock', Path('/private/tmp/srr-training-mps.lock'), args.output / '.training.lock'):
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.operation / 'attempt-started.json').exists():
            raise ValueError('Preserve this bounded attempt; inspect its receipts before any new resume')
        record = specification(args)
        if sha(args.operation / 'registration.json') != args.registration_sha256 or read(args.operation / 'registration.json') != record:
            raise ValueError('Externally recorded operational override changed')
        trainer.release_guard(args)
        initial, _, samples, grouped, training_record = trainer.prepare(args)
        write(args.operation / 'attempt-started.json', {'unix': time.time(), 'pid': os.getpid(),
            'operation_registration_sha256': args.registration_sha256, 'training_arithmetic_changed': False})
        try:
            with operational_hooks(args.until_step, args.operation / 'cache-release-telemetry.jsonl'):
                trainer.execute(args, initial, samples, grouped, training_record)
        except BaseException as error:
            write(args.operation / 'attempt-result.json', {'completed': False, 'exception_type': type(error).__name__,
                'exception': str(error), 'latest_checkpoint_pointer_sha256': sha(args.output / 'recovery/LATEST.json'),
                'resume_rule': 'Only the last fully sealed recovery checkpoint; no mid-operation snapshot'})
            raise
        path, manifest = trainer.checkpoint.load_manifest(args.output / 'recovery', checkpoint_binding(args.output), 'mps')
        if manifest['progress']['step'] > 100 or (args.output / 'full').exists():
            raise ValueError('Bounded operational check exceeded its registered step limit')
        trainer.verify_files(record['source_files'])
        write(args.operation / 'attempt-result.json', {'completed': True, 'final_progress': manifest['progress'],
            'checkpoint_manifest_sha256': sha(path / 'manifest.json'), 'pause': read(args.output / 'paused.json'),
            'reached100': manifest['progress']['step'] == 100, 'no_deployment_checkpoint': True,
            'cache_telemetry_sha256': sha(args.operation / 'cache-release-telemetry.jsonl'),
            'operation_registration_sha256': args.registration_sha256})
        print(json.dumps({'bounded_resume_complete': True, 'step': manifest['progress']['step'],
                          'result_sha256': sha(args.operation / 'attempt-result.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'execute'))
    for name in ('metadata', 'source', 'initial', 'output', 'release', 'operation'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('mps',), default='mps')
    parser.add_argument('--until-step', type=int, default=100)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    {'register': register, 'execute': execute}[args.phase](args)
