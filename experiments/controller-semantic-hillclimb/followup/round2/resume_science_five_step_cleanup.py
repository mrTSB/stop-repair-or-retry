"""Same-recipe step100-to2160 resume with explicitly registered cache cleanup.

Release unused MPS cache after each fifth completed update and before the
existing ten-step telemetry. Original numerical operations, RNG, checkpoint
boundaries, memory guards, final archive and auxiliary calibration are intact.
There is no artificial signal at2160; the frozen finalization runs normally.
"""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import time

import resume_science_cache_cleanup as predecessor

trainer = predecessor.trainer
read, sha, write = predecessor.read, predecessor.sha, predecessor.write
SCHEMA = 'science-v3-five-step-cache-release-full-resume-v1'


def specification(args):
    if args.device != 'mps':
        raise ValueError('Preserve the original MPS RNG domain')
    registration = read(args.output/'training-registration.json')
    trainer.verify_files(registration['source_files'])
    folder, manifest = trainer.checkpoint.load_manifest(args.output/'recovery', predecessor.checkpoint_binding(args.output), 'mps')
    if manifest['progress']['step'] != 100 or (args.output/'full').exists() or (args.output/'training-summary.json').exists():
        raise ValueError('Require the reviewed incomplete step100 checkpoint')
    prior = read(args.previous_operation/'registration.json')
    result = read(args.previous_operation/'attempt-result.json')
    trainer.verify_files(prior['source_files'])
    if (prior['schema'] != predecessor.SCHEMA or result['operation_registration_sha256'] != sha(args.previous_operation/'registration.json')
            or not result['completed'] or not result['reached100'] or result['final_progress'] != manifest['progress']
            or result['checkpoint_manifest_sha256'] != sha(folder/'manifest.json')
            or result['cache_telemetry_sha256'] != sha(args.previous_operation/'cache-release-telemetry.jsonl')
            or prior['training_registration_sha256'] != sha(args.output/'training-registration.json')
            or prior['paths']['output'] != str(args.output.resolve())):
        raise ValueError('Changed reviewed bounded-resume lineage')
    pause = read(args.output/'paused.json')
    if pause != result['pause'] or pause['checkpoint']['manifest_sha256'] != sha(folder/'manifest.json'):
        raise ValueError('Current pause differs from the reviewed step100 boundary')
    release = read(args.release)
    if release['output'] != str(args.output.resolve()) or release['trainer_sha256'] != sha(trainer.__file__):
        raise ValueError('Changed serialized scheduler release')
    trainer.verify_files(release['prerequisites'])
    return {'schema': SCHEMA, 'sources': {**trainer.source_files(),
        str(Path(predecessor.__file__).resolve()): sha(predecessor.__file__), str(Path(__file__).resolve()): sha(__file__)},
        'training_registration_sha256': sha(args.output/'training-registration.json'),
        'starting_checkpoint_manifest_sha256': sha(folder/'manifest.json'), 'starting_progress': manifest['progress'],
        'release_sha256': sha(args.release), 'pause_sha256': sha(args.output/'paused.json'),
        'previous_operation_files': {str((args.previous_operation/name).resolve()): sha(args.previous_operation/name)
            for name in ('registration.json', 'attempt-result.json', 'cache-release-telemetry.jsonl', 'original-paused.json')},
        'paths': {name: str(getattr(args, name).resolve()) for name in
            ('metadata', 'source', 'initial', 'output', 'release', 'operation', 'previous_operation')},
        'cache_release_every_completed_updates': 5, 'additional_pretelemetry_release': True,
        'cleanup_order': ['gc.collect', 'MPS synchronize', 'MPS empty_cache', 'MPS synchronize'],
        'history_hook_boundary': 'After original durable history append, which follows completed optimizer and scheduler steps',
        'unchanged_training_registration': True, 'unchanged_arithmetic_and_RNG': True,
        'unchanged_recipe_batch_order_shapes_and_hyperparameters': True,
        'starting_memory_free_minimum_percent': 35, 'original_pressure_pause_below_percent': 20,
        'scheduled_final_step': 2160, 'no_artificial_stop_signal': True,
        'finalization': 'Original final2160 archive followed by original160-question auxiliary calibration',
        'checkpoint_selection': 'Final2160 only; recovery checkpoints are never selected by quality',
        'new_cloud_calls': 0, 'new_project_development_or_test_inference': False}


@contextmanager
def operational_hooks(history, telemetry):
    original_append, original_memory = trainer.append, trainer.memory_snapshot
    history = Path(history).resolve()
    def cleanup(reason, step=None):
        before = {'tensor_bytes': trainer.torch.mps.current_allocated_memory(),
                  'driver_bytes': trainer.torch.mps.driver_allocated_memory()}
        started = time.monotonic()
        gc.collect(); trainer.torch.mps.synchronize(); trainer.torch.mps.empty_cache(); trainer.torch.mps.synchronize()
        original_append(telemetry, {'unix': time.time(), 'reason': reason, 'step': step,
            'before_release': before, 'after_release': {
                'tensor_bytes': trainer.torch.mps.current_allocated_memory(),
                'driver_bytes': trainer.torch.mps.driver_allocated_memory()},
            'cleanup_seconds': time.monotonic()-started, 'no_arithmetic_or_rng_change': True})
    def append(path, value):
        original_append(path, value)
        if Path(path).resolve() == history:
            step = value.get('step')
            if type(step) is not int or not 0 < step <= 2160:
                raise ValueError('Unexpected completed-step history receipt')
            if step % 5 == 0:
                cleanup('completed-history-step', step)
    def memory(device='cpu'):
        if device == 'mps':
            cleanup('before-original-ten-step-telemetry')
        return original_memory(device)
    trainer.append, trainer.memory_snapshot = append, memory
    try:
        yield
    finally:
        trainer.append, trainer.memory_snapshot = original_append, original_memory


def register(args):
    record = specification(args)
    args.operation.mkdir(parents=True, exist_ok=False)
    write(args.operation/'registration.json', record)
    trainer.checkpoint.atomic_bytes(args.operation/'original-paused.json', (args.output/'paused.json').read_bytes())
    print(json.dumps({'registered': True, 'registration_sha256': sha(args.operation/'registration.json'),
                      'starting_step': 100, 'final_step': 2160, 'training_started': False}), flush=True)


def execute(args):
    with ExitStack() as stack:
        for path in (args.operation/'.operation.lock', Path('/private/tmp/srr-training-mps.lock'), args.output/'.training.lock'):
            handle = stack.enter_context(path.open('a')); fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.operation/'attempt-started.json').exists():
            raise ValueError('Preserve this attempt and inspect its last sealed checkpoint before another resume')
        record = specification(args)
        if (sha(args.operation/'registration.json') != args.registration_sha256
                or read(args.operation/'registration.json') != record):
            raise ValueError('Externally registered full-resume operation changed')
        trainer.release_guard(args)
        initial, _, samples, grouped, registration = trainer.prepare(args)
        write(args.operation/'attempt-started.json', {'unix': time.time(), 'pid': os.getpid(),
            'registration_sha256': args.registration_sha256, 'arithmetic_changed': False})
        try:
            with operational_hooks(args.output/'optimization-history.jsonl', args.operation/'cache-release-telemetry.jsonl'):
                trainer.execute(args, initial, samples, grouped, registration)
        except BaseException as error:
            write(args.operation/'attempt-result.json', {'completed': False, 'exception_type': type(error).__name__,
                'exception': str(error), 'latest_pointer_sha256': sha(args.output/'recovery/LATEST.json'),
                'resume_rule': 'Only last fully sealed recovery checkpoint; no emergency mid-update snapshot'})
            raise
        folder, manifest = trainer.checkpoint.load_manifest(args.output/'recovery', predecessor.checkpoint_binding(args.output), 'mps')
        finished = (args.output/'training-summary.json').exists()
        if finished:
            if manifest['progress']['step'] != 2160:
                raise ValueError('Deployment summary before all2160 updates')
            summary = read(args.output/'training-summary.json'); recipe = read(args.output/'full/recipe.json')
            if (summary['registration_sha256'] != record['training_registration_sha256']
                    or summary['recipe_sha256'] != sha(args.output/'full/recipe.json')
                    or recipe['exact_training_checkpoint_sha256'] != sha(folder/'manifest.json')):
                raise ValueError('Final training archive lost its original recipe/checkpoint binding')
            trainer.verify_files({str(args.output/'full'/name): digest for name, digest in recipe['weights'].items()})
        trainer.verify_files(record['sources'])
        telemetry = args.operation/'cache-release-telemetry.jsonl'
        write(args.operation/'attempt-result.json', {'completed': True, 'training_finished': finished,
            'final_progress': manifest['progress'], 'checkpoint_manifest_sha256': sha(folder/'manifest.json'),
            'registration_sha256': args.registration_sha256,
            'training_summary_sha256': sha(args.output/'training-summary.json') if finished else None,
            'final_recipe_sha256': sha(args.output/'full/recipe.json') if finished else None,
            'pause': None if finished else read(args.output/'paused.json'),
            'cache_telemetry_sha256': sha(telemetry) if telemetry.exists() else None})
        print(json.dumps({'full_resume_returned': True, 'training_finished': finished,
            'step': manifest['progress']['step'], 'result_sha256': sha(args.operation/'attempt-result.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'execute'))
    for name in ('metadata', 'source', 'initial', 'output', 'release', 'operation', 'previous-operation'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('mps',), default='mps')
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    {'register': register, 'execute': execute}[args.phase](args)
