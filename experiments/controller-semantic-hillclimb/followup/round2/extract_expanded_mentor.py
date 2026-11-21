"""Register/extract exactly960 new old-V1 observations, preserving4224 cached rows.

Registration is separate from explicit extraction. CPU is the default; MPS
must acquire the same exclusive lock as the serialized science trainer. This
module cannot fit a controller, collect LLM outputs, or read held-out panels.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

import expanded_mentor_inputs as inputs
import expanded_mentor_io as io
from prepare_mixed_mentor import fingerprint

SCHEMA = 'round2-expanded-oldV1-extraction-v1'
CHUNK, BATCH = 32, 4
ARG_PATHS = ('new_data', 'mentor', 'expansion', 'parent', 'verifier', 'legacy_examples', 'base', 'proposal')


def registration(args, bundle):
    recipe = io.read(args.verifier/'full/recipe.json')
    if recipe['max_length'] != 512 or recipe['fold'] != 'full':
        raise ValueError('Current variant requires unchanged full512 old-V1 observer')
    proposal = io.read(args.proposal)
    if proposal['schema'] != 'round2-expanded-mentor-fit-proposal-v1' or proposal['neural_cache']['new_rows'] != 960:
        raise ValueError('Require the separate expanded-fit proposal')
    sources = io.sources(sys.modules[__name__])
    corpus = Path(inputs.parent_fit.mentor.perception.CORPUS)
    assets = [p for p in (args.verifier/'full').iterdir() if p.is_file()]
    assets += [args.verifier/'training-summary.json', args.legacy_examples,
               corpus/'passages.jsonl', corpus/'tfidf.pkl']
    return {'schema': SCHEMA, 'source_files': sources,
        'asset_files': {str(p.resolve()): io.sha(p) for p in sorted(assets)},
        'paths': {name: str(getattr(args, name).resolve()) for name in ARG_PATHS},
        'output': str(args.output.resolve()), 'input_identity': bundle['identity'],
        'plan': bundle['extraction_plan'], 'device': args.device,
        'chunk_size': CHUNK, 'batch_size': BATCH, 'max_length': 512,
        'memory_guard': {'startup_free_percent': 35, 'continue_free_percent': 25,
                         'cpu_threads': 4, 'release_model_on_exit': True},
        'proposal_sha256': io.sha(args.proposal), 'runtime_recipe_sha256': io.sha(args.verifier/'full/recipe.json'),
        'no_labels_in_neural_inputs': True, 'text_uploaded': False,
        'variant_observer': 'old-full excluded all new512 questions',
        'training_only': True, 'development_or_test_used': False}


def register(args):
    if args.output.exists(): raise ValueError('Use a new extraction output directory')
    bundle = inputs.load_inputs(args)
    record = registration(args, bundle)
    args.output.mkdir(parents=True)
    io.write(args.output/'registration.json', record)
    io.write(args.output/'extraction-plan.json', bundle['extraction_plan'])
    print({'registered': True, 'registration_sha256': io.sha(args.output/'registration.json'),
           'missing_rows': len(bundle['missing_indexes'])}, flush=True)


def check_registration(args, bundle):
    record = io.read(args.output/'registration.json')
    if io.sha(args.output/'registration.json') != args.registration_sha256:
        raise ValueError('Explicit extraction registration SHA changed')
    io.verify(record['source_files'])
    io.verify(record['asset_files'])
    if record != registration(args, bundle):
        raise ValueError('Registered extraction source/inputs/settings changed')
    if io.read(args.output/'extraction-plan.json') != bundle['extraction_plan']:
        raise ValueError('Extraction plan changed')
    return record


def chunk_identity(bundle, indexes, registration_sha256):
    return {'registration_sha256': registration_sha256, 'indexes': indexes,
        'metadata_sha256': fingerprint([bundle['metadata'][i] for i in indexes]),
        'inputs_sha256': fingerprint([bundle['inputs'][i] for i in indexes])}


def validate_arrays(values, n, dtypes):
    if set(values) != set(inputs.WIDTHS): raise ValueError('Changed neural output heads')
    for name, width in inputs.WIDTHS.items():
        value = np.asarray(values[name])
        if value.shape != (n, width) or value.dtype != np.dtype(dtypes[name]) or not np.isfinite(value).all():
            raise ValueError('Changed neural output shape/dtype/finite values')


def read_chunk(path, expected, dtypes):
    receipt_path = path.with_suffix('.json')
    if not path.exists() or not receipt_path.exists():
        raise ValueError('Preserved incomplete chunk requires inspection; never regenerate its completed outputs')
    receipt = io.read(receipt_path)
    if receipt['identity'] != expected or receipt['outputs_sha256'] != io.sha(path):
        raise ValueError('Changed sealed neural chunk')
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    validate_arrays(values, len(expected['indexes']), dtypes)
    if (receipt['successful_rows'] != len(expected['indexes'])
            or receipt['padded_token_slots'] != len(expected['indexes'])*512
            or receipt['runtime_terminal_charge'] is not False):
        raise ValueError('Changed completed chunk row count')
    return values, receipt


def preserve_parent(bundle, arrays):
    for expected, parent in bundle['reused_indexes'].items():
        for name, value in bundle['cached_outputs'].items():
            if arrays[name][expected].tobytes() != value[parent].tobytes():
                raise ValueError('An exact parent neural row changed')


def memory_guard(output, loaded):
    snapshot = {'time_unix': time.time(), 'pid': os.getpid(), 'model_loaded': loaded}
    for name, command in [('memory_pressure', ['memory_pressure', '-Q']),
                          ('rss_kib', ['ps', '-p', str(os.getpid()), '-o', 'rss=']),
                          ('swap', ['sysctl', 'vm.swapusage'])]:
        try: snapshot[name] = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error: snapshot[name+'_unavailable'] = type(error).__name__
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', snapshot.get('memory_pressure', ''))
    free = int(match[1]) if match else None
    threshold = 25 if loaded else 35
    snapshot.update(system_free_percent=free, required_free_percent=threshold)
    io.append(output/'memory.jsonl', snapshot)
    if free is None or free < threshold:
        raise RuntimeError('Memory headroom below extraction threshold; completed chunks preserved, model released on exit')


def verify_complete(output, bundle, registration_sha256):
    complete = io.read(output/'neural-complete.json')
    if (complete['schema'] != SCHEMA or complete['registration_sha256'] != registration_sha256
        or complete['n'] != 5184 or complete['reused'] != 4224 or complete['new'] != 960
        or complete['development_or_test_used'] or complete['input_identity_sha256'] != fingerprint(bundle['identity'])):
        raise ValueError('Changed completed expanded neural extraction')
    if io.sha(output/'registration.json') != registration_sha256:
        raise ValueError('Completed extraction registration SHA changed')
    registered = io.read(output/'registration.json')
    if (registered['schema'] != SCHEMA or registered['max_length'] != 512
            or registered['chunk_size'] != CHUNK or registered['batch_size'] != BATCH
            or registered['device'] not in ('cpu', 'mps') or registered['output'] != str(output.resolve())):
        raise ValueError('Changed registered extraction contract')
    io.verify(registered['source_files'])
    io.verify(registered['asset_files'])
    expected_names = {'registration.json', 'extraction-plan.json', 'metadata.jsonl', 'neural.npz',
                      'neural-receipt.json', 'attempts.jsonl'}
    if registered.get('memory_guard'): expected_names.add('memory.jsonl')
    for start in range(0, len(bundle['missing_indexes']), CHUNK):
        expected_names |= {f'chunks/{start//CHUNK:03d}.npz', f'chunks/{start//CHUNK:03d}.json'}
    if set(complete['digests']) != expected_names:
        raise ValueError('Completed extraction omitted or added sealed artifacts')
    io.verify({str(output/name): value for name, value in complete['digests'].items()})
    if registered['input_identity'] != bundle['identity']:
        raise ValueError('Completed inputs differ from registration')
    if io.read(output/'extraction-plan.json') != bundle['extraction_plan']:
        raise ValueError('Completed extraction plan changed')
    metadata = [json.loads(line) for line in (output/'metadata.jsonl').read_text().splitlines()]
    if metadata != bundle['metadata']: raise ValueError('Completed neural metadata changed')
    with np.load(output/'neural.npz', allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    dtypes = {name: value.dtype.str for name, value in bundle['cached_outputs'].items()}
    validate_arrays(arrays, 5184, dtypes); preserve_parent(bundle, arrays)
    receipt = io.read(output/'neural-receipt.json')
    expected_receipt = {'n': 5184, 'reused_full_observer_nodes': 4224, 'new_rows': 960,
        'encoder_recipes': {'full': registered['runtime_recipe_sha256']},
        'inputs_sha256': fingerprint(bundle['inputs']), 'metadata_sha256': io.sha(output/'metadata.jsonl'),
        'outputs_sha256': io.sha(output/'neural.npz'), 'reference_answers_in_inputs': False,
        'text_uploaded': False, 'unseen_new_questions': 512, 'legacy_questions': 0,
        'registration_sha256': registration_sha256, 'device': registered['device'],
        'new_successful_padded_token_slots': 960*512,
        'attempt_journal_note': 'Failed/interrupted attempts remain recorded; only sealed chunks count as successful rows'}
    if receipt != expected_receipt: raise ValueError('Changed semantic neural completion receipt')
    for start in range(0, len(bundle['missing_indexes']), CHUNK):
        indexes = bundle['missing_indexes'][start:start+CHUNK]
        values, _ = read_chunk(output/'chunks'/f'{start//CHUNK:03d}.npz',
                              chunk_identity(bundle, indexes, registration_sha256), dtypes)
        for name in arrays:
            if arrays[name][indexes].tobytes() != values[name].tobytes():
                raise ValueError('Combined outputs differ from their sealed new chunks')
    return arrays, complete


def extract_arrays(output, bundle, record, registration_sha256, forward, before_forward=None):
    """Durable orchestration, separable from the model for synthetic testing."""
    if (output/'neural-complete.json').exists():
        return verify_complete(output, bundle, registration_sha256)
    chunks = output/'chunks'; chunks.mkdir(exist_ok=True)
    if list(chunks.glob('*.tmp')) or list(output.glob('*.tmp')):
        raise ValueError('Preserved temporary extraction output requires inspection; no re-forward')
    dtypes = {name: value.dtype.str for name, value in bundle['cached_outputs'].items()}
    arrays = {name: np.full((len(bundle['metadata']), width), np.nan, dtype=dtypes[name])
              for name, width in inputs.WIDTHS.items()}
    for expected, parent in bundle['reused_indexes'].items():
        for name in arrays: arrays[name][expected] = bundle['cached_outputs'][name][parent]
    missing = bundle['missing_indexes']; seals = {}
    attempts_path = output/'attempts.jsonl'
    if attempts_path.exists():
        raw = attempts_path.read_bytes()
        if not raw.endswith(b'\n'): raise ValueError('Preserve malformed attempt-journal suffix for inspection')
        for line in raw.splitlines():
            attempt = json.loads(line); start = attempt['chunk']*CHUNK
            indexes = missing[start:start+CHUNK]
            if (not indexes or attempt['identity'] != chunk_identity(bundle, indexes, registration_sha256)
                    or attempt['attempted_rows'] != len(indexes)):
                raise ValueError('Changed interrupted extraction attempt')
    for start in range(0, len(missing), CHUNK):
        indexes = missing[start:start+CHUNK]; path = chunks/f'{start//CHUNK:03d}.npz'
        identity = chunk_identity(bundle, indexes, registration_sha256)
        if path.exists() or path.with_suffix('.json').exists():
            values, receipt = read_chunk(path, identity, dtypes)
        else:
            if before_forward is not None: before_forward()
            io.append(output/'attempts.jsonl', {'chunk': start//CHUNK, 'identity': identity,
                                               'attempted_rows': len(indexes)})
            values, slots = forward([bundle['inputs'][i] for i in indexes])
            validate_arrays(values, len(indexes), dtypes)
            if type(slots) is not int or slots != len(indexes)*record['max_length']:
                raise ValueError('Unexpected padded training encoder slots')
            io.npz(path, values)
            receipt = {'identity': identity, 'outputs_sha256': io.sha(path),
                'successful_rows': len(indexes), 'padded_token_slots': slots,
                'runtime_terminal_charge': False}
            io.write(path.with_suffix('.json'), receipt)
        for name in arrays: arrays[name][indexes] = values[name]
        for p in (path, path.with_suffix('.json')): seals[str(p.relative_to(output))] = io.sha(p)
        print({'completed_new_rows': start+len(indexes), 'total_new_rows': len(missing)}, flush=True)
    validate_arrays(arrays, len(bundle['metadata']), dtypes); preserve_parent(bundle, arrays)
    io.npz(output/'neural.npz', arrays)
    io.jsonl(output/'metadata.jsonl', bundle['metadata'])
    receipt = {'n': len(bundle['metadata']), 'reused_full_observer_nodes': len(bundle['reused_indexes']),
        'new_rows': len(missing), 'encoder_recipes': {'full': record['runtime_recipe_sha256']},
        'inputs_sha256': fingerprint(bundle['inputs']), 'metadata_sha256': io.sha(output/'metadata.jsonl'),
        'outputs_sha256': io.sha(output/'neural.npz'), 'reference_answers_in_inputs': False,
        'text_uploaded': False, 'unseen_new_questions': 512, 'legacy_questions': 0,
        'registration_sha256': registration_sha256, 'device': record['device'],
        'new_successful_padded_token_slots': len(missing)*record['max_length'],
        'attempt_journal_note': 'Failed/interrupted attempts remain recorded; only sealed chunks count as successful rows'}
    io.write(output/'neural-receipt.json', receipt)
    names = ('registration.json', 'extraction-plan.json', 'metadata.jsonl', 'neural.npz',
             'neural-receipt.json', 'attempts.jsonl')
    seals.update({name: io.sha(output/name) for name in names})
    if record.get('memory_guard'): seals['memory.jsonl'] = io.sha(output/'memory.jsonl')
    io.verify(record['source_files'])
    io.verify(record['asset_files'])
    complete = {'schema': SCHEMA, 'registration_sha256': registration_sha256,
        'input_identity_sha256': fingerprint(bundle['identity']), 'n': len(bundle['metadata']),
        'reused': len(bundle['reused_indexes']), 'new': len(missing), 'digests': seals,
        'development_or_test_used': False, 'cloud_calls': 0, 'controller_fitted': False}
    io.write(output/'neural-complete.json', complete)
    return arrays, complete


def extract(args):
    with io.lock(args.output/'.extract.lock'):
        bundle = inputs.load_inputs(args)
        record = check_registration(args, bundle)
        if (args.output/'neural-complete.json').exists():
            verify_complete(args.output, bundle, args.registration_sha256); return
        device_guard = io.lock('/private/tmp/srr-training-mps.lock') if args.device == 'mps' else nullcontext()
        with device_guard:
            from network_large import encode, predict
            observer = inputs.parent_fit.mentor.perception.Observer(str((args.verifier/'full').resolve()),
                bundle['identity']['inherited_temperature'], device=args.device)
            def forward(entries):
                observer.load()
                tensors = encode(observer._tokenizer, entries, record['max_length'])
                values = predict(observer._model, tensors, args.device, batch=BATCH)
                arrays = dict(zip(inputs.WIDTHS, values))
                slots = int(tensors['input_ids'].shape[0]*tensors['input_ids'].shape[1])
                del tensors, values
                return arrays, slots
            try: extract_arrays(args.output, bundle, record, args.registration_sha256, forward,
                                lambda: memory_guard(args.output, observer._model is not None))
            finally:
                observer._model = None; observer._tokenizer = None; gc.collect()
        io.verify(record['source_files'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('register', 'extract'))
    for name in (*ARG_PATHS, 'output'): p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    p.add_argument('--registration-sha256')
    args = p.parse_args()
    if args.mode == 'register': register(args)
    else:
        if not args.registration_sha256: p.error('Extraction requires the reviewed registration SHA')
        extract(args)


if __name__ == '__main__': main()
