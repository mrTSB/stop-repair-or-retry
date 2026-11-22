"""Registered local frozen-reward TRAIN acquisition and nested outcome screen."""
import argparse
import gc
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import policy_crossfit_diagnostic as dataset
import train_joint_outcome_observer as conditional
import reward_outcome_observer as observer

io, fixed, nested = dataset.io, dataset.fixed, dataset.nested
SCHEMA = 'frozen-reward-science1280-conditional-TRAIN-v1'


def inputs(args):
    rows, trees, old, data, parent = dataset.inputs(args)
    receipt = io.read(args.assets/'receipt.json')
    if receipt['repository'] != observer.REPOSITORY or receipt['revision'] != observer.REVISION or receipt['remote_code']:
        raise ValueError('Require pinned public model without remote code')
    files = {str((args.assets/'model'/k).resolve()): v for k, v in receipt['files'].items()}
    io.verify(files)
    files[str((args.assets/'receipt.json').resolve())] = io.sha(args.assets/'receipt.json')
    by_id = {r['id']: r for r in rows}
    conversations = [observer.conversation({**by_id[r['id']], 'answer': ''}, r['path'],
        {p: trees[r['id']][p] for p in ('', r['path'])}) for r in data['records']]
    # Remove all old neural outputs: these are the first twenty raw features.
    for values in old.values_by_key.values():
        if sum(np.asarray(v).size for v in values[:3]) != 20:
            raise ValueError('Old neural feature layout changed')
    base = data['features'][nested.FAMILIES[0]][:, 20:].copy()
    if base.shape != (3840, 110): raise ValueError('Changed paid numeric observations')
    with np.load(args.expanded_fit/'private-oof.npz', allow_pickle=False) as archive:
        control = archive[nested.FAMILIES[0]].copy()
    slots = {key: value[3] for key, value in old.values_by_key.items()}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**parent['input_evidence'], **files,
            str((args.expanded_fit/'private-oof.npz').resolve()): io.sha(args.expanded_fit/'private-oof.npz')},
        'paths': {k: str(getattr(args, k).resolve()) for k in ('expanded_fit', 'assets', 'output')},
        'model': receipt, 'device': args.device, 'dtype': 'float32', 'batch_size': 4,
        'max_length': observer.MAX_LENGTH, 'overlength': 'First768 plus last768 tokens; count and report every truncation',
        'observer_inputs': 'Question and paid current response only; no system message, root/future text, references, generated answers or retrieval',
        'numeric_features': '110 visible features, excluding all20 previous neural outputs',
        'features_sha256': fixed.array_sha(base), 'conversation_sha256': fixed.fingerprint(conversations),
        'records_sha256': fixed.fingerprint(data['records']), 'labels_sha256': fixed.array_sha(data['labels']),
        'training_questions': 1280, 'observations': 3840, 'outer_folds': 3, 'inner_folds': 3,
        'settings': list(observer.SETTINGS), 'selection': 'Minimum inner whole-question joint log loss',
        'standardization_PCA_prior_and_heads_fit_only_training_folds': True,
        'screen': {'conditional_accuracy_gain_at_least': .02, 'compute_ratio_at_most': 1.10},
        'compute': 'Paid solver prefix plus chosen terminal continuation plus actual reward model parameters times current input tokens /3.09; replaces previous observer',
        'batch_padding': 'Offline batching only; inference proxy uses non-padding tokens as per-request inference',
        'scope': 'Conditional TRAIN feasibility, not complete POMDP/controller/fresh validation',
        'pretraining_overlap': 'Not independently established; pretrained weights are not claimed disjoint from benchmark sources',
        'cloud_calls': 0, 'cloud_training': False, 'solver_unchanged': True,
        'development_or_test_read': False, 'full_model_fitted': False}
    return rows, trees, data, base, conversations, slots, control, old._recipe['parameters'], record


def collect(args, conversations, record):
    target = args.output/'features'; target.mkdir(exist_ok=True)
    done_path = target/'complete.json'
    if done_path.exists():
        done = io.read(done_path)
        if done['registration_sha256'] != args.registration_sha256: raise ValueError('Changed feature registration')
        io.verify({str(target/k): v for k, v in done['digests'].items()}); return
    torch.set_num_threads(2)
    model, tokenizer = observer.load(args.assets/'model', args.device)
    parameters = sum(p.numel() for p in model.parameters())
    entries = observer.tokenize(tokenizer, conversations)
    scores = np.full(len(entries), np.nan); hidden = np.full((len(entries), 2048), np.nan, dtype=np.float32)
    lengths = np.asarray([len(r['input_ids']) for r in entries]); started = time.monotonic()
    # Atomic chunks allow reuse of complete local inference without rerunning it.
    for offset in range(0, len(entries), 32):
        stop = min(offset+32, len(entries)); path = target/f'chunk{offset:04}.npz'
        fingerprint = fixed.fingerprint(entries[offset:stop])
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if str(archive['input_sha256']) != fingerprint or str(archive['registration_sha256']) != args.registration_sha256:
                    raise ValueError('Changed checkpoint inputs')
                scores[offset:stop] = archive['scores']; hidden[offset:stop] = archive['hidden']
        else:
            for start in range(offset, stop, 4):
                end = min(start+4, stop)
                scores[start:end], hidden[start:end] = observer.score_batch(model, tokenizer, entries[start:end], args.device)
            if not np.isfinite(scores[offset:stop]).all() or not np.isfinite(hidden[offset:stop]).all():
                raise ValueError('Nonfinite frozen observations')
            io.npz(path, {'scores': scores[offset:stop], 'hidden': hidden[offset:stop],
                'input_sha256': np.asarray(fingerprint), 'registration_sha256': np.asarray(args.registration_sha256)})
        print({'scored': stop, 'total': len(entries), 'seconds': round(time.monotonic()-started, 1)}, flush=True)
    io.npz(target/'all.npz', {'scores': scores, 'hidden': hidden, 'tokens': lengths})
    io.write(target/'metadata.json', {'parameters': parameters, 'observations': len(entries),
        'truncated': sum(r['truncated'] for r in entries), 'input_tokens': int(lengths.sum()),
        'original_input_tokens': sum(r['original_tokens'] for r in entries), 'seconds': time.monotonic()-started,
        'dtype': 'float32', 'model_revision': observer.REVISION})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(done_path, {'registration_sha256': args.registration_sha256,
        'digests': {p.name: io.sha(p) for p in target.iterdir() if p.is_file()}})
    del model; gc.collect()
    if args.device == 'mps': torch.mps.empty_cache()


def choose(data, base, scores, hidden, train):
    folds = np.asarray([nested.inner_fold(r['id']) for r in data['records']]); reports = []
    for setting in observer.SETTINGS:
        predicted = np.full((len(train), 16), np.nan)
        for fold in range(3):
            fit, held = train & (folds != fold), train & (folds == fold)
            nested.group_receipt(data['records'], fit, held)
            model = observer.fit(base[fit], scores[fit], hidden[fit], data['labels'][fit], setting)
            predicted[held] = model.predict(base[held], scores[held], hidden[held], data['valid'][held])
        if not np.isfinite(predicted[train]).all() or np.isfinite(predicted[~train]).any(): raise ValueError('Inner scope changed')
        loss = float(fixed.losses(data['labels'][train], predicted[train])['joint_log_loss'].mean())
        reports.append({'setting': setting, 'joint_log_loss': loss})
    return min(reports, key=lambda x: (x['joint_log_loss'], x['setting']))['setting'], reports


def diagnose(args, rows, trees, data, base, conversations, slots, control, old_parameters, record):
    directory = args.output/'features'; done = io.read(directory/'complete.json')
    if done['registration_sha256'] != args.registration_sha256: raise ValueError('Changed feature registration')
    io.verify({str(directory/k): v for k, v in done['digests'].items()})
    meta = io.read(directory/'metadata.json')
    with np.load(directory/'all.npz', allow_pickle=False) as archive:
        scores, hidden, lengths = (archive[k].copy() for k in ('scores', 'hidden', 'tokens'))
    folds = np.asarray([r['fold'] for r in data['records']]); predicted = np.full((len(base), 16), np.nan); receipts = []
    for fold in range(3):
        train, held = folds != fold, folds == fold
        receipt = nested.group_receipt(data['records'], train, held)
        setting, search = choose(data, base, scores, hidden, train)
        model = observer.fit(base[train], scores[train], hidden[train], data['labels'][train], setting)
        predicted[held] = model.predict(base[held], scores[held], hidden[held], data['valid'][held])
        io.atomic_bytes(args.output/f'outer{fold}.pkl', pickle.dumps(model, protocol=5))
        receipts.append({**receipt, 'fold': fold, 'setting': setting, 'inner_search': search})
        print({'fold': fold, 'selected': setting}, flush=True)
    if not np.isfinite(predicted).all(): raise ValueError('Incomplete excluded forecasts')
    new_slots = {(r['id'], r['path']): int(n) for r, n in zip(data['records'], lengths)}
    old = conditional.conditional(rows, trees, data, slots, control, old_parameters)
    new = conditional.conditional(rows, trees, data, new_slots, predicted, meta['parameters'])
    metrics = lambda values: {'n': len(values), 'correct': sum(int(v['correct']) for v in values),
        'accuracy': float(np.mean([v['correct'] for v in values])), 'mean_compute': float(np.mean([v['compute'] for v in values]))}
    before, after = metrics(old), metrics(new); delta = after['accuracy']-before['accuracy']; ratio = after['mean_compute']/before['mean_compute']
    groups = {r['id']: [] for r in old}
    for i, r in enumerate(old): groups[r['id']].append(i)
    if len(groups) != 1280 or any(len(v) != 2 for v in groups.values()): raise ValueError('Changed paired cohort')
    differences = np.asarray([np.mean([int(new[i]['correct'])-int(old[i]['correct']) for i in group]) for group in groups.values()])
    draws = np.random.default_rng(238031).integers(0, 1280, (2000, 1280))
    screen = {'accuracy_gain': delta, 'compute_ratio': ratio, 'passed': delta >= .02 and ratio <= 1.10,
        'paired_question_accuracy_CI95': np.quantile(differences[draws].mean(1), [.025, .975]).tolist()}
    result = {'schema': SCHEMA, 'control': before, 'candidate': after, 'screen': screen, 'features': meta,
        'forecast_losses': {'control': {k: float(v.mean()) for k, v in fixed.losses(data['labels'], control).items()},
            'candidate': {k: float(v.mean()) for k, v in fixed.losses(data['labels'], predicted).items()}},
        'cloud_calls': 0, 'development_or_test_read': False, 'full_controller_claim': False,
        'fresh_validation': False, 'pretraining_overlap_not_independently_established': True}
    io.npz(args.output/'oof.npz', {'probability': predicted})
    io.write(args.output/'private-conditional-items.json', {'control': old, 'candidate': new})
    io.write(args.output/'fold-receipts.json', receipts); io.write(args.output/'summary.json', result)
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'screen_passed': screen['passed'], 'digests': {str(p.relative_to(args.output)): io.sha(p) for p in args.output.rglob('*')
            if p.is_file() and p != args.output/'complete.json' and not p.name.startswith('.')}})
    print({'control': before, 'candidate': after, 'screen': screen}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'collect', 'diagnose'))
    for name in ('expanded-fit', 'assets', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps'); parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with fixed.training_only_reads(), threadpool_limits(limits=1):
        with fixed.guarded.no_model_calls(): bundle = inputs(args)
        record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}, flush=True); return
        if not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256 or io.read(args.output/'registration.json') != record:
            raise ValueError('Changed registered reward diagnostic')
        with io.lock(args.output/'.run.lock'):
            if (args.output/'complete.json').exists(): raise ValueError('Completed experiment is immutable')
            if args.phase == 'collect': collect(args, bundle[4], record)
            else:
                if any(args.output.glob('outer*.pkl')): raise ValueError('Preserve partial diagnostic before retrying')
                with fixed.guarded.no_model_calls(): diagnose(args, *bundle)


if __name__ == '__main__': main()
