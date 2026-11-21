"""Locally fit an option-correctness observer on disjoint public science QA.

No solver responses, project development labels, or test labels train this
model. The final two-epoch checkpoint is fixed in advance. Only the registered
160-question auxiliary calibration set chooses a scalar temperature.
"""

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from network_large import encode, load, predict, save
from science_auxiliary_data import PROFILE, checked_groups, examples, seal, sha

SEED = 2026112147


def margin(state):
    return torch.logsumexp(state[:, [1, 3]].float(), 1) - torch.logsumexp(state[:, [0, 2]].float(), 1)


def weighted_loss(state, labels, weights, denominator):
    point = torch.nn.functional.binary_cross_entropy_with_logits(margin(state), labels.float(), reduction='none')
    return (point * weights).sum() / denominator


def metrics(logits, samples, temperature):
    y = np.asarray([r['label'] for r in samples], dtype=float)
    w = np.asarray([r['weight'] for r in samples], dtype=float)
    z = np.asarray(logits, dtype=float) / temperature
    p = 1. / (1. + np.exp(-np.clip(z, -40, 40)))
    grouped = defaultdict(list)
    for row, score in zip(samples, z):
        grouped[row['id']].append((score, row['choice'], row['label']))
    # Ties have a fixed alphabetical rule, independent of the reference.
    rank = sum(sorted(values, key=lambda x: (-x[0], x[1]))[0][2] for values in grouped.values())
    return {'questions': len(grouped), 'options': len(samples), 'ranking_correct': int(rank),
            'ranking_accuracy': float(rank / len(grouped)),
            'balanced_bce': float(np.average(np.logaddexp(0., z) - y * z, weights=w)),
            'balanced_brier': float(np.average((p - y) ** 2, weights=w)), 'temperature': temperature}


def run(args):
    if args.output.exists():
        raise ValueError('Use a new science-observer output directory; no implicit resume')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    metadata, groups = checked_groups(args.metadata, args.cache, args.protected)
    exclusion_receipt = seal(args.metadata, args.cache, args.protected)
    initial_recipe = json.loads((args.initial / 'recipe.json').read_bytes())
    if initial_recipe.get('fold') != 'full' or initial_recipe.get('input_profile') != 'full-options-openstax-v1':
        raise ValueError('Initialize from the declared old-only full semantic observer')
    # The historical full observer may also have inherited supervised examples.
    # Its declared input digest must match the supplied legacy metadata, and all
    # auxiliary and current controller IDs must be absent from that training.
    if sha(args.legacy_examples) != initial_recipe['data_sha256']:
        raise ValueError('Initial observer training provenance does not match')
    legacy_ids = {json.loads(line)['id'] for line in args.legacy_examples.read_text().splitlines()}
    auxiliary_ids = {r['id'] for rows in groups.values() for r in rows}
    protected_ids = {json.loads(line)['id'] for split in ('train', 'dev', 'test')
                     for line in (args.protected / 'splits' / (split + '-questions.jsonl')).read_text().splitlines()}
    if legacy_ids & (auxiliary_ids | protected_ids):
        raise ValueError('Initialized observer already fitted an auxiliary or protected question')
    train = examples(groups['train'], initial_recipe['numeric_width'])
    calibration = examples(groups['calibration'], initial_recipe['numeric_width'])
    contract = {'trainer_sha256': sha(__file__), 'input_builder_sha256': sha(Path(__file__).with_name('science_auxiliary_data.py')),
                'network_sha256': sha(Path(__file__).resolve().parent.parent / 'network_large.py'),
                'input_profile': PROFILE, 'seed': SEED, 'initial_recipe_sha256': sha(args.initial / 'recipe.json'),
                'initial_training_examples_sha256': sha(args.legacy_examples), 'auxiliary_metadata_sha256': sha(args.metadata),
                'historical_exclusions_sha256': exclusion_receipt['historical_exclusions_sha256'],
                'source_revision': metadata['revision'], 'source_sha256': metadata['source_sha256'],
                'protected_split_sha256': metadata['protected_split_sha256'],
                'train_questions': len(groups['train']), 'train_options': len(train),
                'calibration_questions': len(groups['calibration']), 'calibration_options': len(calibration),
                'epochs': 2, 'batch': 16, 'max_length': 512, 'last_unfrozen_layers': 6,
                'encoder_lr': 2e-6, 'head_lr': 1e-4, 'question_normalized_class_balanced_weights': True,
                'loss': 'BCE on logsumexp(correct states)-logsumexp(incorrect states); weighted sum / fixed batch16',
                'numeric_features': 'Constant zeros; normalization reset to zero mean and unit scale',
                'transition_heads': 'Frozen and not supervised; online observer returns neutral transition logits',
                'checkpoint_selection': 'Final epoch2 only; reload archived half-precision shards before calibration',
                'temperature_selection': 'Auxiliary160 calibration balanced BCE from reloaded deployed weights only',
                'local_only': True, 'device': args.device, 'project_dev_or_test_labels_used': False,
                'solver_response_training': False, 'fold': 'full'}
    args.output.mkdir(parents=True)
    (args.output / 'training-registration.json').write_text(json.dumps(contract, indent=2) + '\n')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model, tokenizer, _ = load(args.initial)
    model.numeric_mean.zero_()
    model.numeric_std.fill_(1.)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for layer in model.encoder.encoder.layer[-6:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for head in (model.plain, model.guided):
        for p in head.parameters():
            p.requires_grad_(False)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': 2e-6},
        {'params': [p for name, p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad], 'lr': 1e-4}],
        weight_decay=.01, foreach=False)
    steps = 2 * math.ceil(len(train) / 16)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), steps)
    tensors = encode(tokenizer, train, 512)
    labels = np.asarray([r['label'] for r in train], dtype=np.float32)
    weights = np.asarray([r['weight'] for r in train], dtype=np.float32)
    started = time.monotonic()
    history = []
    for epoch in range(2):
        model.train()
        indices = np.random.default_rng(SEED + epoch).permutation(len(train))
        loss_sum = 0.
        for offset in range(0, len(indices), 16):
            ids = indices[offset:offset + 16]
            batch = {k: v[ids].to(args.device) for k, v in tensors.items()}
            y = torch.tensor(labels[ids], device=args.device)
            w = torch.tensor(weights[ids], device=args.device)
            optimizer.zero_grad(set_to_none=True)
            state, _, _ = model(**batch)
            loss = weighted_loss(state, y, w, 16)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite auxiliary training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach().cpu())
            if (offset // 16 + 1) % 50 == 0:
                print(json.dumps({'epoch': epoch + 1, 'steps': offset // 16 + 1,
                                  'total_epoch_steps': math.ceil(len(train) / 16),
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
        history.append({'epoch': epoch + 1, 'mean_loss': loss_sum / math.ceil(len(train) / 16)})
    if scheduler.last_epoch != steps:
        raise ValueError('Actual optimization steps differ from the registered schedule')
    # Save once, then calibrate the exact rounded weights that inference loads.
    # Calibration never chooses an epoch or enters gradient updates.
    directory = args.output / 'full'
    saved_recipe = save(model, tokenizer, directory, {**contract, 'temperature': 1.})
    precalibration_recipe_sha256 = sha(directory / 'recipe.json')
    del model, optimizer, scheduler, tensors, batch, state, loss
    gc.collect()
    if args.device == 'mps':
        torch.mps.empty_cache()
    archived, tokenizer, final_recipe = load(directory)
    if saved_recipe['weights'] != final_recipe['weights']:
        raise ValueError('Archived weight fingerprints changed before calibration')
    archived = archived.to(args.device)
    cal_tensors = encode(tokenizer, calibration, 512)
    states = predict(archived, cal_tensors, args.device, batch=16)[0]
    logits = margin(torch.from_numpy(states)).numpy()
    scores = [metrics(logits, calibration, t) for t in (.5, .75, 1., 1.5, 2., 3., 4., 6., 8.)]
    selected = min(scores, key=lambda r: (r['balanced_bce'], r['temperature']))
    final_recipe['temperature'] = selected['temperature']
    final_recipe['calibration_weight_sha256'] = final_recipe['weights'].copy()
    final_recipe['precalibration_recipe_sha256'] = precalibration_recipe_sha256
    # Temperature is metadata only. Never resave/round weights after measuring it.
    (directory / 'recipe.json').write_text(json.dumps(final_recipe, indent=2) + '\n')
    summary = {'contract_sha256': sha(args.output / 'training-registration.json'),
               'recipe_sha256': sha(args.output / 'full/recipe.json'), 'parameters': final_recipe['parameters'],
               'temperature_selected_on_auxiliary_calibration': selected['temperature'],
               'calibration': selected, 'temperature_search': scores, 'training_history': history,
               'calibration_used_reloaded_archive': True,
               'precalibration_recipe_sha256': precalibration_recipe_sha256,
               'calibration_weight_sha256': final_recipe['weights'],
               'training_steps': steps, 'elapsed_seconds': time.monotonic() - started,
               'project_controller_metrics': 'Not evaluated here', 'project_dev_or_test_used': False}
    (args.output / 'training-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'cache', 'protected', 'initial', 'legacy-examples', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    run(p.parse_args())
