"""Adapt excluded-question observers locally, with old-question replay.

No training text or labels are uploaded. Every old and new fitting question
receives predictions from a model that excluded its question group.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from network_large import encode, load, predict, save

SEED = 2026112111


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    torch.set_num_threads(4)
    samples = [json.loads(s) for s in args.inputs.read_text().splitlines()]
    manifest = json.loads((args.inputs.parent / 'manifest.json').read_bytes())
    if manifest['input_sha256'] != sha(args.inputs) or manifest['development_or_test_used']:
        raise ValueError('Use the verified training-only input bank')
    initial = {str(i): args.initial / 'folds' / str(i) for i in (0, 1, 2)}
    initial['full'] = args.initial / 'full'
    # New GSM8K IDs are recorded by row position, not guessed by the dataset name.
    old_nodes = manifest['nodes'] - manifest['new_nodes']
    old_ids = {r['id'] for r in samples[:old_nodes]}
    new_ids = {r['id'] for r in samples[old_nodes:]}
    assert not old_ids & new_ids and len(new_ids) == manifest['new_questions']
    replay_ids = set(sorted(old_ids, key=lambda q: hashlib.sha256(f'{SEED}:replay:{q}'.encode()).digest())[:args.replay_questions])
    contract = {'trainer_sha256': sha(__file__), 'inputs_sha256': sha(args.inputs),
                'input_profile': 'full-options-openstax-v1', 'corpus_sha256': manifest['corpus_sha256'],
                'initial_recipes': {fold: sha(folder / 'recipe.json') for fold, folder in initial.items()},
                'folds': 3, 'max_length': 512, 'seed': SEED, 'epochs': args.epochs,
                'batch': args.batch, 'last_unfrozen_layers': 6, 'encoder_lr': 2e-6, 'head_lr': 1e-4,
                'new_questions': len(new_ids), 'replayed_old_questions': len(replay_ids),
                'training_questions': len(new_ids | replay_ids), 'all_fitting_questions': len(old_ids | new_ids),
                'balanced_binary_loss_weight': .5, 'transition_loss_weight': .35,
                'question_normalized_weights': True, 'terminal_weight': .35, 'review_weight': .2,
                'development_or_test_used': False, 'training_text_uploaded': False, 'device': args.device}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / 'contract.json'
    if receipt.exists() and json.loads(receipt.read_bytes()) != contract:
        raise ValueError('Changed local refinement contract')
    receipt.write_text(json.dumps(contract, indent=2) + '\n')
    tokenizer = AutoTokenizer.from_pretrained(initial['full'], local_files_only=True, use_fast=False)
    tensors = encode(tokenizer, samples, 512)
    keys = list(tensors)
    labels = np.asarray([r['label'] for r in samples])
    folds = np.asarray([r['fold'] for r in samples])
    target = [np.asarray([r[k] for r in samples], dtype=np.float32) for k in ('plain', 'guided')]
    trained = np.asarray([r['id'] in new_ids or r['id'] in replay_ids for r in samples])
    weights = np.asarray([2. if r['kind'] == 'solver' and not r['path'] else
                          1. if r['kind'] == 'solver' and r['train'] else
                          .35 if r['kind'] == 'solver' else .2 for r in samples])
    weights *= np.asarray([1. if r['valid'] else .2 for r in samples])
    mass = {}
    for row, weight in zip(samples, weights):
        mass[row['id']] = mass.get(row['id'], 0.) + weight
    weights /= np.asarray([mass[r['id']] for r in samples])
    weights /= weights[trained].mean()
    output = [np.zeros((len(samples), n), dtype=np.float32) for n in (4, 8, 8)]
    for fold in (0, 1, 2, 'full'):
        folder = args.output / str(fold)
        selected = trained if fold == 'full' else trained & (folds != fold)
        indices = np.flatnonzero(selected)
        if not (folder / 'recipe.json').exists():
            seed = SEED + (3 if fold == 'full' else fold)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model, _, recipe = load(initial[str(fold)])
            if recipe['fold'] != fold:
                raise ValueError('Wrong initial excluded-question fold')
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            for layer in model.encoder.encoder.layer[-6:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            model = model.to(args.device)
            heads = [p for n, p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
            backbone = [p for p in model.encoder.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 2e-6}, {'params': heads, 'lr': 1e-4}],
                                          weight_decay=.01, foreach=False)
            balanced = weights.copy()
            for truth in (0, 1):
                group = selected & (labels % 2 == truth)
                balanced[labels % 2 == truth] *= weights[selected].sum() / (2 * weights[group].sum())
            balanced /= balanced[selected].mean()
            dataset = TensorDataset(*(tensors[k][indices] for k in keys),
                                    torch.tensor(labels[indices]), *(torch.tensor(t[indices]) for t in target),
                                    torch.tensor(balanced[indices], dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
            steps = len(loader) * args.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), steps)
            for epoch in range(args.epochs):
                started = time.monotonic()
                model.train()
                for step, values in enumerate(loader):
                    batch = {k: v.to(args.device) for k, v in zip(keys, values[:len(keys)])}
                    y, plain, guided, weight = [v.to(args.device) for v in values[len(keys):]]
                    optimizer.zero_grad(set_to_none=True)
                    state, p, g = model(**batch)
                    state = state.float()
                    loss = torch.nn.functional.cross_entropy(state, y, reduction='none')
                    logit = torch.logsumexp(state[:, [1, 3]], 1) - torch.logsumexp(state[:, [0, 2]], 1)
                    loss += .5 * torch.nn.functional.binary_cross_entropy_with_logits(logit, (y % 2).float(), reduction='none')
                    loss = (loss * weight).sum() / weight.sum()
                    for logits, truth in ((p, plain), (g, guided)):
                        valid = (truth >= 0).float() * weight[:, None]
                        terms = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), truth.clamp(min=0), reduction='none')
                        loss += .35 * (terms * valid).sum() / valid.sum().clamp(min=1)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    scheduler.step()
                    if (step + 1) % 100 == 0:
                        print(f'local fold{fold} epoch{epoch + 1} {step + 1}/{len(loader)} loss={float(loss.detach().cpu()):.4f}', flush=True)
                print(f'local fold{fold} epoch{epoch + 1} seconds={time.monotonic() - started:.1f}', flush=True)
            save(model, tokenizer, folder, {**contract, 'fold': fold})
            del model, optimizer, heads, backbone, loader, dataset
            gc.collect()
            if args.device == 'mps':
                torch.mps.empty_cache()
        # Regenerate predictions from the archived half-rounded checkpoint.
        if fold != 'full':
            model, _, recipe = load(folder)
            if any(recipe.get(k) != v for k, v in contract.items()):
                raise ValueError('Changed refinement checkpoint')
            held = np.flatnonzero(folds == fold)
            predicted = predict(model.to(args.device), {k: v[held] for k, v in tensors.items()}, args.device, args.batch)
            for dest, values in zip(output, predicted):
                dest[held] = values
            print(f'Predicted {len(held)} excluded-question fold{fold} observations', flush=True)
            del model
            gc.collect()
            if args.device == 'mps':
                torch.mps.empty_cache()
    np.savez_compressed(args.output / 'oof.npz', state=output[0], plain=output[1], guided=output[2])
    mask = np.asarray([r['kind'] == 'solver' and r['train'] and r['valid'] for r in samples])
    def nll(t):
        z = output[0][mask] / t
        z -= z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), labels[mask]]))
    temperature = min((.5, .75, 1., 1.25, 1.5, 2., 3.), key=nll)
    z = output[0] / temperature
    z -= z.max(1, keepdims=True)
    probabilities = np.exp(z)
    probabilities /= probabilities.sum(1, keepdims=True)
    pc = probabilities[:, 1] + probabilities[:, 3]
    summary = {'temperature_selected_on_training_oof': temperature, 'questions': len(old_ids | new_ids),
               'nodes': len(samples), 'training_nodes': int(trained.sum()), 'question_grouped_folds': 3,
               'fresh_data_used': False, 'development_data_used': False, 'by_group': {}}
    for name, extra in (('all', np.ones(len(samples), dtype=bool)),
                        ('new', np.asarray([r['id'] in new_ids for r in samples])),
                        ('root', np.asarray([not r['path'] for r in samples]))):
        group = mask & extra
        summary['by_group'][name] = {'n': int(group.sum()),
                                    'binary_accuracy': float(np.mean((pc[group] >= .5) == (labels[group] % 2))),
                                    'brier': float(np.mean((pc[group] - labels[group] % 2) ** 2))}
    summary['oof_binary_accuracy_valid'] = summary['by_group']['all']['binary_accuracy']
    summary['oof_brier_valid'] = summary['by_group']['all']['brier']
    (args.output / 'training-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'initial', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--replay-questions', type=int, default=256)
    run(p.parse_args())
