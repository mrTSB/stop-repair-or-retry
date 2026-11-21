"""Refine each excluded-question fold toward current-answer correctness, never test labels."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from network_large import encode, load, predict, save

SEED = 2026112059


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not (args.initial / 'training-summary.json').exists():
        raise ValueError('Finish every initial grouped fold before refinement')
    samples = [json.loads(s) for s in args.input.read_text().splitlines()]
    initial_contract = json.loads((args.initial / 'contract.json').read_bytes())
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == initial_contract['data_sha256']
    contract = {**{k: initial_contract[k] for k in ('model', 'revision', 'data_sha256', 'max_length',
                                                   'input_profile', 'corpus_sha256')},
                'seed': SEED, 'folds': 3, 'epochs': 1, 'batch': 16,
                'unfrozen_last_layers': 6, 'encoder_lr': 2e-6, 'head_lr': 1e-4,
                'root_weight': 2., 'teacher_weight': .2, 'invalid_weight': .2,
                'correctness_auxiliary_weight': .5,
                'balanced_correctness_weights': True, 'transition_loss_weight': .35,
                'trainer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'network_sha256': hashlib.sha256(Path(__file__).with_name('network_large.py').read_bytes()).hexdigest(),
                'initial_recipes': {str(fold): hashlib.sha256((args.initial / str(fold) / 'recipe.json').read_bytes()).hexdigest()
                                    for fold in (0, 1, 2, 'full')}}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / 'contract.json'
    if receipt.exists() and json.loads(receipt.read_bytes()) != contract:
        raise ValueError('Changed refinement contract')
    receipt.write_text(json.dumps(contract, indent=2) + '\n')
    tokenizer = AutoTokenizer.from_pretrained(args.initial / 'full', local_files_only=True, use_fast=False)
    tensors = encode(tokenizer, samples, contract['max_length'])
    keys = list(tensors)
    labels = torch.tensor([r['label'] for r in samples], dtype=torch.long)
    targets = [torch.tensor([r[k] for r in samples], dtype=torch.float32) for k in ('plain', 'guided')]
    trainable = np.array([r['train'] for r in samples])
    folds = np.array([r['fold'] for r in samples])
    base_weights = np.array([2. if r['kind'] == 'solver' and not r['path']
                             else 1. if r['kind'] == 'solver' else .2 for r in samples])
    base_weights *= np.array([1. if r['valid'] else .2 for r in samples])
    truth = labels.numpy() % 2
    outputs = [np.zeros((len(samples), width), dtype=np.float32) for width in (4, 8, 8)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for fold in (0, 1, 2, 'full'):
        directory = args.output / str(fold)
        selected = trainable if fold == 'full' else trainable & (folds != fold)
        if not (directory / 'recipe.json').exists():
            torch.manual_seed(SEED + (3 if fold == 'full' else fold))
            np.random.seed(SEED + (3 if fold == 'full' else fold))
            random.seed(SEED + (3 if fold == 'full' else fold))
            model, _, initial = load(args.initial / str(fold))
            assert initial['fold'] == fold
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            for layer in model.encoder.encoder.layer[-6:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            model = model.to(device)
            ordinary = [p for p in model.encoder.parameters() if p.requires_grad]
            heads = [p for name, p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
            optimizer = torch.optim.AdamW([{'params': ordinary, 'lr': 2e-6}, {'params': heads, 'lr': 1e-4}],
                                          weight_decay=.01, foreach=False)
            indices = np.flatnonzero(selected)
            weights = base_weights.copy()
            for c in (0, 1):
                mass = weights[selected & (truth == c)].sum()
                weights[truth == c] *= base_weights[selected].sum() / (2 * mass)
            weights /= weights[selected].mean()
            dataset = TensorDataset(*(tensors[k][indices] for k in keys), labels[indices], targets[0][indices],
                                    targets[1][indices], torch.tensor(weights[indices], dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(SEED + (3 if fold == 'full' else fold)))
            scheduler = get_linear_schedule_with_warmup(optimizer, max(1, len(loader) // 10), len(loader))
            started = time.monotonic()
            model.train()
            for step, values in enumerate(loader):
                batch = {k: v.to(device) for k, v in zip(keys, values[:len(keys)])}
                y, plain, guided, weight = [v.to(device) for v in values[len(keys):]]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                    state, p, g = model(**batch)
                    state = state.float()
                    point = torch.nn.functional.cross_entropy(state, y, reduction='none')
                    c_logit = torch.logsumexp(state[:, [1, 3]], dim=1) - torch.logsumexp(state[:, [0, 2]], dim=1)
                    point += .5 * torch.nn.functional.binary_cross_entropy_with_logits(
                        c_logit, (y % 2).float(), reduction='none')
                    loss = (point * weight).sum() / weight.sum()
                    for logits, target in ((p, plain), (g, guided)):
                        valid = (target >= 0).float() * weight[:, None]
                        terms = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits.float(), target.clamp(min=0), reduction='none')
                        loss += .35 * (terms * valid).sum() / valid.sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                if (step + 1) % 100 == 0:
                    print(f'refine fold{fold} batch{step+1}/{len(loader)} loss={float(loss.detach().cpu()):.4f}', flush=True)
            print(f'refine fold{fold} seconds={time.monotonic()-started:.1f}', flush=True)
            save(model, tokenizer, directory, {**contract, 'fold': fold})
            del optimizer, model
            if device == 'cuda':
                torch.cuda.empty_cache()
        model, _, recipe = load(directory)
        assert all(recipe.get(k) == v for k, v in contract.items())
        if fold != 'full':
            indices = np.flatnonzero(folds == fold)
            values = predict(model.to(device), {k: v[indices] for k, v in tensors.items()}, device=device, batch=16)
            for destination, value in zip(outputs, values):
                destination[indices] = value
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    np.savez_compressed(args.output / 'oof.npz', state=outputs[0], plain=outputs[1], guided=outputs[2])
    solver = np.array([r['kind'] == 'solver' for r in samples]) & trainable
    valid = np.array([r['valid'] for r in samples])
    calibration = solver & valid

    def nll(temperature):
        z = outputs[0][calibration] / temperature
        z -= z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), labels.numpy()[calibration]]))

    temperature = min((.5, .75, 1., 1.25, 1.5, 2., 3.), key=nll)
    z = outputs[0] / temperature
    z -= z.max(1, keepdims=True)
    probabilities = np.exp(z)
    probabilities /= probabilities.sum(1, keepdims=True)
    pc = probabilities[:, 1] + probabilities[:, 3]
    summary = {'nodes': len(samples), 'training_nodes': int(trainable.sum()),
               'questions': len({r['id'] for r in samples}), 'question_grouped_folds': 3,
               'temperature_selected_on_training_oof': temperature,
               'temperature_cohort': 'valid solver decision checkpoints',
               'fresh_data_used': False, 'development_data_used': False, 'by_kind': {}}
    for name, mask in (('solver', solver & valid),
                       ('solver-root', solver & valid & np.array([not r['path'] for r in samples])),
                       ('original-review', trainable & valid & np.array([r['kind'] == 'original-review' for r in samples]))):
        summary['by_kind'][name] = {'n': int(mask.sum()), 'always_correct_accuracy': float(truth[mask].mean()),
                                    'binary_accuracy': float(np.mean((pc[mask] >= .5) == truth[mask])),
                                    'brier': float(np.mean((pc[mask] - truth[mask]) ** 2))}
    summary['oof_binary_accuracy_valid'] = summary['by_kind']['solver']['binary_accuracy']
    summary['oof_brier_valid'] = summary['by_kind']['solver']['brier']
    (args.output / 'training-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
