"""Pipeline local cross-fitting with complete, prespecified paired-question blocks.

Questions are processed in registered order, never completion or accuracy order.
Each group-excluded model excludes that group in every block, including replay.
The whole training bank must finish before any model is selected on development.
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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import committed
from examples import examples
from evidence import Retriever, left_text
from network_large import load, encode, predict, save
from semantic_data import read, write

SEED = 2026112117
PLAIN = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
BASE_KEYS = {('solver', p) for p in (*PLAIN, 'guided', 'guided/repair', 'guided/retry')} | {('judge', '')}
LATE_KEYS = {(k, p + ('/reviewed' if k == 'solver' else ''))
             for p in ('repair', 'retry', 'guided') for k in ('solver', 'judge')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_records(path):
    output = defaultdict(list)
    for row in committed(path):
        output[row['id']].append(row)
    return output


def run(args):
    if args.output.exists():
        raise ValueError('Streaming refinement cannot overwrite or silently resume a partial model')
    args.output.mkdir(parents=True)
    selection = json.loads((args.data / 'splits/selection.json').read_bytes())
    question_path = args.data / 'splits/train-questions.jsonl'
    if sha(question_path) != selection['splits']['train']['sha256']:
        raise ValueError('Changed registered training questions')
    rows = read(question_path)
    old = read(args.legacy)
    if len(rows) != 512:
        raise ValueError('Use the entire registered512-question fitting set')
    if any(r['fold'] != int(hashlib.sha256(r['id'].encode()).hexdigest()[:8], 16) % 3 for r in old):
        raise ValueError('Changed original excluded-question groups')
    old_groups = defaultdict(list)
    for row in old:
        old_groups[row['id']].append(row)
    if set(old_groups) & {r['id'] for r in rows}:
        raise ValueError('Old/new training overlap')
    replay = sorted(old_groups, key=lambda q: hashlib.sha256(f'{SEED}:replay:{q}'.encode()).digest())[:256]
    tokenizer = AutoTokenizer.from_pretrained(args.initial / 'full', local_files_only=True, use_fast=False)
    retriever = Retriever()
    initial = {str(i): args.initial / 'folds' / str(i) for i in (0, 1, 2)}
    initial['full'] = args.initial / 'full'
    contract = {'trainer_sha256': sha(__file__), 'legacy_sha256': sha(args.legacy),
                'question_split_sha256': sha(question_path), 'selection_sha256': sha(args.data / 'splits/selection.json'),
                'initial_recipes': {k: sha(v / 'recipe.json') for k, v in initial.items()},
                'new_questions': len(rows), 'replay_questions': len(replay), 'block_questions': 16,
                'input_profile': 'full-options-openstax-v1', 'corpus_sha256': retriever.sha256,
                'max_length': 512, 'folds': 3, 'epochs': 1, 'batch': 8, 'seed': SEED,
                'last_unfrozen_layers': 6, 'encoder_lr': 2e-6, 'head_lr': 1e-4,
                'terminal_weight': .35, 'teacher_weight': .2, 'root_weight': 2.,
                'question_normalized_weights': True, 'binary_auxiliary_weight': .5,
                'transition_loss_weight': .35, 'class_balance': 'Within each fixed training block, separate for each excluded fold.',
                'order': 'Registered question order; every complete17-node question; old replay in fixed hash order.',
                'device': args.device, 'training_text_uploaded': False, 'development_or_test_used': False}
    (args.output / 'training-registration.json').write_text(json.dumps(contract, indent=2) + '\n')
    torch.set_num_threads(4)
    models, optimizers, schedulers = {}, {}, {}
    total_steps = {}
    for fold in (0, 1, 2, 'full'):
        count = 0
        for block in range(32):
            qs = rows[16 * block:16 * (block + 1)]
            new_n = 17 * sum(fold == 'full' or int(hashlib.sha256(r['id'].encode()).hexdigest()[:8], 16) % 3 != fold for r in qs)
            replay_n = sum(len(old_groups[q]) for q in replay[8 * block:8 * (block + 1)]
                           if fold == 'full' or int(hashlib.sha256(q.encode()).hexdigest()[:8], 16) % 3 != fold)
            count += math.ceil((new_n + replay_n) / 8)
        total_steps[fold] = count
        model, _, recipe = load(initial[str(fold)])
        if recipe['fold'] != fold:
            raise ValueError('Wrong excluded-question initialization')
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        for layer in model.encoder.encoder.layer[-6:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        model = model.to(args.device)
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': 2e-6},
            {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad], 'lr': 1e-4}],
            weight_decay=.01, foreach=False)
        models[fold], optimizers[fold] = model, optimizer
        schedulers[fold] = get_linear_schedule_with_warmup(optimizer, max(1, count // 10), count)
    workspace = args.output / 'input-blocks'
    workspace.mkdir()
    all_new = []
    for block in range(32):
        chosen = rows[16 * block:16 * (block + 1)]
        while True:
            base = grouped_records(args.data / 'train-observations.jsonl')
            late = grouped_records(args.data / 'train-late-observations.jsonl')
            if all({(n['kind'], n['path']) for n in base[r['id']]} == BASE_KEYS
                   and {(n['kind'], n['path']) for n in late[r['id']]} == LATE_KEYS for r in chosen):
                break
            time.sleep(30)
        folder = workspace / str(block)
        (folder / 'splits').mkdir(parents=True)
        path = folder / 'splits/train-questions.jsonl'
        write(path, chosen)
        (folder / 'splits/selection.json').write_text(json.dumps({'splits': {'train': {'n': len(chosen), 'sha256': sha(path)}}}))
        write(folder / 'train-observations.jsonl', [n for r in chosen for n in base[r['id']]])
        write(folder / 'train-late-observations.jsonl', [n for r in chosen for n in late[r['id']]])
        samples = examples(folder, 'train')
        by_id = {r['id']: r for r in chosen}
        contexts = {r['id']: retriever.context({**r, 'answer': ''})[0] for r in chosen}
        for item in samples:
            item['left'] = left_text(by_id[item['id']], contexts[item['id']])
        if len(samples) != 17 * len(chosen):
            raise ValueError('Missing or extra paired training nodes')
        all_new.extend(samples)
        samples += [n for q in replay[8 * block:8 * (block + 1)] for n in old_groups[q]]
        tensors = encode(tokenizer, samples, 512)
        labels = np.asarray([r['label'] for r in samples])
        folds = np.asarray([r['fold'] for r in samples])
        targets = [np.asarray([r[k] for r in samples], dtype=np.float32) for k in ('plain', 'guided')]
        weights = np.asarray([2. if r['kind'] == 'solver' and not r['path'] else
                              1. if r['kind'] == 'solver' and r['train'] else
                              .35 if r['kind'] == 'solver' else .2 for r in samples])
        weights *= np.asarray([1. if r['valid'] else .2 for r in samples])
        mass = defaultdict(float)
        for r, w in zip(samples, weights):
            mass[r['id']] += w
        weights /= np.asarray([mass[r['id']] for r in samples])
        started = time.monotonic()
        for fold in (0, 1, 2, 'full'):
            seed = SEED + 31 * block + (3 if fold == 'full' else fold)
            torch.manual_seed(seed)
            random.seed(seed)
            selected = np.ones(len(samples), dtype=bool) if fold == 'full' else folds != fold
            indices = np.flatnonzero(selected)
            balanced = weights.copy()
            for truth in (0, 1):
                group = selected & (labels % 2 == truth)
                if not group.any():
                    raise ValueError('A fixed block lacks one correctness class')
                balanced[labels % 2 == truth] *= weights[selected].sum() / (2 * weights[group].sum())
            balanced /= balanced[selected].mean()
            indices = np.random.default_rng(seed).permutation(indices)
            model, optimizer, scheduler = models[fold], optimizers[fold], schedulers[fold]
            model.train()
            for offset in range(0, len(indices), 8):
                ids = indices[offset:offset + 8]
                if fold != 'full' and np.any(folds[ids] == fold):
                    raise ValueError('A question entered its excluded fold')
                batch = {k: v[ids].to(args.device) for k, v in tensors.items()}
                y = torch.tensor(labels[ids], device=args.device)
                weight = torch.tensor(balanced[ids], dtype=torch.float32, device=args.device)
                optimizer.zero_grad(set_to_none=True)
                state, p, g = model(**batch)
                state = state.float()
                point = torch.nn.functional.cross_entropy(state, y, reduction='none')
                logit = torch.logsumexp(state[:, [1, 3]], 1) - torch.logsumexp(state[:, [0, 2]], 1)
                point += .5 * torch.nn.functional.binary_cross_entropy_with_logits(logit, (y % 2).float(), reduction='none')
                loss = (point * weight).sum() / weight.sum()
                for logits, raw in ((p, targets[0]), (g, targets[1])):
                    target = torch.tensor(raw[ids], device=args.device)
                    valid = (target >= 0).float() * weight[:, None]
                    term = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target.clamp(min=0), reduction='none')
                    loss += .35 * (term * valid).sum() / valid.sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
        print(f'Locally trained all excluded folds on registered block {block + 1}/32; seconds={time.monotonic() - started:.1f}', flush=True)
    complete = args.data / 'train-ordinary-complete.json'
    while not complete.exists():
        time.sleep(10)
    bank = json.loads(complete.read_bytes())
    if bank['base_bank_sha256'] != sha(args.data / 'train-observations.jsonl') or bank['late_bank_sha256'] != sha(args.data / 'train-late-observations.jsonl'):
        raise ValueError('Final paired bank changed')
    samples = old + all_new
    write(args.output / 'train.jsonl', samples)
    contract['data_sha256'] = sha(args.output / 'train.jsonl')
    (args.output / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    (args.output / 'training-inputs.json').write_text(json.dumps({
        'input_sha256': sha(args.output / 'train.jsonl'), 'base_bank_sha256': bank['base_bank_sha256'],
        'late_bank_sha256': bank['late_bank_sha256'], 'new_questions': len(rows), 'old_questions': len(old_groups),
        'replayed_old_ids': replay, 'steps': {str(k): v for k, v in total_steps.items()}}, indent=2) + '\n')
    # Release optimizer states before independently reloading archived weights.
    for fold in (0, 1, 2, 'full'):
        if schedulers[fold].last_epoch != total_steps[fold]:
            raise ValueError('The realized training steps differ from the registered schedule')
        save(models[fold], tokenizer, args.output / str(fold), {**contract, 'fold': fold})
    models.clear()
    optimizers.clear()
    schedulers.clear()
    del model, optimizer, scheduler, loss, state, p, g, batch, point, logit
    gc.collect()
    if args.device == 'mps':
        torch.mps.empty_cache()
    tensors = encode(tokenizer, samples, 512)
    fold_array = np.asarray([r['fold'] for r in samples])
    output = [np.zeros((len(samples), n), dtype=np.float32) for n in (4, 8, 8)]
    for fold in (0, 1, 2):
        model, _, recipe = load(args.output / str(fold))
        held = np.flatnonzero(fold_array == fold)
        values = predict(model.to(args.device), {k: v[held] for k, v in tensors.items()}, args.device, 8)
        for dest, result in zip(output, values):
            dest[held] = result
        print(f'Archived fold{fold}: {len(held)} out-of-question observations', flush=True)
        del model
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
    np.savez_compressed(args.output / 'oof.npz', state=output[0], plain=output[1], guided=output[2])
    mask = np.asarray([r['kind'] == 'solver' and r['train'] and r['valid'] for r in samples])
    labels = np.asarray([r['label'] for r in samples])
    def nll(t):
        z = output[0][mask] / t
        z -= z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), labels[mask]]))
    temperature = min((.5, .75, 1., 1.25, 1.5, 2., 3.), key=nll)
    z = output[0] / temperature
    z -= z.max(1, keepdims=True)
    probability = np.exp(z)
    probability /= probability.sum(1, keepdims=True)
    pc = probability[:, 1] + probability[:, 3]
    summary = {'temperature_selected_on_training_oof': temperature, 'questions': len(rows) + len(old_groups),
               'nodes': len(samples), 'question_grouped_folds': 3, 'fresh_data_used': False,
               'development_data_used': False,
               'oof_binary_accuracy_valid': float(np.mean((pc[mask] >= .5) == (labels[mask] % 2))),
               'oof_brier_valid': float(np.mean((pc[mask] - labels[mask] % 2) ** 2))}
    (args.output / 'training-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'legacy', 'initial', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    run(p.parse_args())
