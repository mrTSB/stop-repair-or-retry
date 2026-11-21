"""Cross-fit a semantic observation/transition encoder using retired training data only."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from network_grounded import BASE_MODEL, REVISION, Network, encode, load, predict, save

SEED = 2026112029


def train(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    samples = [json.loads(s) for s in args.input.read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    contract = {'model':BASE_MODEL, 'revision':REVISION,
                'data_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'network_sha256':hashlib.sha256(Path(__file__).with_name('network_grounded.py').read_bytes()).hexdigest(),
                'trainer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'max_length':args.max_length, 'epochs':args.epochs, 'batch':args.batch,
                'seed':SEED, 'folds':3, 'encoder_lr':5e-6, 'head_lr':3e-4,
                'input_profile':'full-options-openstax-v1',
                'corpus_sha256':json.loads(args.input.with_suffix('.manifest.json').read_bytes())['corpus_sha256'],
                'freeze_word_embeddings':True, 'transition_loss_weight':.35}
    p = args.output / 'contract.json'
    if p.exists() and json.loads(p.read_bytes()) != contract:
        raise ValueError('Changed training contract; use a new output directory')
    p.write_text(json.dumps(contract, indent=2) + '\n')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION, cache_dir=args.cache,
                                             trust_remote_code=False, use_fast=False)
    tensors = encode(tokenizer, samples, args.max_length)
    labels = torch.tensor([r['label'] for r in samples], dtype=torch.long)
    targets = [torch.tensor([r[k] for r in samples], dtype=torch.float32) for k in ('plain', 'guided')]
    trainable = np.asarray([r['train'] for r in samples], dtype=bool)
    folds = np.asarray([r['fold'] for r in samples])
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    outputs = [np.zeros((len(samples), width), dtype=np.float32) for width in (4, 8, 8)]
    print(f'{len(samples)} nodes; {trainable.sum()} training nodes; {len({r["id"] for r in samples})} questions; device={device}', flush=True)
    keys = list(tensors)
    for fold in (0, 1, 2, 'full'):
        directory = args.output / str(fold)
        selected = trainable if fold == 'full' else trainable & (folds != fold)
        indices = np.flatnonzero(selected)
        if not (directory / 'recipe.json').exists():
            seed = SEED + (3 if fold == 'full' else fold)
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            encoder = AutoModel.from_pretrained(BASE_MODEL, revision=REVISION, cache_dir=args.cache,
                                                trust_remote_code=False)
            encoder.get_input_embeddings().weight.requires_grad_(False)
            numeric = tensors['numeric'][indices].numpy()
            model = Network(encoder, numeric.shape[1], numeric.mean(0), numeric.std(0).clip(min=.05)).to(device)
            ordinary = [p for p in model.encoder.parameters() if p.requires_grad]
            heads = [p for name,p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
            optimizer = torch.optim.AdamW([{'params':ordinary, 'lr':5e-6}, {'params':heads, 'lr':3e-4}],
                                          weight_decay=.01, foreach=False)
            dataset = TensorDataset(*(tensors[k][indices] for k in keys), labels[indices],
                                    targets[0][indices], targets[1][indices])
            loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
            total = len(loader) * args.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, max(1,total // 10), total)
            for epoch in range(args.epochs):
                started = time.monotonic(); model.train(); loss_sum = 0.; seen = 0
                for step, values in enumerate(loader):
                    batch = {k:v.to(device) for k,v in zip(keys,values[:len(keys)])}
                    y, plain, guided = [v.to(device) for v in values[len(keys):]]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                        state_logits, plain_logits, guided_logits = model(**batch)
                        loss = torch.nn.functional.cross_entropy(state_logits.float(), y)
                        for logits, target in ((plain_logits,plain), (guided_logits,guided)):
                            valid = target >= 0
                            terms = torch.nn.functional.binary_cross_entropy_with_logits(
                                logits.float(), target.clamp(min=0), reduction='none')
                            loss = loss + .35 * (terms * valid).sum() / valid.sum().clamp(min=1)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step(); scheduler.step()
                    loss_sum += float(loss.detach().cpu()) * len(y); seen += len(y)
                    if (step + 1) % 100 == 0:
                        print(f'fold{fold} epoch{epoch+1} batch{step+1}/{len(loader)} loss={loss_sum/seen:.4f}', flush=True)
                print(f'fold{fold} epoch{epoch+1}/{args.epochs}: loss={loss_sum/seen:.4f}, seconds={time.monotonic()-started:.1f}', flush=True)
            save(model, tokenizer, directory, {**contract, 'fold':fold})
            del optimizer, model, encoder
            if device == 'cuda':
                torch.cuda.empty_cache()
        model, _, recipe = load(directory)
        if any(recipe.get(k) != v for k,v in contract.items()):
            raise ValueError('Stale fold checkpoint')
        if fold != 'full':
            model = model.to(device)
            indices = np.flatnonzero(folds == fold)
            values = predict(model, {k:v[indices] for k,v in tensors.items()}, device=device, batch=args.batch)
            for destination, value in zip(outputs,values):
                destination[indices] = value
            print(f'fold{fold}: {len(indices)} out-of-question predictions from archived half-rounded weights', flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    np.savez_compressed(args.output / 'oof.npz', state=outputs[0], plain=outputs[1], guided=outputs[2])
    def nll(temperature):
        logits = outputs[0][trainable] / temperature
        shifted = logits - logits.max(1,keepdims=True)
        return float(np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(shifted)),labels.numpy()[trainable]]))
    temperature = min((.5,.75,1.,1.25,1.5,2.,3.), key=nll)
    logits = outputs[0] / temperature; logits -= logits.max(1,keepdims=True)
    probabilities = np.exp(logits); probabilities /= probabilities.sum(1,keepdims=True)
    score = probabilities[:,1] + probabilities[:,3]
    valid = np.array([r['valid'] for r in samples]) & trainable
    truth = labels.numpy() % 2
    summary = {'nodes':len(samples), 'training_nodes':int(trainable.sum()), 'questions':len({r['id'] for r in samples}),
               'question_grouped_folds':3, 'temperature_selected_on_training_oof':temperature,
               'oof_binary_accuracy_valid':float(np.mean((score[valid] >= .5) == truth[valid])),
               'oof_brier_valid':float(np.mean((score[valid] - truth[valid]) ** 2)),
               'fresh_data_used':False, 'development_data_used':False}
    summary['by_kind'] = {}
    for kind in ('solver', 'original-review', 'late-review'):
        group = valid & np.array([r['kind'] == kind for r in samples])
        if group.any():
            summary['by_kind'][kind] = {'n':int(group.sum()),
                'always_correct_accuracy':float(truth[group].mean()),
                'binary_accuracy':float(np.mean((score[group] >= .5) == truth[group])),
                'brier':float(np.mean((score[group] - truth[group]) ** 2))}
    (args.output / 'training-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.cache/semantic-encoder'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=512)
    train(parser.parse_args())
