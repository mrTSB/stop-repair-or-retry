"""Full local fit only after the sealed joint-observer TRAIN screen passes.

No DEV or TEST input is opened. Fit exactly the registered two-epoch observer
recipe on all1280 questions, then use excluded-question forecasts to fit the
original finite POMDP and the already studied clustered observation variant.
"""
import argparse
import gc
from pathlib import Path
import pickle
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

import train_joint_outcome_observer as screen
import joint_observer_pomdp as runtime

io, fixed, network = screen.io, screen.fixed, screen.network
SCHEMA = 'screened-joint-observer-full-TRAIN-POMDP-fit-v1'


def inputs(args):
    spec = io.read(args.screen/'registration.json')
    done = io.read(args.screen/'complete.json')
    report = io.read(args.screen/'summary.json')
    if (spec['schema'] != screen.SCHEMA or done['schema'] != screen.SCHEMA
            or done['registration_sha256'] != io.sha(args.screen/'registration.json')
            or not done['screen_passed'] or not report['screen']['passed']
            or report['full_model_fitted'] or report['development_or_test_read']
            or report['cloud_calls'] or report['fresh_validation']):
        raise ValueError('Require a completed successful TRAIN-only joint-observer screen')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.screen/n): h for n, h in done['digests'].items()})
    old = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()}, device=spec['device'])
    rows, trees, data, entries, slots, _, expected = screen.inputs(old)
    if spec != expected:
        raise ValueError('Changed screened training inputs')
    with np.load(args.screen/'oof.npz', allow_pickle=False) as archive:
        probability = archive['probability'].copy()
    if probability.shape != (3840, 16) or not np.isfinite(probability).all():
        raise ValueError('All excluded-question forecasts required')
    initial_recipe = io.read(Path(spec['initial_directory'])/'recipe.json')
    builder = runtime.runtime.perception.Observer(spec['initial_directory'], 1.)
    rebuilt = []
    for row in rows:
        for path in ('', 'repair', 'retry'):
            entry = builder.entry({**row, 'answer': ''}, path, {p: trees[row['id']][p] for p in ('', path)})
            rebuilt.append({k: entry[k] for k in ('left', 'right', 'numeric')})
    if fixed.fingerprint(rebuilt) != fixed.fingerprint(entries):
        raise ValueError('Deployable observer input differs from its training inputs')
    if builder._retriever.sha256 != initial_recipe['corpus_sha256']:
        raise ValueError('Input corpus changed')
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'],
            **{str((args.screen/n).resolve()): h for n, h in done['digests'].items()},
            str((args.screen/'complete.json').resolve()): io.sha(args.screen/'complete.json')},
        'paths': {'screen': str(args.screen.resolve()), 'output': str(args.output.resolve())},
        'initial_directory': spec['initial_directory'], 'initial_recipe_sha256': spec['initial_recipe_sha256'],
        'corpus_sha256': initial_recipe['corpus_sha256'], 'training_questions': 1280,
        'nodes': 3840, 'device': args.device, 'epochs': screen.EPOCHS, 'batch': screen.BATCH,
        'seed': screen.SEED+3, 'threads': 2, 'architecture': network.ARCHITECTURE,
        'encoder_lr': 2e-6, 'heads_lr': 1e-4, 'weight_decay': .01,
        'trainable_last_encoder_layers': 4, 'gradient_clip': 1., 'warmup_fraction': .1,
        'smoothing': 32., 'variants': ['original', 'clustered'],
        'emission_forecasts': 'Sealed question-excluded predictions from screen, never full-fit predictions',
        'forecasts_sha256': fixed.array_sha(probability), 'solver_unchanged': True,
        'cloud_calls': 0, 'cloud_training': False, 'development_or_test_read': False,
        'full_policy_or_fresh_result_claimed': False,
        'resume': 'Sealed model permits dynamics-only resume; interrupted unsealed training fails closed'}
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Local MPS unavailable')
    return rows, trees, data, entries, slots, probability, record


def fit_observer(args, data, entries, slots, record):
    checkpoint = args.output/'observer'
    receipt = args.output/'training-receipt.json'
    if receipt.exists():
        done = io.read(receipt)
        if (done['registration_sha256'] != args.registration_sha256
                or done['recipe_sha256'] != io.sha(checkpoint/'recipe.json')):
            raise ValueError('Changed sealed full observer')
        recipe = io.read(checkpoint/'recipe.json')
        io.verify({str(checkpoint/n): h for n, h in recipe['weights'].items()})
        return recipe
    attempt = args.output/'training-attempt.json'
    if attempt.exists():
        raise ValueError('Preserve and inspect interrupted unsealed full training')
    io.write(attempt, {'registration_sha256': args.registration_sha256, 'seed': record['seed']})
    torch.set_num_threads(2)
    random.seed(record['seed']); np.random.seed(record['seed']); torch.manual_seed(record['seed'])
    counts = np.bincount(data['labels'], minlength=16)+1.
    prior = counts/counts.sum()
    model, tokenizer, _ = network.from_initial(record['initial_directory'], prior)
    tensors = network.initial.encode(tokenizer, entries, 512)
    np.testing.assert_array_equal(tensors['attention_mask'].sum(1).numpy(),
        np.array([slots[r['id'], r['path']] for r in data['records']]))
    model = model.to(args.device)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': 2e-6},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad], 'lr': 1e-4}],
        weight_decay=.01, foreach=False)
    keys = list(tensors)
    dataset = TensorDataset(*(tensors[k] for k in keys), torch.tensor(data['labels']), torch.tensor(data['valid']))
    loader = DataLoader(dataset, batch_size=screen.BATCH, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(record['seed']))
    steps = len(loader)*screen.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps//10), steps)
    elapsed = []
    for epoch in range(screen.EPOCHS):
        started = time.monotonic(); model.train()
        for step, values in enumerate(loader):
            batch = dict(zip(keys, values[:-2])); length = max(1, int(batch['attention_mask'].sum(1).max()))
            batch = {k: (v[:, :length] if k != 'numeric' else v).to(args.device) for k, v in batch.items()}
            labels, valid = (v.to(args.device) for v in values[-2:])
            optimizer.zero_grad(set_to_none=True)
            loss = network.loss(model(**batch), labels, valid)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite joint-observer loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step()
            if (step+1) % 100 == 0:
                print({'epoch': epoch+1, 'step': step+1, 'steps_per_epoch': len(loader),
                    'loss': float(loss.detach().cpu())}, flush=True)
        elapsed.append(time.monotonic()-started)
        print({'epoch_complete': epoch+1, 'seconds': elapsed[-1]}, flush=True)
    recipe = network.save(model, tokenizer, checkpoint, {'schema': SCHEMA,
        'registration_sha256': args.registration_sha256, 'prior': prior.tolist(),
        'max_length': 512, 'epochs': screen.EPOCHS, 'trainable_last_encoder_layers': 4,
        'training_questions': 1280, 'input_profile': 'unchanged-full-options-openstax-v1',
        **{k: record[k] for k in ('initial_directory', 'initial_recipe_sha256', 'corpus_sha256')}})
    io.write(receipt, {'registration_sha256': args.registration_sha256,
        'recipe_sha256': io.sha(checkpoint/'recipe.json'), 'training_steps': steps,
        'epoch_seconds': elapsed, 'offline_training_excluded_from_inference_proxy': True})
    del model, optimizer, loader, dataset, scheduler, loss, batch, values
    gc.collect()
    if args.device == 'mps': torch.mps.empty_cache()
    return recipe


def fit(args, rows, trees, data, entries, slots, probability, record):
    recipe = fit_observer(args, data, entries, slots, record)
    builder = runtime.runtime.perception.Observer(record['initial_directory'], 1.)
    observer = runtime.Observer(str((args.output/'observer').resolve()), builder,
        io.sha(args.output/'observer/recipe.json'))
    variants = runtime.fit(rows, trees, observer, data['records'], probability,
        {'registration_sha256': args.registration_sha256, 'screen': str(args.screen.resolve())})
    models = {}
    for name, model in variants.items():
        path = args.output/(name+'.pkl'); io.atomic_bytes(path, pickle.dumps(model, protocol=5))
        models[name] = {'sha256': io.sha(path), 'fit_receipt': model.fit_receipt}
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'summary.json', {'schema': SCHEMA, 'models': models,
        'parameters': recipe['parameters'], 'training_questions': 1280,
        'full_model_fitted': True, 'development_or_test_read': False, 'cloud_calls': 0,
        'full_policy_or_fresh_result_claimed': False})
    io.write(args.output/'complete.json', {'schema': SCHEMA,
        'registration_sha256': args.registration_sha256, 'models': models,
        'digests': {str(p.relative_to(args.output)): io.sha(p) for p in args.output.rglob('*')
            if p.is_file() and p != args.output/'complete.json' and not p.name.startswith('.')}})
    print({'full_fit_complete': True, 'parameters': recipe['parameters'], 'models': list(models)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit'))
    for name in ('screen', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with fixed.training_only_reads():
        with fixed.guarded.no_model_calls(): bundle = inputs(args)
        record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record):
            raise ValueError('Changed full training registration')
        with io.lock(args.output/'.fit.lock'):
            if (args.output/'complete.json').exists(): raise ValueError('Preserve completed fit')
            fit(args, *bundle)


if __name__ == '__main__': main()
