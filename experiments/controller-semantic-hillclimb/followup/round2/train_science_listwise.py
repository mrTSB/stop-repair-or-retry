"""Question-grouped CE + balanced-BCE auxiliary science training.

New outputs only. The sanity mode uses four prespecified auxiliary TRAIN
questions and 32 updates; full mode uses the unchanged 1440/160 reservation.
Neither mode reads project development/test question or response values.
"""

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import network_large as network
import science_auxiliary_data as data
import train_science_auxiliary as original

SEED = 2026112219
SANITY_SELECTION = 'science-listwise-four-training-questions-v1'
TEMPERATURES = (.5, .75, 1., 1.5, 2., 3., 4., 6., 8.)
RECIPE = {'epochs': 3, 'questions_per_batch': 2, 'max_length': 512,
          'last_unfrozen_layers': 6, 'encoder_lr': 1e-5, 'head_lr': 5e-4,
          'gradient_clip': 1., 'weight_decay': .01, 'seed': SEED,
          'loss': 'mean_question[CE(all option correctness margins, target) + balanced BCE]',
          'balanced_bce': '0.5 * correct-option BCE + 0.5 * mean incorrect-option BCE',
          'question_weight': 'Equal, regardless of number of official options',
          'checkpoint_selection': 'Final epoch3 only; no development or calibration epoch selection',
          'temperature_selection': 'Minimum balanced BCE on the existing160 auxiliary calibration questions after archive reload',
          'input_profile': data.PROFILE, 'numeric_features': 'Constant zeros; zero mean and unit scale',
          'transition_heads': 'Frozen and not supervised; online observer returns neutral transition logits'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_registered(metadata_path, source, split):
    """Decode only the requested auxiliary split, never protected rows."""
    metadata = json.loads(metadata_path.read_bytes())
    train, calibration = metadata['train_ids'], metadata['calibration_ids']
    if len(train) != 1440 or len(calibration) != 160 or len(set(train + calibration)) != 1600:
        raise ValueError('Require the unchanged disjoint1440/160 auxiliary reservation')
    if split not in ('train', 'calibration'):
        raise ValueError('Only registered auxiliary splits are permitted')
    if sha(source) != metadata['source_sha256']:
        raise ValueError('Pinned auxiliary source changed')
    allowed = set(metadata[split + '_ids']); found = {}
    pattern = re.compile(rb'"uuid"\s*:\s*"([a-zA-Z0-9-]+)"')
    with source.open('rb') as stream:
        for line in stream:
            match = pattern.search(line)
            if match is None or match[1].decode() not in allowed:
                continue
            raw = json.loads(line); key = raw['uuid']
            options = [text.strip() for text in raw['options']]
            letter = raw['answer_letter'].strip().upper()
            if (key in found or not 4 <= len(options) <= 10 or len(set(options)) != len(options)
                    or not all(options) or len(letter) != 1 or letter not in data.LETTERS[:len(options)]):
                raise ValueError('Malformed or duplicate registered auxiliary question')
            if ' '.join(options[data.LETTERS.index(letter)].split()).casefold() != ' '.join(raw['answer'].split()).casefold():
                raise ValueError('Official letter and answer text disagree')
            found[key] = {'id': 'supergpqa-' + key, 'subject': raw['field'].casefold(),
                          'question': data.question(raw), 'options': options, 'answer': letter}
    if set(found) != allowed:
        raise ValueError('Incomplete auxiliary reservation')
    rows = [found[key] for key in metadata[split + '_ids']]
    if Counter(row['subject'] for row in rows) != {'chemistry': len(rows) // 2, 'physics': len(rows) // 2}:
        raise ValueError('Changed auxiliary subject balance')
    return rows


def select_sanity(rows):
    selected = []
    for subject in ('chemistry', 'physics'):
        group = [row for row in rows if row['subject'] == subject]
        selected.extend(sorted(group, key=lambda row: hashlib.sha256(
            (SANITY_SELECTION + ':' + row['id']).encode()).digest())[:2])
    if len(selected) != 4 or len({row['id'] for row in selected}) != 4:
        raise ValueError('Expected four distinct prespecified training questions')
    return selected


def groups(samples):
    grouped = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[sample['id']].append(index)
    result = []
    for indices in grouped.values():
        values = [samples[index] for index in indices]
        if (not 4 <= len(values) <= 10 or sum(row['label'] for row in values) != 1
                or [row['choice'] for row in values] != list(data.LETTERS[:len(values)])):
            raise ValueError('Each complete question must retain every official option in order')
        result.append(indices)
    return result


def grouped_loss(logits, labels, sizes):
    """Equal-question objective, with no free common margin offset."""
    if sum(sizes) != len(logits) or len(labels) != len(logits) or not sizes:
        raise ValueError('Incomplete grouped loss inputs')
    ranking, binary = [], []
    offset = 0
    for size in sizes:
        z, y = logits[offset:offset + size], labels[offset:offset + size]
        if size < 2 or int(y.sum().detach().cpu()) != 1 or not bool(((y == 0) | (y == 1)).all()):
            raise ValueError('Grouped targets require exactly one correct official option')
        target = torch.nonzero(y == 1, as_tuple=False).flatten()
        ranking.append(torch.nn.functional.cross_entropy(z.unsqueeze(0), target))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(z, y.float(), reduction='none')
        binary.append(.5 * bce[y == 1].sum() + .5 * bce[y == 0].mean())
        offset += size
    ce, bce = torch.stack(ranking).mean(), torch.stack(binary).mean()
    return ce + bce, ce, bce


def configure(model):
    model.numeric_mean.zero_(); model.numeric_std.fill_(1.)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for layer in model.encoder.encoder.layer[-RECIPE['last_unfrozen_layers']:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for head in (model.plain, model.guided):
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    heads = [p for name, p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
    return encoder, heads


def norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.


def scores(logits, samples, temperature=1.):
    result = original.metrics(logits, samples, temperature)
    group_indices = groups(samples)
    ordered = [index for indices in group_indices for index in indices]
    z = torch.tensor(np.asarray(logits)[ordered] / temperature, dtype=torch.float64)
    y = torch.tensor([samples[index]['label'] for index in ordered])
    total, ce, bce = grouped_loss(z, y, [len(indices) for indices in group_indices])
    result.update(joint_loss=float(total), listwise_cross_entropy=float(ce),
                  question_balanced_bce=float(bce))
    return result


def identity(args):
    metadata = json.loads(args.metadata.read_bytes())
    recipe = json.loads((args.initial / 'recipe.json').read_bytes())
    completion_path = args.initial.parent / 'training-summary.json'
    registration_path = args.initial.parent / 'training-registration.json'
    completion = json.loads(completion_path.read_bytes())
    seal_path = args.metadata.parent / 'provenance/seal-receipt.json'
    exclusions_path = args.metadata.parent / 'provenance/historical-exclusions.json'
    seal = json.loads(seal_path.read_bytes())
    if (recipe.get('input_profile'), recipe.get('auxiliary_metadata_sha256'), recipe.get('source_sha256')) != (
            data.PROFILE, sha(args.metadata), metadata['source_sha256']):
        raise ValueError('Initialize from the existing auxiliary option checkpoint with exact split provenance')
    if (recipe.get('input_builder_sha256'), recipe.get('network_sha256'), recipe.get('fold'),
            recipe.get('protected_split_sha256')) != (sha(data.__file__), sha(network.__file__), 'full',
                                                      metadata['protected_split_sha256']):
        raise ValueError('Initial checkpoint input/runtime/exclusion semantics changed')
    if (completion.get('recipe_sha256'), completion.get('contract_sha256'),
            completion.get('calibration_weight_sha256'), completion.get('calibration_used_reloaded_archive'),
            recipe.get('epochs')) != (sha(args.initial / 'recipe.json'), sha(registration_path),
                                      recipe['weights'], True, 2):
        raise ValueError('Require the completed final two-epoch auxiliary v1 checkpoint')
    if (seal['metadata_sha256'], seal['source_sha256'], seal['historical_exclusions_sha256'],
            recipe['historical_exclusions_sha256'], seal['protected_overlap']) != (
            sha(args.metadata), metadata['source_sha256'], sha(exclusions_path), sha(exclusions_path), 0):
        raise ValueError('Auxiliary exclusion receipt changed')
    for name, digest in recipe['weights'].items():
        if sha(args.initial / name) != digest:
            raise ValueError('Initial checkpoint weight changed')
    files = sorted(path for path in args.initial.iterdir() if path.is_file())
    return {'recipe': recipe, 'metadata_sha256': sha(args.metadata),
            'initial_completion_sha256': sha(completion_path),
            'initial_training_registration_sha256': sha(registration_path),
            'seal_receipt_sha256': sha(seal_path), 'historical_exclusions_sha256': sha(exclusions_path),
            'initial_files': {path.name: sha(path) for path in files},
            'source_sha256': metadata['source_sha256'], 'source_revision': metadata['revision'],
            'protected_split_sha256': metadata['protected_split_sha256']}


def run(args):
    if args.output.exists():
        raise ValueError('Use a new output area; never implicitly repeat a completed run')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    initial = identity(args)
    rows = load_registered(args.metadata, args.source, 'train')
    if args.mode == 'sanity':
        rows = select_sanity(rows)
    samples = data.examples(rows, initial['recipe']['numeric_width']); grouped = groups(samples)
    steps = 32 if args.mode == 'sanity' else RECIPE['epochs'] * math.ceil(len(grouped) / RECIPE['questions_per_batch'])
    args.output.mkdir(parents=True)
    inputs = args.output / 'training-inputs.jsonl'
    inputs.write_text(''.join(json.dumps(row) + '\n' for row in samples))
    sources = {str(Path(path).resolve()): sha(path) for path in (__file__, data.__file__, network.__file__, original.__file__)}
    registration = {**RECIPE, 'schema': 'science-grouped-loss-v1', 'mode': args.mode,
        'training_questions': len(rows), 'training_options': len(samples), 'steps': steps,
        'warmup_steps': max(1, steps // 10), 'lr_schedule': 'Linear warmup, then linear decay to zero',
        'sanity_selection_rule': SANITY_SELECTION if args.mode == 'sanity' else None,
        'selected_training_ids': [row['id'] for row in rows], 'initial_directory': str(args.initial.resolve()),
        'initial_recipe_sha256': initial['initial_files']['recipe.json'],
        'initial_completion_sha256': initial['initial_completion_sha256'],
        'initial_training_registration_sha256': initial['initial_training_registration_sha256'],
        'initial_files': initial['initial_files'], 'initial_weights': initial['recipe']['weights'],
        'auxiliary_metadata_sha256': initial['metadata_sha256'], 'source_sha256': initial['source_sha256'],
        'source_revision': initial['source_revision'], 'protected_split_sha256': initial['protected_split_sha256'],
        'seal_receipt_sha256': initial['seal_receipt_sha256'],
        'historical_exclusions_sha256': initial['historical_exclusions_sha256'],
        'training_inputs_sha256': sha(inputs), 'source_files': sources,
        'device': args.device, 'local_only': True, 'fold': 'full',
        'initialization': 'Continuation from copied final auxiliary v1 checkpoint; old checkpoint is immutable',
        'sanity_evaluation_steps': [0, 8, 16, 24, 32] if args.mode == 'sanity' else [],
        'auxiliary_calibration_authorized_after_final_archive': args.mode == 'train',
        'no_project_development_test_question_response_values_read': True,
        'runtime': {name: version(name) for name in ('torch', 'transformers', 'numpy', 'safetensors')},
        'execution_requested': args.execute, 'inference_started': False}
    if args.mode == 'sanity':
        registration['epochs'] = 16
        registration['proposed_full_training_epochs'] = RECIPE['epochs']
        registration['checkpoint_selection'] = 'No deployable checkpoint; fixed32-step diagnostic only'
        registration['temperature_selection'] = 'None; auxiliary calibration is never opened'
    write(args.output / 'training-registration.json', registration)
    print(json.dumps({'registered': True, 'mode': args.mode, 'questions': len(rows), 'options': len(samples),
                      'steps': steps, 'registration_sha256': sha(args.output / 'training-registration.json'),
                      'inference_started': False}), flush=True)
    if not args.execute:
        return
    copied = args.output / 'initial-checkpoint-copy'
    shutil.copytree(args.initial, copied)
    if any(sha(copied / name) != digest for name, digest in initial['initial_files'].items()):
        raise ValueError('Checkpoint copy does not match its registered source')
    torch.set_num_threads(2); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model, tokenizer, _ = network.load(copied)
    model = model.to(args.device); encoder, heads = configure(model)
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': RECIPE['encoder_lr']},
                                  {'params': heads, 'lr': RECIPE['head_lr']}],
                                 weight_decay=RECIPE['weight_decay'], foreach=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), steps)
    tensors = network.encode(tokenizer, samples, RECIPE['max_length'])
    np.savez_compressed(args.output / 'training-encoded.npz', **{key: value.numpy() for key, value in tensors.items()})
    labels = np.asarray([row['label'] for row in samples], dtype=np.int64)
    history, evaluations, step = [], [], 0
    started = time.monotonic()

    def evaluate(at_step):
        logits = original.margin(torch.from_numpy(network.predict(model, tensors, args.device, batch=8)[0])).numpy()
        np.savez_compressed(args.output / f'sanity-step-{at_step:02d}.npz', margin=logits)
        result = {'step': at_step, **scores(logits, samples)}
        evaluations.append(result); print(json.dumps({'sanity': result}), flush=True)

    if args.mode == 'sanity':
        evaluate(0)
    epochs = math.ceil(steps / math.ceil(len(grouped) / RECIPE['questions_per_batch']))
    for epoch in range(epochs):
        order = np.random.default_rng(SEED + epoch).permutation(len(grouped))
        for start in range(0, len(order), RECIPE['questions_per_batch']):
            if step == steps:
                break
            selected = [grouped[index] for index in order[start:start + RECIPE['questions_per_batch']]]
            ids = [index for group in selected for index in group]
            model.train(); optimizer.zero_grad(set_to_none=True)
            batch = {key: value[ids].to(args.device) for key, value in tensors.items()}
            state, _, _ = model(**batch)
            total, ce, bce = grouped_loss(original.margin(state), torch.tensor(labels[ids], device=args.device),
                                        [len(group) for group in selected])
            if not bool(torch.isfinite(total)):
                raise ValueError('Nonfinite grouped training objective')
            total.backward()
            enc_norm, head_norm = norm(encoder), norm(heads)
            if not all(math.isfinite(value) and value > 0 for value in (enc_norm, head_norm)):
                raise ValueError('Missing or nonfinite encoder/head learning gradient')
            before_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['gradient_clip']).cpu())
            optimizer.step(); scheduler.step(); step += 1
            history.append({'step': step, 'epoch': epoch + 1, 'questions': len(selected), 'options': len(ids),
                            'joint_loss': float(total.detach().cpu()), 'listwise_ce': float(ce.detach().cpu()),
                            'balanced_bce': float(bce.detach().cpu()), 'encoder_gradient_norm': enc_norm,
                            'head_gradient_norm': head_norm, 'preclip_gradient_norm': before_clip})
            with (args.output / 'optimization-history.jsonl').open('a') as stream:
                stream.write(json.dumps(history[-1], allow_nan=False) + '\n')
            if args.mode == 'sanity' and step % 8 == 0:
                evaluate(step)
            elif step % 50 == 0:
                print(json.dumps({'step': step, 'steps': steps, 'epoch': epoch + 1,
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
    if step != steps or scheduler.last_epoch != steps:
        raise ValueError('Actual optimization schedule differs from registration')
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Registered runtime source changed during training')
    if any(sha(args.initial / name) != digest for name, digest in initial['initial_files'].items()):
        raise ValueError('Original checkpoint changed')
    summary = {'registration_sha256': sha(args.output / 'training-registration.json'), 'mode': args.mode,
               'training_steps': step, 'training_questions': len(rows), 'training_options': len(samples),
               'history_sha256': sha(args.output / 'optimization-history.jsonl'),
               'training_encoded_sha256': sha(args.output / 'training-encoded.npz'),
               'elapsed_seconds': time.monotonic() - started,
               'all_encoder_head_gradients_finite_nonzero': True, 'initial_checkpoint_unchanged': True,
               'project_dev_or_test_used': False, 'cloud_calls': 0}
    if args.mode == 'sanity':
        summary.update(evaluations=evaluations, calibration_used=False, saved_deployable_checkpoint=False,
                       interpretation='Four prespecified TRAIN questions and32steps only; no generalization conclusion',
                       evaluation_files={f'sanity-step-{s:02d}.npz': sha(args.output / f'sanity-step-{s:02d}.npz')
                                         for s in (0, 8, 16, 24, 32)})
    else:
        saved = network.save(model, tokenizer, args.output / 'full', {**registration, 'temperature': 1.})
        precalibration = sha(args.output / 'full/recipe.json')
        del model, optimizer, scheduler, tensors, batch, state, total, ce, bce, encoder, heads
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
        archived, tokenizer, final_recipe = network.load(args.output / 'full')
        if saved['weights'] != final_recipe['weights']:
            raise ValueError('Archive weight mismatch before calibration')
        archived = archived.to(args.device)
        cal_rows = load_registered(args.metadata, args.source, 'calibration')
        calibration = data.examples(cal_rows, final_recipe['numeric_width'])
        logits = original.margin(torch.from_numpy(network.predict(archived,
            network.encode(tokenizer, calibration, RECIPE['max_length']), args.device, batch=8)[0])).numpy()
        np.savez_compressed(args.output / 'calibration-margins.npz', margin=logits)
        search = [scores(logits, calibration, temperature) for temperature in TEMPERATURES]
        selected = min(search, key=lambda row: (row['balanced_bce'], row['temperature']))
        final_recipe.update(temperature=selected['temperature'], calibration_weight_sha256=final_recipe['weights'],
                            precalibration_recipe_sha256=precalibration)
        write(args.output / 'full/recipe.json', final_recipe)
        summary.update(calibration=selected, temperature_search=search,
                       temperature_selected_on_auxiliary_calibration=selected['temperature'],
                       recipe_sha256=sha(args.output / 'full/recipe.json'), parameters=final_recipe['parameters'],
                       calibration_weight_sha256=final_recipe['weights'], calibration_used_reloaded_archive=True,
                       calibration_margins_sha256=sha(args.output / 'calibration-margins.npz'),
                       checkpoint_selection='Final epoch3 only')
    write(args.output / 'summary.json', summary)
    if args.mode == 'train':
        write(args.output / 'training-summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'source', 'initial', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=('sanity', 'train'), required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--execute', action='store_true', help='Run the registered job; omit for preparation only')
    run(parser.parse_args())
