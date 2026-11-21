"""Forward-only final-checkpoint diagnostic on32 preregistered auxiliary TRAIN questions.

No gradient updates, project development/test records, cloud calls, or
calibration re-selection. Full private inputs/encodings/logits are archived;
only aggregates are printed. Existing160-question calibration metrics stay
sealed and are compared only where the original summary supplies a metric.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import network_large as network
import science_auxiliary_data as data
import train_science_auxiliary as training

SELECTION = 'auxiliary-final-training-diagnostic-v1'


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


def registered_rows(metadata, source):
    allowed = set(metadata['train_ids'])
    if len(allowed) != 1440 or allowed & set(metadata['calibration_ids']):
        raise ValueError('Require the original disjoint auxiliary training reservation')
    rows = []
    pattern = re.compile(rb'"uuid"\s*:\s*"([a-zA-Z0-9-]+)"')
    with source.open('rb') as stream:
        for line in stream:
            match = pattern.search(line)
            if match is None or match[1].decode() not in allowed:
                continue  # Never decode unrelated/protected source records.
            raw = json.loads(line)
            letter = raw['answer_letter'].strip().upper()
            if letter not in data.LETTERS[:len(raw['options'])] or len(letter) != 1:
                raise ValueError('Invalid auxiliary training target')
            rows.append({'id': 'supergpqa-' + raw['uuid'], 'subject': raw['field'].lower(),
                'question': data.question(raw), 'options': raw['options'], 'answer': letter})
    if len(rows) != 1440 or len({r['id'] for r in rows}) != 1440:
        raise ValueError('Incomplete registered auxiliary training source')
    chosen = []
    for subject in ('chemistry', 'physics'):
        group = [r for r in rows if r['subject'] == subject]
        if len(group) != 720:
            raise ValueError('Unexpected auxiliary training subject population')
        chosen.extend(sorted(group, key=lambda r: hashlib.sha256((SELECTION + ':' + r['id']).encode()).hexdigest())[:16])
    return chosen


def additional_metrics(logits, samples, temperature):
    groups = defaultdict(list)
    for sample, logit in zip(samples, np.asarray(logits) / temperature):
        groups[sample['id']].append((float(logit), int(sample['label'])))
    margins, losses, chances = [], [], []
    for entries in groups.values():
        scores, labels = np.asarray(entries).T
        if labels.sum() != 1:
            raise ValueError('Each diagnostic question must have one official correct option')
        correct_score = float(scores[labels == 1][0])
        margins.append(correct_score - float(scores[labels == 0].max()))
        losses.append(float(np.logaddexp.reduce(scores) - correct_score))
        chances.append(1. / len(scores))
    return {'listwise_cross_entropy': float(np.mean(losses)), 'random_choice_expected_accuracy': float(np.mean(chances)),
        'correct_minus_best_wrong_margin_mean': float(np.mean(margins)),
        'correct_minus_best_wrong_margin_quantiles': dict(zip(('min', 'p10', 'median', 'p90', 'max'),
            np.quantile(margins, [0, .1, .5, .9, 1.]).tolist())),
        'option_margin_quantiles': dict(zip(('min', 'p10', 'median', 'p90', 'max'),
            np.quantile(np.asarray(logits) / temperature, [0, .1, .5, .9, 1.]).tolist()))}


def run(args):
    if args.output.exists():
        raise ValueError('Use a new diagnostic area; no implicit repeat or selection')
    metadata = json.loads(args.metadata.read_bytes())
    if sha(args.source) != metadata['source_sha256']:
        raise ValueError('Pinned auxiliary source changed')
    rows = registered_rows(metadata, args.source)
    calibration = json.loads(args.calibration_summary.read_bytes())
    recipe = json.loads((args.checkpoint / 'recipe.json').read_bytes())
    if calibration['recipe_sha256'] != sha(args.checkpoint / 'recipe.json') or recipe['input_profile'] != data.PROFILE:
        raise ValueError('Use exactly the archived final auxiliary checkpoint')
    temperature = calibration['temperature_selected_on_auxiliary_calibration']
    samples = data.examples(rows, recipe['numeric_width'])
    args.output.mkdir(parents=True)
    inputs = args.output / 'inputs.jsonl'
    inputs.write_text(''.join(json.dumps(row) + '\n' for row in samples))
    registration = {'schema': SELECTION, 'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection': [{'id': r['id'], 'subject': r['subject']} for r in rows],
        'selection_rule': '16per subject from auxiliary TRAIN, ordered by SHA256(fixed salt:question ID); no outcome filtering',
        'auxiliary_training_questions': 32, 'subject_counts': {'chemistry': 16, 'physics': 16},
        'options': len(samples), 'metadata_sha256': sha(args.metadata), 'source_sha256': metadata['source_sha256'],
        'checkpoint': str(args.checkpoint.resolve()), 'recipe_sha256': sha(args.checkpoint / 'recipe.json'),
        'weights': recipe['weights'], 'calibration_summary_sha256': sha(args.calibration_summary),
        'temperature': temperature, 'temperature_reselected': False, 'inputs_sha256': sha(inputs),
        'sources': {str(Path(p).resolve()): sha(p) for p in (__file__, network.__file__, data.__file__, training.__file__)},
        'batch': 8, 'max_length': 512, 'device': args.device, 'gradient_updates': 0, 'cloud_calls': 0,
        'project_dev_test_opened': False, 'inference_started': False,
        'runtime': {name: version(name) for name in ('torch', 'transformers', 'numpy', 'safetensors')}}
    registration_path = args.output / 'registration.json'
    write(registration_path, registration)
    print(json.dumps({'registered_questions': 32, 'registered_options': len(samples),
                      'registration_sha256': sha(registration_path), 'inference_started': False}), flush=True)
    # Only after the immutable selection/inputs are written is a model loaded.
    torch.set_num_threads(2)
    model, tokenizer, loaded = network.load(args.checkpoint)
    if loaded['weights'] != registration['weights'] or sha(args.checkpoint / 'recipe.json') != registration['recipe_sha256']:
        raise ValueError('Checkpoint changed after registration')
    model = model.to(args.device)
    tensors = network.encode(tokenizer, samples, 512)
    np.savez_compressed(args.output / 'encoded.npz', **{k: v.numpy() for k, v in tensors.items()})
    outputs = network.predict(model, tensors, args.device, batch=8)
    logits = training.margin(torch.from_numpy(outputs[0])).numpy()
    np.savez_compressed(args.output / 'outputs.npz', state=outputs[0], plain=outputs[1], guided=outputs[2], binary_margin=logits)
    if any(sha(path) != digest for path, digest in registration['sources'].items()):
        raise ValueError('Diagnostic runtime changed during inference')
    write(args.output / 'inference-complete.json', {'registration_sha256': sha(registration_path),
        'inputs_sha256': sha(inputs), 'encoded_sha256': sha(args.output / 'encoded.npz'),
        'outputs_sha256': sha(args.output / 'outputs.npz'), 'questions': 32, 'options': len(samples),
        'physical_padded_encoder_slots': len(samples) * 512,
        'nonpadding_tokens': int(tensors['attention_mask'].sum()), 'gradient_updates': 0, 'project_test_used': False})
    at_temperature = {str(t): {**training.metrics(logits, samples, t), **additional_metrics(logits, samples, t)}
                      for t in sorted({1., float(temperature)})}
    deployed = at_temperature[str(float(temperature))]
    sealed = calibration['calibration']
    result = {'scope': 'Forward-only diagnostic on32 auxiliary training questions; not project A/B/C or fresh-test evidence',
        'training_sample': deployed, 'training_temperature_comparison': at_temperature,
        'sealed_auxiliary_calibration': sealed,
        'training_minus_calibration': {k: deployed[k] - sealed[k] for k in ('ranking_accuracy', 'balanced_bce', 'balanced_brier')},
        'calibration_listwise_ce_and_margin_distribution': 'Unavailable in sealed summary; no extra calibration inference performed',
        'inference_complete_sha256': sha(args.output / 'inference-complete.json'),
        'gradient_updates': 0, 'cloud_calls': 0, 'project_dev_test_opened': False,
        'interpretation_limit': 'Small prespecified training sample; uncertainty and sample difficulty can affect train/calibration gaps. Dropout-off final-weight scores, not the original dropout-on training loss.'}
    write(args.output / 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'source', 'checkpoint', 'calibration-summary', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    run(parser.parse_args())
