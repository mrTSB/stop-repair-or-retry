"""Fit action-fixability states from complete new pairs, select only on development."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary
from semantic_data import read
from study import fixed
from study_common import BASELINES, statistics
from network_large import encode, predict
import dynamic_grounded_pomdp as perception
import repairability_pomdp as semantic


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_plain(data, split):
    path = data / (split + '-plain-complete.json')
    while not path.exists():
        time.sleep(30)
    receipt = json.loads(path.read_bytes())
    raw = (data / (split + '-observations.jsonl')).read_bytes()
    prefix = b''.join(raw.splitlines(keepends=True)[:receipt['base_records']])
    if hashlib.sha256(prefix).hexdigest() != receipt['base_bank_sha256']:
        raise ValueError('The completed plain response prefix changed')
    return ordinary(data, split)


def run(args):
    if args.output.exists():
        raise ValueError('Use a new action-fixability development directory')
    args.output.mkdir(parents=True)
    legacy_ids = {r['id'] for r in read(args.legacy_examples)}
    train_ids = {r['id'] for r in read(args.data / 'splits/train-questions.jsonl')}
    if legacy_ids & train_ids:
        raise ValueError('This fast encoder candidate must not have fitted any new training question')
    recipe = json.loads((args.verifier / 'full/recipe.json').read_bytes())
    if recipe['data_sha256'] != sha(args.legacy_examples):
        raise ValueError('Use the declared old-only frozen neural encoder')
    identity = {'source_sha256': sha(__file__), 'pomdp_sha256': sha(semantic.__file__),
                'numeric_consistency_sha256': sha(Path(__file__).with_name('numeric_consistency.py')),
                'encoder_recipe_sha256': sha(args.verifier / 'full/recipe.json'),
                'old_only_encoder_inputs_sha256': sha(args.legacy_examples),
                'new_train_questions_sha256': sha(args.data / 'splits/train-questions.jsonl'),
                'development_questions_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
                'test_used': False, 'encoder_trained_on_new_fit_questions': False,
                'selection': 'Fixability model fits the full new512plain paired questions; only new256development outcomes select settings.'}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    train, trees, _, _ = wait_plain(args.data, 'train')
    observer = perception.Observer(str((args.verifier / 'full').resolve()),
        json.loads((args.verifier / 'training-summary.json').read_bytes())['temperature_selected_on_training_oof'], device=args.device)
    observer.load()
    entries, keys, metadata = [], [], []
    for row in train:
        safe = {**row, 'answer': ''}
        for path in ('', 'repair', 'retry'):
            paid = {p: trees[row['id']][p] for p in ('', path)}
            entry = observer.entry(safe, path, paid)
            entries.append(entry)
            keys.append((row['id'], path))
            metadata.append({'id': row['id'], 'path': path,
                             'prompt_sha256': paid[path]['prompt_sha256'],
                             'input_sha256': hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()})
    tensors = encode(observer._tokenizer, entries, observer._recipe['max_length'])
    result = predict(observer._model, tensors, args.device, batch=8)
    lookup = {key: (*[r[i] for r in result], int(tensors['attention_mask'][i].sum())) for i, key in enumerate(keys)}
    np.savez_compressed(args.output / 'training-neural.npz', state=result[0], plain=result[1], guided=result[2])
    (args.output / 'training-neural.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metadata))
    print(f'Encoded {len(keys)} unseen training decision nodes with the frozen old-only observer', flush=True)
    dev, dt, _, _ = wait_plain(args.data, 'dev')
    baseline = {name: statistics([fixed(r, dt[r['id']], *p) for r in dev]) for name, p in BASELINES.items()}
    candidates = []
    penalties = (0., .01, .02, .04, .06, .08, .10, .16, .24)
    for regularization in (.1, 1.):
        for smoothing in (8., 32.):
            key = f'C{regularization:g}-smooth{smoothing:g}'
            model = semantic.fit(train, trees, observer, lookup,
                                 smoothing=smoothing, regularization=regularization)
            (args.output / (key + '.pkl')).write_bytes(pickle.dumps(model, protocol=5))
            by_subject = {s: [] for s in ('math', 'chemistry', 'physics')}
            for penalty in penalties:
                settings = {'penalty': penalty}
                items = [semantic.evaluate(r, dt[r['id']], {}, model, settings, 'A') for r in dev]
                candidates.append({'model': key, 'settings': settings, 'dev': statistics(items)})
                for subject in by_subject:
                    subset = [item for row, item in zip(dev, items) if row['subject'] == subject]
                    by_subject[subject].append({'settings': settings, 'metrics': statistics(subset)})
            settings = {'by_subject': {s: max(values, key=lambda r: (r['metrics']['accuracy'], -r['metrics']['mean_compute']))['settings']
                                       for s, values in by_subject.items()}}
            items = [semantic.evaluate(r, dt[r['id']], {}, model, settings, 'A') for r in dev]
            candidates.append({'model': key, 'settings': settings, 'dev': statistics(items)})
            best = max(candidates, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
            print('Completed repairability fit', key, 'best', json.dumps(best), flush=True)
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
    best = max(candidates, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
    highest = max(v['accuracy'] for v in baseline.values())
    summary = {'questions': len(dev), 'fitting_questions': len(train), 'fresh_data_used': False,
               'baselines': baseline, 'selected_A': best,
               'A_above_highest_baseline': best['dev']['accuracy'] - highest,
               'A_plus_10_points': best['dev']['accuracy'] >= highest + .1 - 1e-12,
               'architecture': '8statePOMDP,current/repair/retry correctness hidden; paid observations; exact two-step Bellman backup',
               'test_inference_started': False}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'verifier', 'legacy-examples', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    run(p.parse_args())
