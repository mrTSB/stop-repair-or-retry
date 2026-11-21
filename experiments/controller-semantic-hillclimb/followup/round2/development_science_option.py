"""Select the auxiliary science POMDP on development, preserving the math model.

Only the observer matching the observed subject executes. The full development
table always contains all256questions and the unchanged five fixed policies.
The science observer is independently trained on disjoint auxiliary questions;
its downstream POMDP fits all256current paired science training questions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from development_repairability import wait_plain
from study import fixed
from study_common import BASELINES, statistics
from network_large import encode, predict
from science_option_observer import from_checkpoint, binary_outputs
from subject_science_pomdp import SubjectPlanner, evaluate
import repairability_pomdp as core
import continuous_repairability as policy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_json(path):
    while not path.exists():
        time.sleep(30)
    return json.loads(path.read_bytes())


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(core.__file__), Path(policy.__file__),
               Path(__file__).with_name('science_option_observer.py'),
               Path(__file__).with_name('science_auxiliary_data.py'),
               Path(__file__).with_name('subject_science_pomdp.py')]
    identity = {'source_sha256': {p.name: sha(p) for p in sources},
                'train_question_sha256': sha(args.data / 'splits/train-questions.jsonl'),
                'dev_question_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
                'science_model': str(args.verifier.resolve()),
                'baseline_continuous_development': str(args.continuous.resolve()),
                'fitting': 'New256science paired train only; auxiliary observer never fitted these question IDs',
                'selection': 'Entire new256development only; math uses previously selected continuous model/settings',
                'science_settings': {'regularization': [.1, 1.], 'smoothing': [8., 32.],
                    'penalty': [0., .01, .02, .04, .06, .08, .10, .16, .24], 'posterior_weight': [0., .5, 1.]},
                'test_used': False}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    trained = wait_json(args.verifier / 'training-summary.json')
    continuous = wait_json(args.continuous / 'summary.json')
    if not trained.get('calibration_used_reloaded_archive'):
        raise ValueError('Use the archived-weight-calibrated auxiliary observer')
    if sha(args.verifier / 'full/recipe.json') != trained['recipe_sha256']:
        raise ValueError('Auxiliary observer recipe changed')
    recipe = json.loads((args.verifier / 'full/recipe.json').read_bytes())
    if any(recipe['protected_split_sha256'][split] != sha(args.data / 'splits' / (split + '-questions.jsonl'))
           for split in ('train', 'dev', 'test')):
        raise ValueError('Observer was reserved against different protected splits')
    train, trees, _, _ = wait_plain(args.data, 'train')
    dev, dev_trees, _, _ = wait_plain(args.data, 'dev')
    science_train = [r for r in train if r['subject'] in ('chemistry', 'physics')]
    science_dev = [r for r in dev if r['subject'] in ('chemistry', 'physics')]
    if len(science_train) != 256 or len(science_dev) != 128 or len(dev) != 256:
        raise ValueError('Use all registered science training/development questions')
    observer = from_checkpoint(args.verifier / 'full', args.device)
    observer.load()
    entries, keys, metadata = [], [], []
    for row in science_train:
        for path in ('', 'repair', 'retry'):
            paid = {p: trees[row['id']][p] for p in ('', path)}
            entry = observer.entry({**row, 'answer': ''}, path, paid)
            entries.append(entry)
            keys.append((row['id'], path))
            metadata.append({'id': row['id'], 'path': path,
                'prompt_sha256': paid[path]['prompt_sha256'],
                'input_sha256': hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()})
    tensors = encode(observer._tokenizer, entries, observer._recipe['max_length'])
    state = binary_outputs(predict(observer._model, tensors, args.device, batch=8)[0])
    neutral = np.zeros((len(entries), 8))
    lookup = {key: (state[i], neutral[i], neutral[i], int(tensors['attention_mask'][i].sum()))
              for i, key in enumerate(keys)}
    np.savez_compressed(args.output / 'training-neural.npz', state=state, plain=neutral, guided=neutral)
    (args.output / 'training-neural.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metadata))
    print('Encoded768paired science decision nodes with the disjoint auxiliary observer', flush=True)

    selected = continuous['selected_A']
    base_path = args.base / (selected['model'] + '.pkl')
    base = pickle.loads(base_path.read_bytes())
    base.classifier.device = args.device
    baseline_items = [policy.evaluate(r, dev_trees[r['id']], {}, base, selected['settings'], 'A') for r in dev]
    by_subject_base = {s: statistics([v for r, v in zip(dev, baseline_items) if r['subject'] == s])
                       for s in ('math', 'chemistry', 'physics')}
    fixed_metrics = {name: statistics([fixed(r, dev_trees[r['id']], *setting) for r in dev])
                     for name, setting in BASELINES.items()}
    candidates, fitted = [], {}
    best_subject = {}
    for reg in (.1, 1.):
        for smoothing in (8., 32.):
            key = f'science-C{reg:g}-smooth{smoothing:g}'
            model = core.fit(science_train, trees, observer, lookup, smoothing=smoothing, regularization=reg)
            fitted[key] = model
            (args.output / (key + '.pkl')).write_bytes(pickle.dumps(model, protocol=5))
            groups = {s: [] for s in ('chemistry', 'physics')}
            for weight in (0., .5, 1.):
                for penalty in (0., .01, .02, .04, .06, .08, .10, .16, .24):
                    setting = {'penalty': penalty, 'posterior_weight': weight}
                    items = [policy.evaluate(r, dev_trees[r['id']], {}, model, setting, 'A') for r in science_dev]
                    candidate = {'model': key, 'settings': setting, 'science_dev': statistics(items)}
                    candidates.append(candidate)
                    for subject in groups:
                        subset = [v for r, v in zip(science_dev, items) if r['subject'] == subject]
                        groups[subject].append({'model': key, 'settings': setting, 'dev': statistics(subset)})
            for subject, values in groups.items():
                best = max(values, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
                if subject not in best_subject or (best['dev']['accuracy'], -best['dev']['mean_compute']) > (
                        best_subject[subject]['dev']['accuracy'], -best_subject[subject]['dev']['mean_compute']):
                    best_subject[subject] = best
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
            print('Finished science candidate', key, json.dumps(best_subject), flush=True)

    models, settings, choices = {}, {}, {}
    for subject in ('math', 'chemistry', 'physics'):
        old_setting = selected['settings'].get('by_subject', {}).get(subject, selected['settings'])
        if subject == 'math' or (by_subject_base[subject]['accuracy'], -by_subject_base[subject]['mean_compute']) >= (
                best_subject[subject]['dev']['accuracy'], -best_subject[subject]['dev']['mean_compute']):
            models[subject], settings[subject] = base, old_setting
            choices[subject] = {'observer': 'old-semantic-v1', 'model': str(base_path.resolve()),
                                'settings': old_setting, 'dev': by_subject_base[subject]}
        else:
            chosen = best_subject[subject]
            models[subject], settings[subject] = fitted[chosen['model']], chosen['settings']
            choices[subject] = {'observer': 'auxiliary-science-option-v1',
                'model': str((args.output / (chosen['model'] + '.pkl')).resolve()),
                'settings': chosen['settings'], 'dev': chosen['dev']}
    receipt = {'selection': 'Observed subject only; one encoder per decision', 'subject_models': choices,
               'auxiliary_recipe_sha256': trained['recipe_sha256'], 'legacy_model_sha256': sha(base_path),
               'continuous_summary_sha256': sha(args.continuous / 'summary.json'),
               'science_neural_sha256': sha(args.output / 'training-neural.npz'), 'test_used': False}
    combined = SubjectPlanner(models, settings, receipt)
    items = [evaluate(r, dev_trees[r['id']], {}, combined, {}, 'A') for r in dev]
    (args.output / 'selected-A.pkl').write_bytes(pickle.dumps(combined, protocol=5))
    overall = statistics(items)
    highest = max(v['accuracy'] for v in fixed_metrics.values())
    result = {'development_questions': 256, 'paired_science_training_questions': 256,
              'selected_A': {'dev': overall, 'subject_choices': choices, 'pickle': 'selected-A.pkl'},
              'baselines': fixed_metrics, 'comparison_old_V1': {'dev': statistics(baseline_items), 'by_subject': by_subject_base},
              'auxiliary_best_science_candidates': best_subject,
              'A_above_highest_baseline': overall['accuracy'] - highest,
              'A_plus_10_points': overall['accuracy'] >= highest + .1 - 1e-12,
              'compute': 'Single observer selected by observed subject; actual per-decision slots and parameter counts charged',
              'test_used': False, 'fresh_test_inference_started': False, 'fit_receipt': receipt}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'verifier', 'base', 'continuous', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    run(p.parse_args())
