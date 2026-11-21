"""Select a correlated binary-forecast POMDP on new development questions only."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from development_repairability import wait_plain
from study_common import statistics
import continuous_repairability as policy
import factorized_repairability as factorized


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.base / 'summary.json').read_bytes())
    base_identity = json.loads((args.base / 'identity.json').read_bytes())
    if base_identity['new_train_questions_sha256'] != sha(args.data / 'splits/train-questions.jsonl'):
        raise ValueError('Neural feature cache belongs to another training split')
    train, trees, _, _ = wait_plain(args.data, 'train')
    dev, dt, _, _ = wait_plain(args.data, 'dev')
    metadata = [json.loads(s) for s in (args.base / 'training-neural.jsonl').read_text().splitlines()]
    outputs = np.load(args.base / 'training-neural.npz')
    if len(metadata) != 3 * len(train) or any(len(outputs[k]) != len(metadata) for k in ('state', 'plain', 'guided')):
        raise ValueError('Incomplete training neural cache')
    lookup = {}
    for i, r in enumerate(metadata):
        key = r['id'], r['path']
        if key in lookup or trees[r['id']][r['path']]['prompt_sha256'] != r['prompt_sha256']:
            raise ValueError('Duplicate or changed training neural input')
        lookup[key] = (outputs['state'][i], outputs['plain'][i], outputs['guided'][i], 0)
    base_model = pickle.loads((args.base / 'C1-smooth8.pkl').read_bytes())
    observer = base_model.classifier
    if sha(Path(observer.directory) / 'recipe.json') != base_identity['encoder_recipe_sha256']:
        raise ValueError('Changed old-only neural assets')
    observer.device = args.device
    identity = {'source_sha256': sha(__file__), 'policy_sha256': sha(policy.__file__),
                'factorized_sha256': sha(factorized.__file__),
                'core_sha256': sha(factorized.core.__file__),
                'base_identity_sha256': sha(args.base / 'identity.json'),
                'training_neural_sha256': sha(args.base / 'training-neural.npz'),
                'training_neural_metadata_sha256': sha(args.base / 'training-neural.jsonl'),
                'development_question_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
                'test_used': False, 'posterior_weights': [0., .5, 1.]}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    observer.load()
    candidates = []
    for regularization in (.1, 1.):
        for smoothing in (8., 32.):
            key = f'binaryC{regularization:g}-smooth{smoothing:g}'
            model = factorized.fit(train, trees, observer, lookup, smoothing, regularization)
            (args.output / (key + '.pkl')).write_bytes(pickle.dumps(model, protocol=5))
            by_subject = {s: [] for s in factorized.SUBJECTS}
            for weight in (0., .5, 1.):
                for penalty in (0., .01, .02, .04, .06, .08, .10, .16, .24):
                    settings = {'penalty': penalty, 'posterior_weight': weight}
                    values = [policy.evaluate(r, dt[r['id']], {}, model, settings, 'A') for r in dev]
                    candidates.append({'model': key, 'settings': settings, 'dev': statistics(values)})
                    for subject in by_subject:
                        subset = [v for r, v in zip(dev, values) if r['subject'] == subject]
                        by_subject[subject].append({'settings': settings, 'dev': statistics(subset)})
            settings = {'by_subject': {s: max(v, key=lambda x: (x['dev']['accuracy'], -x['dev']['mean_compute']))['settings'] for s, v in by_subject.items()}}
            values = [policy.evaluate(r, dt[r['id']], {}, model, settings, 'A') for r in dev]
            candidates.append({'model': key, 'settings': settings, 'dev': statistics(values)})
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
            print('Completed correlated-binary candidate', key, json.dumps(candidates[-1]), flush=True)
    best = max(candidates, key=lambda x: (x['dev']['accuracy'], -x['dev']['mean_compute']))
    highest = max(v['accuracy'] for v in source['baselines'].values())
    result = {'selected_A': best, 'baselines': source['baselines'], 'test_used': False,
              'development_questions': len(dev), 'fitting_questions': len(train),
              'A_above_highest_baseline': best['dev']['accuracy'] - highest,
              'A_plus_10_points': best['dev']['accuracy'] >= highest + .1 - 1e-12,
              'architecture': 'Eight-state POMDP, IPF-correlated binary forecasts, question-excluded emissions, exact finite-observation Bellman backup',
              'belief_approximation': 'Fixed blend of finite-filter and discriminative paid-history posterior'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'base', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    run(p.parse_args())
