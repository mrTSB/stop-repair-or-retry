"""CPU cached-feature fitting and fixed development selection of nonlinear A.

`fit` uses only saved training neural outputs. `develop` requires a complete
verified development feature cache and cannot run a neural model. The separate
explicit `features` phase can produce that cache when inference resources are
available; it never reads test questions or calls an LLM.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from development_repairability import wait_plain
from development_refined_observer import reference_candidate
from study import fixed
from study_common import BASELINES, statistics
import continuous_repairability as policy
import dynamic_grounded_pomdp as perception
import nonlinear_repairability as nonlinear

PENALTIES = (0., .01, .02, .04, .06, .08, .10, .16, .24)
WEIGHTS = (0., .5, 1.)
SMOOTHING = (8., 32.)
PATHS = ('', 'repair', 'retry')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def fingerprint(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def observer_from(base):
    identity = read(base / 'identity.json')
    model = pickle.loads((base / 'C1-smooth8.pkl').read_bytes())
    observer = model.classifier
    if sha(Path(observer.directory) / 'recipe.json') != identity['encoder_recipe_sha256']:
        raise ValueError('Old-only observer recipe changed')
    observer._recipe = read(Path(observer.directory) / 'recipe.json')
    observer.device = 'cpu'
    if observer._model is not None:
        raise ValueError('Fitting must use the unloaded saved observer')
    return observer


def verify_features(rows, trees, observer, metadata, outputs, slots_required=False):
    expected = {(r['id'], p) for r in rows for p in PATHS}
    if len(metadata) != len(expected):
        raise ValueError('Every question needs all three paid decision-node feature records')
    for name, width in (('state', 4), ('plain', 8), ('guided', 8)):
        if outputs[name].shape != (len(expected), width) or not np.isfinite(outputs[name]).all():
            raise ValueError('Malformed or nonfinite cached neural outputs')
    index = {}
    for i, item in enumerate(metadata):
        key = item['id'], item['path']
        if key in index or key not in expected:
            raise ValueError('Duplicate or foreign neural feature record')
        index[key] = i
    lookup, checked = {}, []
    for row in rows:
        safe = {**row, 'answer': ''}
        for path in PATHS:
            i = index[row['id'], path]; item = metadata[i]
            paid = {p: trees[row['id']][p] for p in ('', path)}
            digest = fingerprint(observer.entry(safe, path, paid))
            if (item['prompt_sha256'], item['input_sha256']) != (paid[path]['prompt_sha256'], digest):
                raise ValueError('Cached input differs from the exact paid question/response')
            slots = int(item.get('token_slots', 0))
            if slots_required and not 0 < slots <= observer._recipe['max_length']:
                raise ValueError('Development features need actual unpadded encoder token slots')
            lookup[row['id'], path] = (*[outputs[k][i] for k in ('state', 'plain', 'guided')], slots)
            checked.append({'id': row['id'], 'path': path, 'input_sha256': digest,
                            'prompt_sha256': item['prompt_sha256']})
    return lookup, checked


def fit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    base_identity = read(args.base / 'identity.json')
    train_sha = sha(args.data / 'splits/train-questions.jsonl')
    if base_identity['new_train_questions_sha256'] != train_sha or base_identity['encoder_trained_on_new_fit_questions']:
        raise ValueError('Require the old-only observer and this exact new training panel')
    rows, trees, _, _ = wait_plain(args.data, 'train')
    if len(rows) != 512:
        raise ValueError('Fit on the entire registered512question training panel')
    observer = observer_from(args.base)
    metadata = [json.loads(line) for line in (args.base / 'training-neural.jsonl').read_text().splitlines()]
    outputs = np.load(args.base / 'training-neural.npz')
    lookup, checks = verify_features(rows, trees, observer, metadata, outputs)
    identity = {'source_sha256': sha(__file__), 'nonlinear_sha256': sha(nonlinear.__file__),
        'core_sha256': sha(nonlinear.core.__file__), 'policy_sha256': sha(policy.__file__),
        'base_directory': str(args.base.resolve()), 'base_identity_sha256': sha(args.base / 'identity.json'),
        'training_neural_sha256': sha(args.base / 'training-neural.npz'),
        'training_neural_metadata_sha256': sha(args.base / 'training-neural.jsonl'),
        'train_questions_sha256': train_sha, 'dev_questions_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
        'encoder_recipe_sha256': base_identity['encoder_recipe_sha256'],
        'fitting_questions': 512, 'decision_nodes': len(checks), 'test_used': False,
        'model_family': 'small prespecified histogram gradient boosting joint-state forecasts',
        'leaf_counts': list(nonlinear.LEAVES), 'smoothing': list(SMOOTHING),
        'posterior_weights': list(WEIGHTS), 'penalties': list(PENALTIES),
        'neural_inference_during_fit': False, 'runtime_policy_module': 'continuous_repairability',
        'reference_summaries': [str(p.resolve()) for p in args.reference]}
    write(args.output / 'fitting-registration.json', identity)
    write(args.output / 'training-alignment.json', checks)
    model_hashes = {}
    for leaves in nonlinear.LEAVES:
        for smoothing in SMOOTHING:
            name = f'nonlinear-leaf{leaves}-smooth{smoothing:g}'
            model = nonlinear.fit(rows, trees, observer, lookup, smoothing, leaves)
            path = args.output / (name + '.pkl')
            path.write_bytes(pickle.dumps(model, protocol=5))
            model_hashes[path.name] = sha(path)
            print('Fitted cached-feature nonlinear A', name, flush=True)
    identity['model_sha256'] = model_hashes
    identity['training_alignment_sha256'] = sha(args.output / 'training-alignment.json')
    write(args.output / 'identity.json', identity)
    write(args.output / 'fit-complete.json', {'identity_sha256': sha(args.output / 'identity.json'),
        'models': model_hashes, 'training_only': True, 'no_neural_inference': True, 'test_used': False})


def features(args):
    if args.features is None:
        raise ValueError('Specify a new --features directory')
    args.features.mkdir(parents=True, exist_ok=False)
    rows, trees, _, _ = wait_plain(args.data, 'dev')
    if len(rows) != 256:
        raise ValueError('Use the full registered256development panel')
    observer = observer_from(args.base)
    observer.device = args.device
    observer.load()  # Only the explicit features phase is allowed neural inference.
    metadata, values = [], []
    for row in rows:
        for path in PATHS:
            paid = {p: trees[row['id']][p] for p in ('', path)}
            safe = {**row, 'answer': ''}
            result = observer.values(safe, path, paid)
            values.append(result)
            metadata.append({'id': row['id'], 'path': path, 'token_slots': int(result[3]),
                'prompt_sha256': paid[path]['prompt_sha256'], 'input_sha256': fingerprint(observer.entry(safe, path, paid))})
    np.savez_compressed(args.features / 'neural.npz', **{name: np.asarray([v[i] for v in values])
        for i, name in enumerate(('state', 'plain', 'guided'))})
    (args.features / 'metadata.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in metadata))
    write(args.features / 'complete.json', {'dev_questions_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
        'encoder_recipe_sha256': sha(Path(observer.directory) / 'recipe.json'), 'source_sha256': sha(__file__),
        'nodes': len(values), 'neural_sha256': sha(args.features / 'neural.npz'),
        'metadata_sha256': sha(args.features / 'metadata.jsonl'), 'test_used': False})


class CachedObserver(perception.Observer):
    """Development replay adapter: missing saved observations never trigger NN."""
    def load(self):
        if not self._recipe:
            raise ValueError('A cache-only observer needs its verified physical recipe')

    def values(self, row, path, tree, judge=None):
        if judge is not None:
            raise ValueError('Plain A cannot consume LLM observations')
        safe = {**row, 'answer': ''}; paid = {p: tree[p] for p in ('', path)}
        digest = fingerprint(self.entry(safe, path, paid))
        key = row['id'], path
        if key not in self.saved or self.input_sha256[key] != digest:
            raise ValueError('Uncached or changed development observation; no neural fallback allowed')
        return self.saved[key]


def develop(args):
    identity = read(args.output / 'identity.json'); complete = read(args.output / 'fit-complete.json')
    if complete['identity_sha256'] != sha(args.output / 'identity.json'):
        raise ValueError('Completed nonlinear fitting identity changed')
    if any(identity[k] != sha(p) for k, p in (('source_sha256', __file__), ('nonlinear_sha256', nonlinear.__file__),
             ('core_sha256', nonlinear.core.__file__), ('policy_sha256', policy.__file__))):
        raise ValueError('Nonlinear runtime source changed after registration')
    if (args.output / 'summary.json').exists():
        raise ValueError('Do not overwrite a completed development search')
    if args.features is None:
        raise ValueError('Development requires a complete --features cache; no implicit neural inference')
    cache = read(args.features / 'complete.json')
    if cache['dev_questions_sha256'] != identity['dev_questions_sha256'] or cache['encoder_recipe_sha256'] != identity['encoder_recipe_sha256']:
        raise ValueError('Development cache uses another panel or observer')
    if cache['neural_sha256'] != sha(args.features / 'neural.npz') or cache['metadata_sha256'] != sha(args.features / 'metadata.jsonl'):
        raise ValueError('Completed development neural outputs changed')
    rows, trees, _, _ = wait_plain(args.data, 'dev')
    if len(rows) != 256 or sha(args.data / 'splits/dev-questions.jsonl') != identity['dev_questions_sha256']:
        raise ValueError('Preserve the registered entire development panel')
    original = observer_from(args.base)
    metadata = [json.loads(line) for line in (args.features / 'metadata.jsonl').read_text().splitlines()]
    lookup, checked = verify_features(rows, trees, original, metadata, np.load(args.features / 'neural.npz'), True)
    observer = CachedObserver(original.directory, original.temperature, calibration=original.calibration)
    observer._recipe, observer._retriever, observer._backgrounds = original._recipe, original._retriever, original._backgrounds
    observer.saved = lookup
    observer.input_sha256 = {(r['id'], r['path']): r['input_sha256'] for r in checked}
    references = [reference_candidate(Path(p), identity['dev_questions_sha256']) for p in identity['reference_summaries']]
    baselines = {name: statistics([fixed(r, trees[r['id']], *settings) for r in rows]) for name, settings in BASELINES.items()}
    candidates = []
    for filename, digest in identity['model_sha256'].items():
        path = args.output / filename
        if sha(path) != digest:
            raise ValueError('Saved nonlinear POMDP changed')
        model = pickle.loads(path.read_bytes()); model.classifier = observer
        subject = {s: [] for s in ('math', 'chemistry', 'physics')}
        with threadpool_limits(limits=1):
            for weight in WEIGHTS:
                for penalty in PENALTIES:
                    settings = {'penalty': penalty, 'posterior_weight': weight}
                    values = [policy.evaluate(r, trees[r['id']], {}, model, settings, 'A') for r in rows]
                    candidates.append({'model': path.stem, 'model_path': str(path.resolve()),
                        'policy_module': 'continuous_repairability', 'fit_module': 'nonlinear_repairability',
                        'settings': settings, 'dev': statistics(values)})
                    for s in subject:
                        subject[s].append({'settings': settings, 'dev': statistics([v for r, v in zip(rows, values) if r['subject'] == s])})
            settings = {'by_subject': {s: max(group, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))['settings']
                                      for s, group in subject.items()}}
            values = [policy.evaluate(r, trees[r['id']], {}, model, settings, 'A') for r in rows]
        candidates.append({'model': path.stem, 'model_path': str(path.resolve()), 'policy_module': 'continuous_repairability',
            'fit_module': 'nonlinear_repairability', 'settings': settings, 'dev': statistics(values)})
        write(args.output / 'search.json', candidates)
        print('Evaluated cached-feature nonlinear A', path.stem, flush=True)
    best = max(candidates, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
    retained = max([best, *references], key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
    write(args.output / 'summary.json', {'selected_A': retained, 'nonlinear_A': best,
        'existing_A_references': references, 'baselines': baselines, 'test_used': False,
        'development_questions': 256, 'fitting_questions': 512,
        'development_feature_complete_sha256': sha(args.features / 'complete.json'),
        'development_feature_directory': str(args.features.resolve()),
        'A_above_highest_baseline': retained['dev']['accuracy'] - max(b['accuracy'] for b in baselines.values()),
        'neural_inference_during_development_search': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'features', 'develop'))
    for name in ('data', 'base', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--reference', type=Path, action='append', default=[])
    parser.add_argument('--device', choices=('mps', 'cpu'), default='cpu')
    args = parser.parse_args()
    {'fit': fit, 'features': features, 'develop': develop}[args.phase](args)
