"""Development-only science fusion with explicitly paid complementary encoders."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from development_repairability import wait_plain
from development_refined_observer import wait_json, reference_candidate
from study import fixed
from study_common import statistics, BASELINES
from science_option_observer import from_checkpoint
from science_fusion_pomdp import FusionObserver, SubjectPlanner, evaluate, paid_prefix
import science_fusion_pomdp as fusion_policy
import repairability_pomdp as core
import continuous_repairability as continuous


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def verified_component_lookup(directory, rows, trees, observer, expected_count):
    metadata = [json.loads(line) for line in (directory / 'training-neural.jsonl').read_text().splitlines()]
    outputs = np.load(directory / 'training-neural.npz')
    if len(metadata) != expected_count:
        raise ValueError('Unexpected component metadata count')
    for name, width in (('state', 4), ('plain', 8), ('guided', 8)):
        if outputs[name].shape != (expected_count, width) or not np.isfinite(outputs[name]).all():
            raise ValueError('Wrong component neural shape or nonfinite output')
    indices = {}
    for i, row in enumerate(metadata):
        k = row['id'], row['path']
        if k in indices:
            raise ValueError('Duplicate component input identity')
        indices[k] = i
    result, checks = {}, []
    for row in rows:
        for path in ('', 'repair', 'retry'):
            k = row['id'], path
            if k not in indices:
                raise ValueError('Missing science component decision node')
            i = indices[k]; recorded = metadata[i]
            paid = paid_prefix(trees[row['id']], path)
            entry = observer.entry({**row, 'answer': ''}, path, paid)
            digest = fingerprint(entry)
            if (recorded['prompt_sha256'], recorded['input_sha256']) != (paid[path]['prompt_sha256'], digest):
                raise ValueError('Component prompt or exact neural input changed')
            result[k] = tuple(outputs[name][i] for name in ('state', 'plain', 'guided'))
            checks.append({'id': k[0], 'path': k[1], 'index': i, 'prompt_sha256': recorded['prompt_sha256'], 'input_sha256': digest})
    return result, checks


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    panel_sha = sha(args.data / 'splits/dev-questions.jsonl')
    reference_paths = list(dict.fromkeys([args.continuous / 'summary.json', args.auxiliary / 'summary.json', *args.reference]))
    identity = {'source_sha256': sha(__file__), 'fusion_sha256': sha(fusion_policy.__file__),
                'pomdp_sha256': sha(core.__file__), 'policy_sha256': sha(continuous.__file__),
                'dev_questions_sha256': panel_sha, 'train_question_sha256': sha(args.data / 'splits/train-questions.jsonl'),
                'old_features': str(args.base.resolve()), 'option_features': str(args.auxiliary.resolve()),
                'regularization': [.1, 1.], 'smoothing': [8., 32.], 'weights': [0., .5, 1.],
                'science_observers': 2, 'math_observers': 1, 'test_used': False,
                'compute': 'Both science encoder passes charged componentwise; physical totalweights recorded separately',
                'references': [str(path.resolve()) for path in reference_paths]}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    auxiliary = wait_json(args.auxiliary / 'summary.json')
    continuous_summary = wait_json(args.continuous / 'summary.json')
    old_identity = json.loads((args.base / 'identity.json').read_bytes())
    aux_identity = json.loads((args.auxiliary / 'identity.json').read_bytes())
    if old_identity['new_train_questions_sha256'] != identity['train_question_sha256'] or aux_identity['train_question_sha256'] != identity['train_question_sha256']:
        raise ValueError('Component neural caches use different training questions')
    if aux_identity['dev_question_sha256'] != panel_sha:
        raise ValueError('Auxiliary development used another panel')
    references = [reference_candidate(path, panel_sha) for path in reference_paths]
    selected = continuous_summary['selected_A']
    base_path = args.base / (selected['model'] + '.pkl')
    base = pickle.loads(base_path.read_bytes())
    base.classifier.device = args.device
    if sha(Path(base.classifier.directory) / 'recipe.json') != old_identity['encoder_recipe_sha256']:
        raise ValueError('Old rationale observer checkpoint changed')
    option = from_checkpoint(args.verifier / 'full', args.device)
    option._recipe = json.loads((args.verifier / 'full/recipe.json').read_bytes())
    if sha(args.verifier / 'full/recipe.json') != auxiliary['fit_receipt']['auxiliary_recipe_sha256']:
        raise ValueError('Auxiliary option observer checkpoint changed')
    if option._recipe['protected_split_sha256']['train'] != identity['train_question_sha256']:
        raise ValueError('Option observer was not protected from these controller training questions')
    train, trees, _, _ = wait_plain(args.data, 'train')
    dev, dt, _, _ = wait_plain(args.data, 'dev')
    science_train = [r for r in train if r['subject'] in ('chemistry', 'physics')]
    science_dev = [r for r in dev if r['subject'] in ('chemistry', 'physics')]
    if len(train) != 512 or len(dev) != 256 or len(science_train) != 256 or len(science_dev) != 128:
        raise ValueError('Use all registered paired training and development questions')
    old_values, old_checks = verified_component_lookup(args.base, science_train, trees, base.classifier, 1536)
    option_values, option_checks = verified_component_lookup(args.auxiliary, science_train, trees, option, 768)
    if set(old_values) != set(option_values):
        raise ValueError('Complementary component observations are not aligned')
    lookup = {k: (*[np.concatenate([old_values[k][i], option_values[k][i]]) for i in (0, 1, 2)], 0) for k in old_values}
    observer = FusionObserver([base.classifier, option])
    observer.load()
    component_identity = {'old_metadata_sha256': sha(args.base / 'training-neural.jsonl'),
                          'old_neural_sha256': sha(args.base / 'training-neural.npz'),
                          'option_metadata_sha256': sha(args.auxiliary / 'training-neural.jsonl'),
                          'option_neural_sha256': sha(args.auxiliary / 'training-neural.npz'),
                          'old_checks': old_checks, 'option_checks': option_checks,
                          'science_questions': 256, 'decision_nodes': 768, 'observer': observer.provenance()}
    (args.output / 'component-alignment.json').write_text(json.dumps(component_identity, indent=2) + '\n')
    baselines = {name: statistics([fixed(r, dt[r['id']], *s) for r in dev]) for name, s in BASELINES.items()}
    old_settings = selected['settings'].get('by_subject', {}).get('math', selected['settings'])
    math_items = {r['id']: continuous.evaluate(r, dt[r['id']], {}, base, old_settings, 'A') for r in dev if r['subject'] == 'math'}
    candidates, fitted = [], {}
    for reg in (.1, 1.):
        for smooth in (8., 32.):
            name = f'fusion-C{reg:g}-smooth{smooth:g}'
            model = core.fit(science_train, trees, observer, lookup, smoothing=smooth, regularization=reg)
            model.fit_receipt.update(science_observer_fusion=observer.provenance(),
                                    component_alignment_sha256=sha(args.output / 'component-alignment.json'))
            fitted[name] = model
            (args.output / (name + '.pkl')).write_bytes(pickle.dumps(model, protocol=5))
            by_subject = {s: [] for s in ('chemistry', 'physics')}
            def evaluate_setting(setting):
                settings = {'math': old_settings, **{s: setting.get('by_subject', {}).get(s, setting) for s in by_subject}}
                wrapper = SubjectPlanner({'math': base, 'chemistry': model, 'physics': model}, settings, {})
                sciences = {r['id']: evaluate(r, dt[r['id']], {}, wrapper, {}, 'A') for r in science_dev}
                return [math_items[r['id']] if r['subject'] == 'math' else sciences[r['id']] for r in dev]
            for weight in (0., .5, 1.):
                for penalty in (0., .01, .02, .04, .06, .08, .10, .16, .24):
                    settings = {'penalty': penalty, 'posterior_weight': weight}
                    items = evaluate_setting(settings)
                    candidates.append({'model': name, 'settings': settings, 'dev': statistics(items)})
                    for subject in by_subject:
                        by_subject[subject].append({'settings': settings, 'dev': statistics([v for r, v in zip(dev, items) if r['subject'] == subject])})
            settings = {'by_subject': {s: max(v, key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))['settings'] for s, v in by_subject.items()}}
            candidates.append({'model': name, 'settings': settings, 'dev': statistics(evaluate_setting(settings))})
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
            print('Completed complementary science observer', name, json.dumps(candidates[-1]), flush=True)
    best = max(candidates, key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
    setting = best['settings']
    settings = {'math': old_settings, **{s: setting.get('by_subject', {}).get(s, setting) for s in ('chemistry', 'physics')}}
    wrapper = SubjectPlanner({'math': base, 'chemistry': fitted[best['model']], 'physics': fitted[best['model']]}, settings,
                             {'component_alignment_sha256': sha(args.output / 'component-alignment.json'),
                              'physical_science_parameters': observer._recipe['physical_parameters'],
                              'paid_science_encoder_passes_per_decision': 2, 'no_LLM_reviews': True})
    selected_path = args.output / 'selected-fusion-A.pkl'
    selected_path.write_bytes(pickle.dumps(wrapper, protocol=5))
    best = {**best, 'model_path': str(selected_path.resolve()), 'policy_module': 'science_fusion_pomdp'}
    retained = max([best, *references], key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
    highest = max(v['accuracy'] for v in baselines.values())
    result = {'selected_A': retained, 'fusion_A': best, 'existing_A_references': references, 'baselines': baselines,
              'development_questions': 256, 'fitting_science_questions': 256, 'test_used': False,
              'A_above_highest_baseline': retained['dev']['accuracy'] - highest,
              'A_plus_10_points': retained['dev']['accuracy'] >= highest + .1 - 1e-12,
              'science_physical_parameters': observer._recipe['physical_parameters'],
              'science_parameters_per_forward_pass': observer._recipe['per_forward_parameters'],
              'science_encoder_passes_per_decision': 2, 'component_alignment_sha256': sha(args.output / 'component-alignment.json')}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'base', 'continuous', 'auxiliary', 'verifier', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--reference', type=Path, action='append', default=[])
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    run(p.parse_args())
