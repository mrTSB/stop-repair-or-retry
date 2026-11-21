"""Fit the new POMDP from V4 question-excluded observations, then select on dev.

Neural question exclusion and second-stage forecaster cross-fitting are
reported separately. They do not constitute nested neural/forecaster CV.
The final controller remains subject to the still-untouched test evaluation.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import committed
from semantic_data import read
from examples import examples
from evidence import Retriever, left_text
from development_repairability import wait_plain
from study import fixed
from study_common import BASELINES, statistics
import dynamic_grounded_pomdp as perception
import repairability_pomdp as core
import continuous_repairability as policy

EXCLUSION_NOTE = ('Every neural feature comes from the checkpoint excluding its question hash-fold. '
                  'The forecaster separately cross-fits by the same question groups. This is not nested CV: '
                  'held-out forecaster labels may have trained neural models producing other folds features. '
                  'Training OOF metrics are diagnostics; only held-out development/test assess the final policy.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def wait_json(path):
    while not path.exists():
        time.sleep(30)
    return json.loads(path.read_bytes())


def key(row):
    return row['id'], row['kind'], row['path']


def fold_for(qid):
    return int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 3


def aligned_lookup(metadata, outputs, expected, prompts, expected_questions):
    """Join raw OOF arrays by explicit node identity, validating all new inputs."""
    n = len(metadata)
    for name, width in (('state', 4), ('plain', 8), ('guided', 8)):
        if outputs[name].shape != (n, width) or not np.isfinite(outputs[name]).all():
            raise ValueError('Incomplete, nonfinite, or misaligned OOF output shape: ' + name)
    indices = {}
    for i, row in enumerate(metadata):
        k = key(row)
        if k in indices or row['fold'] != fold_for(row['id']):
            raise ValueError('Duplicate node or changed question-excluded fold flag')
        indices[k] = i
    expected_map = {key(r): r for r in expected}
    if len(expected_map) != len(expected) or {r['id'] for r in expected} != set(expected_questions):
        raise ValueError('Expected rebuilt input set has missing or duplicate questions/nodes')
    new_keys = {k for k in indices if k[0] in expected_questions}
    if new_keys != set(expected_map) or set(prompts) != set(expected_map):
        raise ValueError('OOF new-question nodes differ from exact registered paired metadata')
    lookup, aligned = {}, []
    for k, expected_row in expected_map.items():
        i = indices[k]; row = metadata[i]
        if fingerprint(row) != fingerprint(expected_row):
            raise ValueError('Saved neural input, label, or train flag differs from rebuilt registered input')
        entry = {name: row[name] for name in ('left', 'right', 'numeric')}
        aligned.append({'id': k[0], 'kind': k[1], 'path': k[2], 'fold': row['fold'],
                        'oof_index': i, 'prompt_sha256': prompts[k], 'input_sha256': fingerprint(entry),
                        'metadata_sha256': fingerprint(row)})
        if k[1] == 'solver' and k[2] in ('', 'repair', 'retry'):
            lookup[k[0], k[2]] = (outputs['state'][i], outputs['plain'][i], outputs['guided'][i], 0)
    if set(lookup) != {(qid, path) for qid in expected_questions for path in ('', 'repair', 'retry')}:
        raise ValueError('Missing one of the three paired decision observations')
    return lookup, aligned


def load_verified(args, train):
    summary = wait_json(args.verifier / 'training-summary.json')
    registration = json.loads((args.verifier / 'training-registration.json').read_bytes())
    contract = json.loads((args.verifier / 'contract.json').read_bytes())
    inputs = json.loads((args.verifier / 'training-inputs.json').read_bytes())
    data_sha = sha(args.verifier / 'train.jsonl')
    if contract != {**registration, 'data_sha256': data_sha} or inputs['input_sha256'] != data_sha:
        raise ValueError('Completed neural inputs differ from their registration or contract')
    if (contract['trainer_sha256'] != sha(Path(__file__).with_name('refine_stream.py'))
            or contract['question_split_sha256'] != sha(args.data / 'splits/train-questions.jsonl')
            or contract['selection_sha256'] != sha(args.data / 'splits/selection.json')
            or contract['new_questions'] != 512 or contract['folds'] != 3):
        raise ValueError('Wrong refinement source, question reservation, or fold setup')
    complete = json.loads((args.data / 'train-ordinary-complete.json').read_bytes())
    for part, filename in (('base', 'train-observations.jsonl'), ('late', 'train-late-observations.jsonl')):
        actual = sha(args.data / filename)
        if actual != complete[part + '_bank_sha256'] or actual != inputs[part + '_bank_sha256']:
            raise ValueError('Refined observer trained on another paired response bank')
    metadata = read(args.verifier / 'train.jsonl')
    if summary['nodes'] != len(metadata) or summary['questions'] != len({r['id'] for r in metadata}) or summary['question_grouped_folds'] != 3:
        raise ValueError('Neural summary shape or question count differs from metadata')
    protected_ids = {r['id'] for split in ('dev', 'test')
                     for r in read(args.data / 'splits' / (split + '-questions.jsonl'))}
    if protected_ids & {r['id'] for r in metadata}:
        raise ValueError('A protected development/test question entered neural fitting metadata')
    if summary.get('development_data_used') or summary.get('fresh_data_used'):
        raise ValueError('Refined observer must have used training data only')
    recipes = {}
    for fold in (0, 1, 2, 'full'):
        folder = args.verifier / str(fold)
        recipe = json.loads((folder / 'recipe.json').read_bytes())
        if recipe['fold'] != fold or any(recipe.get(k) != v for k, v in contract.items()):
            raise ValueError('Fold checkpoint does not match the completed training contract')
        for filename, digest in recipe['weights'].items():
            if sha(folder / filename) != digest:
                raise ValueError('Archived neural weight shard changed')
        recipes[str(fold)] = sha(folder / 'recipe.json')
    outputs = np.load(args.verifier / 'oof.npz')
    expected = examples(args.data, 'train')
    retriever = Retriever()
    if retriever.sha256 != contract['corpus_sha256']:
        raise ValueError('Grounding corpus changed')
    by_id = {r['id']: r for r in train}
    contexts = {r['id']: retriever.context({**r, 'answer': ''})[0] for r in train}
    for item in expected:
        item['left'] = left_text(by_id[item['id']], contexts[item['id']])
    prompts = {}
    for suffix, kind_name in (('observations', 'original-review'), ('late-observations', 'late-review')):
        for r in committed(args.data / ('train-' + suffix + '.jsonl')):
            k = r['id'], ('solver' if r['kind'] == 'solver' else kind_name), r['path']
            if k in prompts:
                raise ValueError('Duplicate registered prompt receipt')
            prompts[k] = r['response']['prompt_sha256']
    ids = {r['id'] for r in train}
    if len(expected) != 17 * 512 or set(Counter(r['id'] for r in expected).values()) != {17}:
        raise ValueError('All512newquestions must have all17registered neural examples')
    lookup, alignment = aligned_lookup(metadata, outputs, expected, prompts, ids)
    folds = np.asarray([r['fold'] for r in metadata])
    if set(folds) != {0, 1, 2}:
        raise ValueError('All three question-excluded folds must be represented')
    receipt = {'metadata_sha256': data_sha, 'oof_sha256': sha(args.verifier / 'oof.npz'),
               'training_contract_sha256': sha(args.verifier / 'contract.json'),
               'training_summary_sha256': sha(args.verifier / 'training-summary.json'),
               'fold_recipe_sha256': recipes, 'new_questions': len(ids), 'all_new_nodes_verified': len(alignment),
               'decision_nodes_used': len(lookup),
               'neural_oof_nodes_by_fold': dict(Counter(str(r['fold']) for r in metadata)),
               'neural_oof_shapes_by_fold': {str(f): {k: list(outputs[k][folds == f].shape)
                                                    for k in ('state', 'plain', 'guided')} for f in (0, 1, 2)},
               'development_test_question_overlap': 0,
               'neural_question_exclusion': 'Original+replay+new training question hash fold excluded from its neural model',
               'forecaster_crossfit': 'Separate grouped3fold forecaster fit using the fixed neural OOF feature table',
               'nested_crossfit': False, 'crossfit_limitation': EXCLUSION_NOTE}
    return lookup, alignment, receipt, summary


def reference_candidate(path, panel_sha):
    summary = wait_json(path)
    identity = json.loads((path.parent / 'identity.json').read_bytes())
    actual = identity.get('dev_questions_sha256', identity.get('development_questions_sha256',
             identity.get('development_question_sha256', identity.get('dev_question_sha256'))))
    if actual != panel_sha or summary.get('development_questions', summary.get('questions')) != 256:
        raise ValueError('Existing A reference has a different development panel')
    return {**summary['selected_A'], 'source_summary': str(path.resolve()), 'source_summary_sha256': sha(path)}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    panel_sha = sha(args.data / 'splits/dev-questions.jsonl')
    identity = {'source_sha256': sha(__file__), 'pomdp_sha256': sha(core.__file__),
                'policy_sha256': sha(policy.__file__), 'dev_questions_sha256': panel_sha,
                'train_questions_sha256': sha(args.data / 'splits/train-questions.jsonl'),
                'verifier': str(args.verifier.resolve()), 'test_used': False,
                'regularization': [.1, 1.], 'smoothing': [8., 32.], 'posterior_weights': [0., .5, 1.],
                'references': [str(p.resolve()) for p in args.reference], 'nested_crossfit': False}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    train, trees, _, _ = wait_plain(args.data, 'train')
    dev, dt, _, _ = wait_plain(args.data, 'dev')
    if len(train) != 512 or len(dev) != 256:
        raise ValueError('Use the entire registered512train/256dev split')
    lookup, alignment, verified, summary = load_verified(args, train)
    (args.output / 'oof-alignment.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in alignment))
    verified['alignment_sha256'] = sha(args.output / 'oof-alignment.jsonl')
    (args.output / 'oof-provenance.json').write_text(json.dumps(verified, indent=2) + '\n')
    references = [reference_candidate(p, panel_sha) for p in args.reference]
    observer = perception.Observer(str((args.verifier / 'full').resolve()),
                                  summary['temperature_selected_on_training_oof'], device=args.device)
    observer.load()
    baselines = {name: statistics([fixed(r, dt[r['id']], *s) for r in dev]) for name, s in BASELINES.items()}
    candidates = []
    for reg in (.1, 1.):
        for smoothing in (8., 32.):
            name = f'v4-C{reg:g}-smooth{smoothing:g}'
            model = core.fit(train, trees, observer, lookup, smoothing=smoothing, regularization=reg)
            model.fit_receipt.update(neural_observer_oof_provenance=verified, nested_crossfit=False)
            model_path = args.output / (name + '.pkl')
            model_path.write_bytes(pickle.dumps(model, protocol=5))
            by_subject = {s: [] for s in ('math', 'chemistry', 'physics')}
            for weight in (0., .5, 1.):
                for penalty in (0., .01, .02, .04, .06, .08, .10, .16, .24):
                    settings = {'penalty': penalty, 'posterior_weight': weight}
                    items = [policy.evaluate(r, dt[r['id']], {}, model, settings, 'A') for r in dev]
                    candidates.append({'model': name, 'settings': settings, 'dev': statistics(items),
                                       'model_path': str(model_path.resolve()), 'policy_module': 'continuous_repairability'})
                    for subject in by_subject:
                        by_subject[subject].append({'settings': settings,
                            'dev': statistics([v for r, v in zip(dev, items) if r['subject'] == subject])})
            settings = {'by_subject': {s: max(v, key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))['settings']
                                       for s, v in by_subject.items()}}
            items = [policy.evaluate(r, dt[r['id']], {}, model, settings, 'A') for r in dev]
            candidates.append({'model': name, 'settings': settings, 'dev': statistics(items),
                               'model_path': str(model_path.resolve()), 'policy_module': 'continuous_repairability'})
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
            print('Completed V4OOF repairability', name, json.dumps(candidates[-1]), flush=True)
    best = max(candidates, key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
    retained = max([best, *references], key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
    highest = max(m['accuracy'] for m in baselines.values())
    result = {'selected_A': retained, 'refined_A': best, 'existing_A_references': references, 'baselines': baselines,
              'development_questions': 256, 'fitting_questions': 512, 'test_used': False,
              'A_above_highest_baseline': retained['dev']['accuracy'] - highest,
              'A_plus_10_points': retained['dev']['accuracy'] >= highest + .1 - 1e-12,
              'oof_provenance_sha256': sha(args.output / 'oof-provenance.json'),
              'nested_crossfit': False, 'crossfit_limitation': EXCLUSION_NOTE}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'verifier', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--reference', type=Path, action='append', default=[], help='Retain an existing same-panel A summary if stronger; repeatable.')
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    run(p.parse_args())
