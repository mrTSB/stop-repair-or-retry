"""CPU-only source-matched mentor refit from already archived neural outputs.

All512 new training questions and the128 registered new strong-root pairs are
retained by source, never correctness. The old64 cohort is excluded uniformly.
Both the ordinary-emission models and question-excluded-emission refinement
are produced in new areas. No neural, LLM, development or test inference runs.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import sys
from unittest import mock

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from banks import committed
import mentor_pomdp as mentor
import hybrid_repairability as hybrid
import crossfit_mentor_emissions as crossfit
import new_source_mentor_core as specialized
import prepare_mixed_mentor as prepare
from prepare_mixed_mentor import sha, fingerprint, plain_data, overlay, entries


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def no_neural_inference():
    def forbidden(*args, **kwargs):
        raise AssertionError('This refit is CPU-only; neural inference is forbidden')
    with mock.patch.object(mentor.perception.Observer, 'load', side_effect=forbidden), \
         mock.patch.object(prepare, 'predict', side_effect=forbidden), \
         mock.patch.object(mentor.perception, 'predict', side_effect=forbidden):
        yield


def select_new(rows, selection):
    if len(rows) != 512 or len({r['id'] for r in rows}) != 512:
        raise ValueError('Use all512 unique new training questions')
    if Counter(r['subject'] for r in rows) != {'math': 256, 'chemistry': 128, 'physics': 128}:
        raise ValueError('Changed new training subject mix')
    if len(selection) != 192 or len({r['id'] for r in selection}) != 192:
        raise ValueError('Require the original192-question mentor registration')
    chosen = [r for r in selection if r['bank'] == 'new']
    excluded = [r for r in selection if r['bank'] == 'old']
    ids = {r['id'] for r in rows}
    if (len(chosen) != 128 or len(excluded) != 64 or not {r['id'] for r in chosen} <= ids
            or {r['id'] for r in excluded} & ids
            or Counter(r['subject'] for r in chosen) != {'math': 64, 'chemistry': 32, 'physics': 32}):
        raise ValueError('Require exactly128 registered new pairs and64 uniformly excluded old pairs')
    by_id = {r['id']: r for r in rows}
    if any(by_id[r['id']]['subject'] != r['subject'] for r in chosen):
        raise ValueError('Registered subject changed')
    return chosen, excluded


def aligned_subset(metadata, outputs, expected):
    keys = [(r['id'], r['kind'], r['path']) for r in metadata]
    if len(set(keys)) != len(keys) or any(len(value) != len(metadata) for value in outputs.values()):
        raise ValueError('Duplicate or misaligned parent neural cache')
    lookup = dict(zip(keys, range(len(keys))))
    selected = []
    for item in expected:
        key = item['id'], item['kind'], item['path']
        index = lookup.get(key)
        if index is None or metadata[index] != item or item['fold'] != 'full':
            raise ValueError('Exact new-source input, label, prompt or fold alignment failed')
        selected.append(index)
    expected_ids = {r['id'] for r in expected}
    if sum(r['id'] in expected_ids for r in metadata) != len(expected):
        raise ValueError('Unexpected extra parent node on a retained question')
    result = {name: np.asarray(value)[selected].copy() for name, value in outputs.items()}
    if set(result) != {'state', 'plain', 'guided'} or any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Incomplete cached neural outputs')
    if any(result[name].shape != (len(expected), width)
           for name, width in (('state', 4), ('plain', 8), ('guided', 8))):
        raise ValueError('Changed neural output shape')
    return result, selected


def source_files():
    root = Path(__file__).resolve().parents[4]
    files = {Path(module.__file__).resolve() for module in list(sys.modules.values())
             if getattr(module, '__file__', None) and Path(module.__file__).is_file()
             and Path(module.__file__).resolve().is_relative_to(root)}
    return {str(path): sha(path) for path in sorted(files)}


def load_inputs(args):
    original = read(args.parent / 'complete.json'); parent_identity = read(args.parent / 'identity.json')
    neural_receipt = read(args.parent / 'neural-receipt.json')
    contract = read(args.mentor / 'contract.json'); completed = read(args.mentor / 'complete.json')
    if original['identity_sha256'] != sha(args.parent / 'identity.json') or original['development_or_test_used']:
        raise ValueError('Changed original training fit identity')
    if (parent_identity['mentor_bank_sha256'], parent_identity['mentor_contract_sha256'],
            completed['bank_sha256'], completed['contract_sha256']) != (
            sha(args.mentor / 'train.jsonl'), sha(args.mentor / 'contract.json'),
            sha(args.mentor / 'train.jsonl'), sha(args.mentor / 'contract.json')):
        raise ValueError('Original registered strong mentor bank changed')
    if parent_identity['source_sha256'] != sha(prepare.__file__):
        raise ValueError('Original input preparer source changed')
    if any(sha(path) != digest for path, digest in parent_identity['model_files'].items()):
        raise ValueError('Original model implementation changed')
    if parent_identity['new_train_split_sha256'] != sha(args.new_data / 'splits/train-questions.jsonl'):
        raise ValueError('New training split changed')
    if (neural_receipt['metadata_sha256'], neural_receipt['outputs_sha256']) != (
            sha(args.parent / 'metadata.jsonl'), sha(args.parent / 'neural.npz')):
        raise ValueError('Parent archived neural outputs changed')
    if neural_receipt['encoder_recipes']['full'] != sha(args.verifier / 'full/recipe.json'):
        raise ValueError('Changed old-only full observer recipe')
    recipe = read(args.verifier / 'full/recipe.json')
    inherited_summary_path = args.verifier / 'training-summary.json'
    inherited_temperature = read(inherited_summary_path)['temperature_selected_on_training_oof']
    parent_spec = original['models'][0]
    parent_path = args.parent / (parent_spec['key'] + '.pkl')
    if sha(parent_path) != parent_spec['sha256']:
        raise ValueError('Changed parent model calibration anchor')
    parent_model = pickle.loads(parent_path.read_bytes())
    if (not np.isfinite(inherited_temperature) or inherited_temperature <= 0
            or parent_model.mentor.classifier.temperature != inherited_temperature):
        raise ValueError('Inherited training temperature differs from the verified parent model')
    if recipe['fold'] != 'full' or recipe['input_profile'] != 'full-options-openstax-v1':
        raise ValueError('Use the declared unchanged full old-V1 observer')
    if (recipe['data_sha256'], parent_identity['legacy_examples_sha256']) != (sha(args.legacy_examples), sha(args.legacy_examples)):
        raise ValueError('Old observer supervised-input provenance changed')
    rows, plain = plain_data(args.new_data, 'train')
    chosen, excluded = select_new(rows, contract['selection'])
    old_ids = {json.loads(line)['id'] for line in args.legacy_examples.read_text().splitlines()}
    if old_ids & {r['id'] for r in rows}:
        raise ValueError('A new training question was supervised by the old full neural observer')
    chosen_ids = {r['id'] for r in chosen}
    all_records = committed(args.mentor / 'train.jsonl')
    records = [r for r in all_records if r['id'] in chosen_ids]
    trees, judges = overlay(rows, plain, records, chosen, validate=True)
    observer = mentor.perception.Observer(str((args.verifier / 'full').resolve()), inherited_temperature, device='cpu')
    corpus = Path(mentor.perception.CORPUS)
    if not (corpus / 'tfidf.pkl').exists() or sha(corpus / 'passages.jsonl') != recipe['corpus_sha256']:
        raise ValueError('Require the existing unchanged textbook corpus and retrieval cache')
    # entry() uses paid text and deterministic textbook retrieval only; the
    # no-neural guard rejects any accidental observer.load()/predict() path.
    expected, inputs = entries(rows, trees, judges, set(), observer)
    metadata = [json.loads(line) for line in (args.parent / 'metadata.jsonl').read_text().splitlines()]
    with np.load(args.parent / 'neural.npz') as loaded:
        outputs, indexes = aligned_subset(metadata, {key: loaded[key] for key in loaded.files}, expected)
    base_spec = original['base_A']
    base_path = args.base / (base_spec['model'] + '.pkl')
    if sha(base_path) != parent_identity['base_model_sha256']:
        raise ValueError('The fixed parent A8 model changed')
    identity = {'source_sha256': sha(__file__), 'source_files': source_files(),
                'parent_models_directory': str(args.parent.resolve()),
                'parent_complete_sha256': sha(args.parent / 'complete.json'),
                'parent_identity_sha256': sha(args.parent / 'identity.json'),
                'parent_neural_receipt_sha256': sha(args.parent / 'neural-receipt.json'),
                'parent_neural_metadata_sha256': sha(args.parent / 'metadata.jsonl'),
                'parent_neural_sha256': sha(args.parent / 'neural.npz'),
                'mentor_contract_sha256': sha(args.mentor / 'contract.json'),
                'mentor_bank_sha256': sha(args.mentor / 'train.jsonl'),
                'new_train_question_sha256': sha(args.new_data / 'splits/train-questions.jsonl'),
                'legacy_examples_sha256': sha(args.legacy_examples),
                'encoder_recipe_sha256': sha(args.verifier / 'full/recipe.json'),
                'encoder_training_summary_sha256': sha(inherited_summary_path),
                'inherited_temperature': inherited_temperature,
                'temperature_parent_model_sha256': parent_spec['sha256'],
                'corpus_sha256': sha(corpus / 'passages.jsonl'), 'tfidf_sha256': sha(corpus / 'tfidf.pkl'),
                'base_model_sha256': sha(base_path), 'base_model_path': str(base_path.resolve()),
                'source_selection': 'All512 new training rows and all128 registered bank=new mentor pairs; exclude all64 bank=old uniformly',
                'selected_pairs': chosen, 'excluded_pairs': excluded,
                'fit_questions': len(rows), 'paired_review_questions': len(judges),
                'tree_sha256': fingerprint(trees), 'reused_parent_indexes_sha256': fingerprint(indexes),
                'input_sha256': fingerprint(inputs), 'input_rows': len(inputs),
                'all_retained_neural_rows_from_old_full_unseen_model': True,
                'neural_inference_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False,
                'smoothing': [8., 32.], 'semantic_review': [False, True]}
    return rows, trees, judges, expected, outputs, base_path, base_spec, identity


def refit_emissions(model, rows, trees, judges, metadata, outputs, smoothing):
    by_id = {r['id']: r for r in rows}
    indices = [i for i, r in enumerate(metadata) if r['kind'] == 'solver']
    solver_meta = [metadata[i] for i in indices]
    features = [model.mentor.classifier.calibration_features(by_id[r['id']], r['path'], outputs['state'][i])
                for i, r in zip(indices, solver_meta)]
    probabilities, folds = crossfit.calibration_crossfit(features, [r['label'] for r in solver_meta],
        [r['id'] for r in solver_meta], [r['train'] and r['valid'] for r in solver_meta])
    lookup = {(r['id'], r['path']): probabilities[i] for i, r in enumerate(solver_meta)}
    records = []
    paired = [r for r in rows if (r['id'], '') in judges]
    if len(paired) != 128:
        raise ValueError('Expected every registered new source pair')
    for row in paired:
        qid = row['id']; judge = judges[qid, '']; tree = trees[qid]
        teacher = int(mentor.correct(judge, row['answer']))
        hard = int(np.mean([mentor.correct(tree[p], row['answer']) for p in mentor.core.FIXED]) < .6)
        for action, path in enumerate(('guided', 'retry')):
            state = mentor.state(2 * hard + int(mentor.correct(tree[path], row['answer'])), teacher)
            observation = mentor.core.observation(lookup[qid, path], bool(tree[path]['answer']))
            records.append((row['subject'], action, state, observation * 3 + mentor.comparison(judge, tree[path])))
    before = crossfit.preserved_digest(model)
    values = crossfit.rebuild_emissions(records, model.mentor.dynamics, smoothing)
    for subject, dynamics in model.mentor.dynamics.items():
        dynamics.emission = values[subject]
        dynamics.mentor = values[subject].reshape(2, 8, mentor.N_OBS, 3).sum(2)
    if crossfit.preserved_digest(model) != before:
        raise ValueError('Question-excluded emission refinement changed another model component')
    return probabilities, folds, before


def run(args):
    if args.output.exists() or args.refined.exists():
        raise ValueError('Use two new output directories; no implicit refit or overwrite')
    with no_neural_inference(), threadpool_limits(limits=1):
        rows, trees, judges, metadata, outputs, base_path, base_spec, identity = load_inputs(args)
        args.output.mkdir(parents=True); args.refined.mkdir(parents=True)
        write(args.output / 'identity.json', identity)
        (args.output / 'metadata.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metadata))
        np.savez_compressed(args.output / 'neural.npz', **outputs)
        write(args.output / 'neural-receipt.json', {'metadata_sha256': sha(args.output / 'metadata.jsonl'),
            'outputs_sha256': sha(args.output / 'neural.npz'), 'source': 'Exact aligned parent subset; no new neural calls',
            'parent_neural_receipt_sha256': identity['parent_neural_receipt_sha256'],
            'input_sha256': identity['input_sha256'], 'n': len(metadata)})
        models, refined = [], []
        print(json.dumps({'registered': True, 'questions': len(rows), 'pairs': len(judges),
                          'cached_neural_rows': len(metadata), 'new_neural_calls': 0}), flush=True)
        for smoothing in (8., 32.):
            for semantic in (False, True):
                key = f'smooth{smoothing:g}-review{int(semantic)}'
                m = specialized.fit_new(rows, trees, judges, args.verifier / 'full', metadata, outputs,
                                        smoothing=smoothing, semantic_review=semantic)
                m.fit_receipt.update(new_source_only=True, new_questions=512, legacy_questions=0,
                    paired_source_questions=128, inherited_neural_outputs=True,
                    full_observer_never_trained_on_new_questions=True,
                    new_source_fit_identity_sha256=sha(args.output / 'identity.json'))
                base = pickle.loads(base_path.read_bytes())
                model = specialized.build_new(base, m, rows, trees, judges, smoothing=smoothing)
                model.fit_receipt.update(new_source_only=True, paired_source_questions=128,
                    fit_module='new_source_mentor_core',
                    new_source_fit_identity_sha256=sha(args.output / 'identity.json'))
                path = args.output / (key + '.pkl'); path.write_bytes(pickle.dumps(model, protocol=5))
                spec = {'key': key, 'sha256': sha(path), 'smoothing': smoothing, 'semantic_review': semantic}
                models.append(spec)
                probabilities, folds, preserved = refit_emissions(model, rows, trees, judges, metadata, outputs, smoothing)
                if not refined:
                    np.savez_compressed(args.refined / 'calibration-oof.npz', probabilities=probabilities)
                    refinement_identity = {'source_sha256': sha(__file__), 'source_files': identity['source_files'],
                        'new_source_parent_identity_sha256': sha(args.output / 'identity.json'),
                        'neural_metadata_sha256': sha(args.output / 'metadata.jsonl'),
                        'neural_outputs_sha256': sha(args.output / 'neural.npz'),
                        'fit_questions': 512, 'paired_review_questions': 128, 'emission_records': 256,
                        'calibration': 'C1 logistic, same valid solver training mask, whole-question3fold exclusion',
                        'folds': folds, 'raw_neural_inputs_question_disjoint': True,
                        'inherited_temperature_never_used_new_question_labels': True,
                        'runtime_calibration_changed_from_parent': False,
                        'transitions_costs_transfer_changed_from_parent': False,
                        'calibration_oof_sha256': sha(args.refined / 'calibration-oof.npz'),
                        'neural_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False}
                    write(args.refined / 'identity.json', refinement_identity)
                model.mentor.fit_receipt.update(question_excluded_calibration_emissions=True,
                    calibration_folds=3, source_refinement_identity_sha256=sha(args.refined / 'identity.json'))
                model.fit_receipt['mentor'] = model.mentor.fit_receipt
                model.fit_receipt['emission_refinement'] = {'parent_model_sha256': spec['sha256'],
                    'identity_sha256': sha(args.refined / 'identity.json'), 'question_excluded_calibration_emissions': True}
                target = args.refined / (key + '.pkl'); target.write_bytes(pickle.dumps(model, protocol=5))
                refined.append({**spec, 'sha256': sha(target), 'parent_sha256': spec['sha256'],
                                'preserved_non_emission_sha256': preserved})
                print('Fitted and crossfit new-source mentor:', key, flush=True)
        if any(sha(path) != digest for path, digest in identity['source_files'].items()):
            raise ValueError('Registered source changed during CPU refit')
        shared = {'fit_questions': 512, 'paired_review_questions': 128, 'base_A': base_spec,
                  'new_source_only': True, 'neural_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False}
        write(args.output / 'complete.json', {**shared, 'models': models, 'identity_sha256': sha(args.output / 'identity.json')})
        write(args.refined / 'complete.json', {**shared, 'models': refined,
            'identity_sha256': sha(args.refined / 'identity.json'),
            'parent_complete_sha256': sha(args.output / 'complete.json')})
        print(json.dumps({'complete': True, 'models': len(models), 'refined_models': len(refined),
                          'raw_complete_sha256': sha(args.output / 'complete.json'),
                          'crossfit_complete_sha256': sha(args.refined / 'complete.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('new-data', 'mentor', 'parent', 'base', 'verifier', 'legacy-examples', 'output', 'refined'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
