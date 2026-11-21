"""Preregistered CPU-only sparse-text diagnostic on all512 training questions.

Only sealed training responses and already computed old-only neural features
are read. Development scores/data, fresh tests, and model inference are absent.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
from unittest import mock

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sparse_text_repairability as sparse_core
import development_nonlinear as cached
import development_repairability as original
import dynamic_grounded_pomdp as perception
from prepare_mixed_mentor import plain_data


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def no_model_calls():
    def forbidden(*args, **kwargs):
        raise AssertionError('Sparse-text diagnostic forbids neural inference and model loading')
    with mock.patch.object(perception.Observer, 'load', side_effect=forbidden), \
            mock.patch.object(perception, 'predict', side_effect=forbidden), \
            mock.patch.object(original, 'predict', side_effect=forbidden), \
            mock.patch.object(cached.perception, 'load', side_effect=forbidden):
        yield


def source_files():
    root = Path(__file__).resolve().parents[4]
    paths = {Path(module.__file__).resolve() for module in list(sys.modules.values())
             if getattr(module, '__file__', None) and Path(module.__file__).is_file()
             and Path(module.__file__).resolve().is_relative_to(root)}
    return {str(path): sha(path) for path in sorted(paths)}


def load_inputs(args):
    identity_path = args.base / 'identity.json'; identity = read(identity_path)
    if (identity['new_train_questions_sha256'] != sha(args.data / 'splits/train-questions.jsonl') or
            identity['encoder_trained_on_new_fit_questions'] or identity['source_sha256'] != sha(original.__file__) or
            identity['pomdp_sha256'] != sha(sparse_core.core.__file__)):
        raise ValueError('Require the exact prior training feature-cache provenance and unchanged core')
    rows, trees = plain_data(args.data, 'train')
    if len(rows) != 512 or len({r['id'] for r in rows}) != 512:
        raise ValueError('Use all512 unique registered training questions')
    from collections import Counter
    if Counter(r['subject'] for r in rows) != {'math': 256, 'chemistry': 128, 'physics': 128}:
        raise ValueError('Changed training subject mix')
    observer = cached.observer_from(args.base)
    legacy_sha = sha(args.legacy_examples)
    if (identity['old_only_encoder_inputs_sha256'], observer._recipe['data_sha256']) != (legacy_sha, legacy_sha):
        raise ValueError('Old-only neural supervision provenance changed')
    legacy_ids = {json.loads(line)['id'] for line in args.legacy_examples.read_text().splitlines()}
    if legacy_ids & {r['id'] for r in rows}:
        raise ValueError('A new training question was used in the old neural encoder')
    corpus = Path(perception.CORPUS)
    if not (corpus / 'tfidf.pkl').exists() or sha(corpus / 'passages.jsonl') != observer._recipe['corpus_sha256']:
        raise ValueError('Require the unchanged existing retrieval corpus and CPU cache')
    metadata_path, neural_path = args.base / 'training-neural.jsonl', args.base / 'training-neural.npz'
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    with np.load(neural_path) as values:
        lookup, alignment = cached.verify_features(rows, trees, observer, metadata, values)
    data = sparse_core.table(rows, trees, lookup)
    evidence_paths = [identity_path, args.base / 'C1-smooth8.pkl', metadata_path, neural_path,
        args.data / 'splits/train-questions.jsonl', args.data / 'train-plain-complete.json', args.legacy_examples,
        Path(observer.directory) / 'recipe.json', corpus / 'passages.jsonl', corpus / 'tfidf.pkl']
    evidence = {str(path.resolve()): sha(path) for path in evidence_paths}
    receipt = read(args.data / 'train-plain-complete.json')
    provenance = {'input_evidence': evidence, 'base_directory': str(args.base.resolve()),
        'training_neural_sha256': sha(neural_path), 'training_neural_metadata_sha256': sha(metadata_path),
        'encoder_recipe_sha256': identity['encoder_recipe_sha256'],
        'old_only_encoder_inputs_sha256': legacy_sha, 'encoder_new_question_overlap': 0,
        'train_question_sha256': sha(args.data / 'splits/train-questions.jsonl'),
        'sealed_plain_prefix_sha256': receipt['base_bank_sha256'], 'sealed_plain_prefix_records': receipt['base_records'],
        'text_inputs_sha256': sparse_core.fingerprint(data['texts']),
        'numeric_features_sha256': sparse_core.array_sha(data['numeric']),
        'joint_training_labels_sha256': sparse_core.array_sha(data['labels']),
        'paid_alignment_sha256': sparse_core.fingerprint(alignment), 'neural_logit_columns': 20,
        'numeric_columns': data['numeric'].shape[1], 'decision_nodes': len(data['labels']),
        'training_questions': len(rows), 'observer_temperature_from_saved_parent': float(observer.temperature)}
    return data, alignment, provenance


def registration(provenance):
    from datetime import datetime, timezone
    text = {name: sparse_core.vectorizer(name).get_params() for name in sparse_core.VARIANTS[1:]}
    # sklearn's dtype is a Python type; use its exact name in the JSON recipe.
    for value in text.values():
        value['dtype'] = str(np.dtype(value['dtype']))
        value['ngram_range'] = list(value['ngram_range'])
    return {**provenance, 'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_files': source_files(), 'scope': 'Training-only diagnostic; no POMDP runtime replacement or development selection',
        'variants': list(sparse_core.VARIANTS), 'text_parameters': text,
        'numeric_control': 'Existing paid numeric observations and20cached logits; identical grouped fitting pipeline',
        'text_fields': ['original question', 'root response', 'paid action and current response when current differs from root'],
        'text_excludes': ['reference answer', 'question ID', 'future response', 'unpaid branch', 'retrieval background text'],
        'case_preserved': True, 'numeric_scaling': 'StandardScaler fit only on outer-training question rows',
        'classifier': {**sparse_core.MODEL_PARAMETERS, 'penalty': 'l2', 'multiclass': 'multinomial softmax'},
        'temperature': 1., 'additional_calibrator': False,
        'folds': 'int(SHA256(question_id)[:8],16)%3; all three decision nodes held together',
        'vocabulary_idf_scaler_and_classifier_exclude_held_question': True,
        'fold_models_only': True, 'no_full_runtime_model_fit': True, 'no_hyperparameter_search': True,
        'report_panels': list(sparse_core.PANELS), 'metrics': list(sparse_core.METRICS),
        'comparisons': 'Report both text variants minus the numeric C1 control on every fixed panel; lower losses are better',
        'uncertainty': '2000 paired-question bootstrap draws, fitted fold models held fixed; descriptive training OOF uncertainty',
        'timing': 'Single-node CPU head on16fixedhash held questions per fold, all three nodes; excludes neural inference',
        'sparse_work': 'Record nonzero features and2*nnz*number_of_classes linear-dot FLOPs separately from unchanged solver/NN proxy',
        'POMDP': 'No transition, support, Bellman, horizon or price change; this stage only measures candidate observations',
        'split': 'training only', 'test_used': False, 'development_values_read': False,
        'neural_calls': 0, 'cloud_calls': 0, 'seed': sparse_core.SEED,
        'environment': {'python': list(sys.version_info[:3]), 'packages': {
            name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')}}}


def run(args):
    if args.output.exists():
        raise ValueError('Use a new diagnostic area; never overwrite a registered/scored run')
    with no_model_calls(), threadpool_limits(limits=1):
        data, alignment, provenance = load_inputs(args)
        args.output.mkdir(parents=True)
        specification = registration(provenance)
        write(args.output / 'registration.json', specification)
        write(args.output / 'training-alignment.json', alignment)
        print(json.dumps({'registered': True, 'questions': 512, 'decision_nodes': len(data['labels']),
            'variants': list(sparse_core.VARIANTS), 'registration_sha256': sha(args.output / 'registration.json'),
            'scored': False, 'neural_calls': 0, 'cloud_calls': 0}), flush=True)
        predictions, receipts = {}, {}
        for name in sparse_core.VARIANTS:
            out = np.full((len(data['labels']), 8), np.nan); receipts[name] = []
            for fold in (0, 1, 2):
                held, probabilities, model, receipt = sparse_core.outer_prediction(data, name, fold)
                out[held] = probabilities
                path = args.output / f'{name}-fold{fold}.pkl'; path.write_bytes(pickle.dumps(model, protocol=5))
                receipts[name].append({**receipt, 'model_sha256': sha(path)})
            if not np.isfinite(out).all():
                raise ValueError('Incomplete held-question forecasts')
            predictions[name] = out
            print('Completed training-only sparse-text folds:', name, flush=True)
        np.savez_compressed(args.output / 'private-oof.npz', **predictions)
        write(args.output / 'fold-receipts.json', receipts)
        reports = sparse_core.diagnostics(data, predictions)
        for path, digest in {**specification['source_files'], **specification['input_evidence']}.items():
            if sha(path) != digest:
                raise ValueError('Registered training evidence or source changed during the diagnostic')
        plain_data(args.data, 'train')  # Recheck the sealed prefix despite later ordinary collector appends.
        summary = {'registration_sha256': sha(args.output / 'registration.json'),
            'alignment_sha256': sha(args.output / 'training-alignment.json'),
            'predictions_sha256': sha(args.output / 'private-oof.npz'),
            'fold_receipts_sha256': sha(args.output / 'fold-receipts.json'),
            'training_questions': 512, 'decision_nodes': len(data['labels']), 'reports': reports,
            'convergence_warnings': {name: [r['convergence_warnings'] for r in values] for name, values in receipts.items()},
            'head_cpu_latency': {name: [r['head_cpu_latency'] for r in values] for name, values in receipts.items()},
            'split': 'training only', 'test_used': False, 'development_values_read': False,
            'neural_calls': 0, 'cloud_calls': 0, 'POMDP_runtime_changed': False,
            'interpretation': 'Question-excluded observation-model diagnostic. These scores are not controller accuracy or fresh-data results. Both fixed text candidates and numeric control are reported; no runtime model is selected here.'}
        write(args.output / 'summary.json', summary)
        print(json.dumps({'complete': True, 'summary_sha256': sha(args.output / 'summary.json'),
                          'all_nodes': reports['all'], 'root': reports['root']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'base', 'legacy-examples', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
