"""Nested whole-question TRAIN screen for a sparse lexical POMDP forecaster."""
import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

import policy_crossfit_diagnostic as dataset
import train_joint_outcome_observer as conditional
import lexical_outcome_forecast as forecast

io, fixed, nested = dataset.io, dataset.fixed, dataset.nested
SCHEMA = 'sparse-paid-text-joint16-outcome-TRAIN-v1'


def inputs(args):
    rows, trees, observer, data, parent = dataset.inputs(args)
    by_id = {r['id']: r for r in rows}; texts = []
    for r in data['records']:
        row = {**by_id[r['id']], 'answer': ''}; path = r['path']
        texts.append(forecast.public_text(row, path, {p: trees[row['id']][p] for p in ('', path)}))
    with np.load(args.expanded_fit/'private-oof.npz', allow_pickle=False) as archive:
        control = archive[nested.FAMILIES[0]].copy()
    slots = {key: value[3] for key, value in observer.values_by_key.items()}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**parent['input_evidence'],
            str((args.expanded_fit/'private-oof.npz').resolve()): io.sha(args.expanded_fit/'private-oof.npz')},
        'paths': {k: str(getattr(args, k).resolve()) for k in ('expanded_fit', 'output')},
        'training_questions': 1280, 'nodes': 3840, 'outer_question_folds': 3, 'inner_question_folds': 3,
        'features': 'TFIDF paid question/root/current word unigrams+bigrams,20000 cap,min_df2,sublinear; normalized raw130',
        'text_sha256': fixed.fingerprint(texts), 'labels_sha256': fixed.array_sha(data['labels']),
        'numeric_sha256': fixed.array_sha(data['features'][nested.FAMILIES[0]]),
        'regularization': list(forecast.SETTINGS), 'selection': 'Lowest inner question-held joint log loss',
        'vocabulary_scaling_prior_and_classifier_fit_only_inner_training': True,
        'prior_weight': forecast.PRIOR_WEIGHT, 'penalty': .02,
        'screen': 'At least2pp conditional child accuracy gain and no higher compute than excluded raw1280 forest',
        'scope': 'Conditional last-step TRAIN diagnostic; not full-controller or fresh validation',
        'new_neural_calls': 0, 'cloud_calls': 0, 'encoder_training': False,
        'development_or_test_read': False, 'CPU_head_latency_outside_parameter_token_proxy': True}
    return rows, trees, data, texts, slots, control, observer._recipe['parameters'], record


def select(data, texts, train):
    inner = np.asarray([nested.inner_fold(r['id']) for r in data['records']]); x = data['features'][nested.FAMILIES[0]]
    scores = []
    for regularization in forecast.SETTINGS:
        predicted = np.full((len(train), 16), np.nan)
        for fold in range(3):
            fit, held = train & (inner != fold), train & (inner == fold)
            nested.group_receipt(data['records'], fit, held)
            model = forecast.fit([t for t, keep in zip(texts, fit) if keep], x[fit], data['labels'][fit], regularization)
            predicted[held] = model.predict([t for t, keep in zip(texts, held) if keep], x[held], data['valid'][held])
        if not np.isfinite(predicted[train]).all() or np.isfinite(predicted[~train]).any():
            raise ValueError('Inner prediction scope changed')
        score = float(fixed.losses(data['labels'][train], predicted[train])['joint_log_loss'].mean())
        scores.append({'C': regularization, 'joint_log_loss': score})
    return min(scores, key=lambda r: (r['joint_log_loss'], r['C']))['C'], scores


def diagnose(args, rows, trees, data, texts, slots, control, parameters, record):
    outer = np.asarray([r['fold'] for r in data['records']]); x = data['features'][nested.FAMILIES[0]]
    predicted = np.full((len(texts), 16), np.nan); receipts = []
    for fold in range(3):
        train, held = outer != fold, outer == fold
        receipt = nested.group_receipt(data['records'], train, held)
        selected, scores = select(data, texts, train)
        model = forecast.fit([t for t, keep in zip(texts, train) if keep], x[train], data['labels'][train], selected)
        predicted[held] = model.predict([t for t, keep in zip(texts, held) if keep], x[held], data['valid'][held])
        io.atomic_bytes(args.output/f'outer{fold}.pkl', pickle.dumps(model, protocol=5))
        receipts.append({**receipt, 'fold': fold, 'selected_C': selected, 'inner_scores': scores,
            'vocabulary_size': len(model.vectorizer.vocabulary_)})
        print({'outer_fold_complete': fold, 'selected_C': selected}, flush=True)
    if not np.isfinite(predicted).all(): raise ValueError('Incomplete excluded-question forecasts')
    old = conditional.conditional(rows, trees, data, slots, control, parameters)
    new = conditional.conditional(rows, trees, data, slots, predicted, parameters)
    metrics = lambda values: {'n': len(values), 'correct': sum(int(v['correct']) for v in values),
        'accuracy': float(np.mean([v['correct'] for v in values])), 'mean_compute': float(np.mean([v['compute'] for v in values]))}
    before, after = metrics(old), metrics(new)
    delta = after['accuracy']-before['accuracy']; ratio = after['mean_compute']/before['mean_compute']
    groups = {r['id']: [] for r in old}
    for i, r in enumerate(old): groups[r['id']].append(i)
    differences = np.asarray([np.mean([int(new[i]['correct'])-int(old[i]['correct']) for i in indices]) for indices in groups.values()])
    draws = np.random.default_rng(238031).integers(0, 1280, (2000, 1280))
    screen = {'conditional_accuracy_gain': delta, 'compute_ratio': ratio, 'passed': delta >= .02 and ratio <= 1.,
        'paired_question_accuracy_CI95': np.quantile(differences[draws].mean(1), [.025, .975]).tolist()}
    result = {'schema': SCHEMA, 'control': before, 'candidate': after, 'screen': screen,
        'forecast_losses': {'control': {k: float(v.mean()) for k, v in fixed.losses(data['labels'], control).items()},
            'candidate': {k: float(v.mean()) for k, v in fixed.losses(data['labels'], predicted).items()}},
        'cloud_calls': 0, 'new_neural_calls': 0, 'development_or_test_read': False,
        'fresh_validation': False, 'full_model_fitted': False}
    io.npz(args.output/'oof.npz', {'probability': predicted})
    io.write(args.output/'private-conditional-items.json', {'control': old, 'candidate': new})
    io.write(args.output/'fold-receipts.json', receipts); io.write(args.output/'summary.json', result)
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'screen_passed': screen['passed'], 'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'control': before, 'candidate': after, 'screen': screen}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'diagnose'))
    for name in ('expanded-fit', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered lexical experiment')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()):
                raise ValueError('Preserve partial or complete lexical run')
            diagnose(args, *bundle)


if __name__ == '__main__': main()
