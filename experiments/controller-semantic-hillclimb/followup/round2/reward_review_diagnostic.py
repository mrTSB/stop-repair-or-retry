"""Frozen reward scorer for paid LLM reviews, compared with strongest prior critic.

This is a TRAIN sensor screen, not a controller result or a new answer path.
All review-generation and scorer costs remain required for later hybrid use.
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import science_review_critic as previous
import reward_outcome_diagnostic as acquisition
import matched_review_b as charges

io, fixed, observer, nested = acquisition.io, acquisition.fixed, acquisition.observer, acquisition.nested
SCHEMA = 'frozen-reward-paid-review256-TRAIN-v1'
SETTINGS = (('score', .01), ('score', .1), ('score', 1.), ('dense', .01), ('dense', .1), ('dense', 1.))


def conversation(row, review):
    if row.get('answer') != '' or not review.get('text') or not review.get('calls'):
        raise ValueError('Only blank-reference questions and complete paid reviews may be scored')
    charges.validate_review_charge(review)
    return [{'role': 'user', 'content': row['question']}, {'role': 'assistant', 'content': review['text']}]


def inputs(args):
    spec = io.read(args.previous/'registration.json')
    prior_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    prior_args.registration_sha256 = io.sha(args.previous/'registration.json')
    rows, trees, reviews, old_logits, expected = previous.load_inputs(prior_args)
    if expected != spec: raise ValueError('Changed prior critic provenance')
    done = io.read(args.previous/'complete.json'); graded = io.read(args.previous/'graded/summary.json')
    if graded['complete_sha256'] != io.sha(args.previous/'complete.json') or graded['development_or_test_read'] or graded['fresh_validation']:
        raise ValueError('Require original TRAIN-only completed critic')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    if done['registration_sha256'] != prior_args.registration_sha256: raise ValueError('Changed previous completion')
    io.verify({str(args.previous/k): v for k, v in done['digests'].items()})
    # Verify the previous completion using its own physical observation replay.
    previous.restore(prior_args, rows, True)
    with np.load(args.previous/'graded/predictions.npz', allow_pickle=False) as archive:
        controls = {k: archive[k].copy() for k in ('old', 'option', 'combined')}
        labels = archive['labels'].copy()
        np.testing.assert_array_equal(archive['old_logits'], old_logits)
    actual = np.asarray([int(bool(reviews[r['id']]['answer']) and reviews[r['id']]['answer'] == r['answer']) for r in rows])
    np.testing.assert_array_equal(actual, labels)
    strongest = min(controls, key=lambda k: (float(np.mean((controls[k]-labels)**2)), k))
    assets = io.read(args.assets/'receipt.json')
    if assets['repository'] != observer.REPOSITORY or assets['revision'] != observer.REVISION or assets['remote_code']:
        raise ValueError('Changed scorer assets')
    files = {str((args.assets/'model'/k).resolve()): v for k, v in assets['files'].items()}; io.verify(files)
    conversations = [conversation({**r, 'answer': ''}, reviews[r['id']]) for r in rows]
    evidence = {**spec['input_evidence'], **files,
        **{str((args.previous/k).resolve()): io.sha(args.previous/k) for k in
            ('registration.json', 'complete.json', 'observations.jsonl', 'attempts.jsonl', 'graded/summary.json', 'graded/predictions.npz')},
        str((args.assets/'receipt.json').resolve()): io.sha(args.assets/'receipt.json')}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {k: str(getattr(args, k).resolve()) for k in ('previous', 'assets', 'output')},
        'questions': 256, 'subjects': {'chemistry': 128, 'physics': 128}, 'model': assets,
        'device': args.device, 'dtype': 'float32', 'batch_size': 4, 'max_length': observer.MAX_LENGTH,
        'truncation': 'First768 plus last768 tokens, every truncation reported',
        'observations': 'Original question and complete already-paid root LLM review, no references or future solver responses',
        'conversation_sha256': fixed.fingerprint(conversations), 'labels_sha256': fixed.array_sha(labels),
        'strongest_prior_critic': strongest, 'control_selection': 'Lowest Brier among all three previous excluded-question controls',
        'outer_folds': previous.fold_ids(rows).tolist(), 'inner_folds': 'Independent whole-question hash, three folds',
        'settings': [list(v) for v in SETTINGS], 'selection': 'Minimum inner whole-question Brier; fixed tie order',
        'features': 'Reward score alone, or score plus32 train-only whitened hidden PCA dimensions',
        'screen': 'OOF Brier at least0.005 below strongest prior critic and strictly higher AUC',
        'new_answer_paths': False, 'new_cloud_calls': 0, 'encoder_training': False,
        'cost': 'All actual scorer parameters times nonpadding input tokens /3.09, additive to paid LLM and any solver sensors',
        'runtime_promotion_requires': 'Full POMDP fit and development accuracy/compute gain against strongest B before any fresh validation',
        'development_or_test_read': False, 'full_controller_claim': False, 'fresh_validation': False,
        'pretraining_overlap_not_independently_established': True}
    return rows, labels, controls, conversations, record


def fit_predict(scores, hidden, labels, train, held, setting):
    kind, regularization = setting
    if len(np.unique(labels[train])) == 1: return np.full(int(held.sum()), float(labels[train][0]))
    x = scores[:, None]
    if kind == 'dense':
        pca = PCA(n_components=32, whiten=True, svd_solver='randomized', random_state=2381132).fit(hidden[train])
        x = np.column_stack((x, pca.transform(hidden)))
    learner = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=2000))
    learner.fit(x[train], labels[train])
    return learner.predict_proba(x[held])[:, 1]


def diagnose(args, rows, labels, controls, conversations, record):
    folder = args.output/'features'; done = io.read(folder/'complete.json')
    if done['registration_sha256'] != args.registration_sha256: raise ValueError('Changed feature registration')
    io.verify({str(folder/k): v for k, v in done['digests'].items()}); meta = io.read(folder/'metadata.json')
    with np.load(folder/'all.npz', allow_pickle=False) as archive:
        scores, hidden, lengths = (archive[k].copy() for k in ('scores', 'hidden', 'tokens'))
    outer = previous.fold_ids(rows); inner = np.asarray([nested.inner_fold(r['id']) for r in rows])
    predictions = np.full(len(rows), np.nan); receipts = []
    for fold in range(3):
        train, held = outer != fold, outer == fold; receipt = nested.group_receipt(rows, train, held); searches = []
        for setting in SETTINGS:
            p = np.full(len(rows), np.nan)
            for k in range(3):
                fit, validation = train & (inner != k), train & (inner == k)
                nested.group_receipt(rows, fit, validation)
                p[validation] = fit_predict(scores, hidden, labels, fit, validation, setting)
            if not np.isfinite(p[train]).all() or np.isfinite(p[~train]).any(): raise ValueError('Inner scope changed')
            searches.append({'setting': list(setting), 'Brier': float(np.mean((p[train]-labels[train])**2))})
        winner = min(range(len(searches)), key=lambda k: (searches[k]['Brier'], k)); setting = SETTINGS[winner]
        predictions[held] = fit_predict(scores, hidden, labels, train, held, setting)
        receipts.append({**receipt, 'fold': fold, 'selected': list(setting), 'inner_search': searches})
        print({'fold': fold, 'selected': setting}, flush=True)
    if not np.isfinite(predictions).all(): raise ValueError('Incomplete held-question predictions')
    reports = {k: {'Brier': float(np.mean((p-labels)**2)), 'AUC': float(roc_auc_score(labels, p))}
        for k, p in {**controls, 'reward': predictions}.items()}
    strongest = record['strongest_prior_critic']; delta = reports['reward']['Brier']-reports[strongest]['Brier']
    differences = (predictions-labels)**2-(controls[strongest]-labels)**2
    draws = np.random.default_rng(2381132).integers(0, 256, (2000, 256))
    screen = {'Brier_delta_vs_strongest': delta, 'AUC_gain_vs_strongest': reports['reward']['AUC']-reports[strongest]['AUC'],
        'paired_question_Brier_delta_CI95': np.quantile(differences[draws].mean(1), [.025, .975]).tolist(),
        'passed': delta <= -.005 and reports['reward']['AUC'] > reports[strongest]['AUC']}
    result = {'schema': SCHEMA, 'reports': reports, 'strongest_prior': strongest, 'screen': screen,
        'mean_new_observer_compute': float(meta['parameters']/1e9*lengths.mean()/3.09), 'features': meta,
        'questions': len(rows), 'new_answer_paths': False, 'new_cloud_calls': 0,
        'full_controller_claim': False, 'fresh_validation': False, 'development_or_test_read': False}
    io.npz(args.output/'oof.npz', {'probability': predictions, 'labels': labels})
    io.write(args.output/'fold-receipts.json', receipts); io.write(args.output/'summary.json', result)
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'screen_passed': screen['passed'], 'digests': {str(p.relative_to(args.output)): io.sha(p) for p in args.output.rglob('*')
            if p.is_file() and p != args.output/'complete.json' and not p.name.startswith('.')}})
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'collect', 'diagnose'))
    for name in ('previous', 'assets', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps'); parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with fixed.training_only_reads(), threadpool_limits(limits=1):
        with fixed.guarded.no_model_calls(): bundle = inputs(args)
        record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}, flush=True); return
        if not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256 or io.read(args.output/'registration.json') != record:
            raise ValueError('Changed registered review experiment')
        with io.lock(args.output/'.run.lock'):
            if (args.output/'complete.json').exists(): raise ValueError('Preserve completed experiment')
            if args.phase == 'collect': acquisition.collect(args, bundle[3], record)
            else:
                with fixed.guarded.no_model_calls(): diagnose(args, *bundle)


if __name__ == '__main__': main()
