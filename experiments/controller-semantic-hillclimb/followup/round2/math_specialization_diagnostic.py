"""TRAIN-only nested check of math-specific eight-state observation forecasts.

The retained pooled C=.1 logistic procedure is refitted without each outer
question fold. The candidate learns only math; its model/settings are selected
inside each outer training fold. No neural or development inference occurs.
"""
import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import fit_completion_pomdp as source
import nonlinear_repairability as table_source
import nested_dense_diagnostic as nested
import completion_v4_diagnostic as guarded

io, core = source.io, table_source.core
SCHEMA = 'math-specialized-eight-state-nested-TRAIN-v1'
SETTINGS = (*({'kind': 'linear', 'C': c} for c in (.001, .01, .1)),
            *({'kind': 'forest', 'leaf': n} for n in (8, 24)))
TREES = 96


@dataclass
class Forecast:
    estimator: object
    prior: np.ndarray
    smoothing: float
    @property
    def classes_(self): return np.arange(8)
    def predict_proba(self, x):
        p = np.tile(self.smoothing*self.prior, (len(x), 1))
        p[:, self.estimator.classes_.astype(int)] += (1-self.smoothing)*self.estimator.predict_proba(x)
        return p


def fit(x, y, setting, control=False):
    if len(np.unique(y)) == 1: model = DummyClassifier(strategy='prior').fit(x, y)
    elif setting['kind'] == 'linear':
        model = make_pipeline(StandardScaler(), LogisticRegression(C=setting['C'], max_iter=1000 if control else 2000)).fit(x, y)
    else:
        model = ExtraTreesClassifier(n_estimators=TREES, max_depth=8, min_samples_leaf=setting['leaf'],
            max_features=.7, n_jobs=1, random_state=2381124).fit(x, y)
    prior = np.bincount(y, minlength=8)+1.
    return Forecast(model, prior/prior.sum(), 0. if control else .01)


def predict(model, x, valid):
    p = model.predict_proba(x)+1e-9
    p[~valid, 4:] = 0.
    return p/p.sum(1, keepdims=True)


def losses(y, p):
    bits = np.stack([(np.arange(8) >> n) & 1 for n in (2, 1, 0)], axis=1)
    brier = (p @ bits-bits[y])**2
    return {'joint_log_loss': -np.log(p[np.arange(len(y)), y]),
            'current_Brier': brier[:, 0], 'repair_Brier': brier[:, 1], 'retry_Brier': brier[:, 2],
            'mean_three_marginal_Brier': brier.mean(1)}


def select(data, eligible):
    inner = np.array([nested.inner_fold(r['id']) for r in data['records']]); scores = []
    if set(inner[eligible]) != {0, 1, 2}: raise ValueError('Three nonempty inner math folds required')
    for setting in SETTINGS:
        p = np.full((len(inner), 8), np.nan)
        for fold in range(3):
            train, held = eligible & (inner != fold), eligible & (inner == fold)
            nested.group_receipt(data['records'], train, held)
            p[held] = predict(fit(data['x'][train], data['y'][train], setting), data['x'][held], data['valid'][held])
        if not np.isfinite(p[eligible]).all() or np.isfinite(p[~eligible]).any(): raise ValueError('Inner scope changed')
        scores.append({'setting': setting, 'joint_log_loss': float(losses(data['y'][eligible], p[eligible])['joint_log_loss'].mean())})
    return min(scores, key=lambda v: v['joint_log_loss'])['setting'], scores


def outer(data, fold):
    splits = np.array([r['fold'] for r in data['records']]); math = np.array([r['subject'] == 'math' for r in data['records']])
    held, train = (splits == fold) & math, splits != fold
    receipt = nested.group_receipt(data['records'], train, held)
    control = fit(data['x'][train], data['y'][train], {'kind': 'linear', 'C': .1}, control=True)
    setting, scores = select(data, train & math)
    model = fit(data['x'][train & math], data['y'][train & math], setting)
    return held, predict(control, data['x'][held], data['valid'][held]), predict(model, data['x'][held], data['valid'][held]), {
        **receipt, 'selected': setting, 'inner_search': scores, 'outer_held_labels_used': False}


def inputs(args):
    source.verify_complete(args.base_fit)
    spec = io.read(args.base_fit/'registration.json')
    old_args = SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()})
    rows, trees = source.plain_data(old_args.data, 'train')
    # Reconstruct only the sealed TRAIN inputs. The original acquisition
    # registration also rechecks a DEV prerequisite and is deliberately not
    # re-entered by this training-only diagnostic.
    observer = source.cached.observer_from(old_args.base)
    legacy = {io.json.loads(line)['id'] for line in old_args.legacy_examples.read_bytes().splitlines()}
    if legacy & {r['id'] for r in rows} or observer._recipe['data_sha256'] != io.sha(old_args.legacy_examples):
        raise ValueError('Frozen encoder saw a current training question or its ancestry changed')
    metadata = [io.json.loads(line) for line in (old_args.base/'training-neural.jsonl').read_bytes().splitlines()]
    with np.load(old_args.base/'training-neural.npz', allow_pickle=False) as values:
        lookup, alignment = source.cached.verify_features(rows, trees, observer, metadata, values)
    if guarded.fingerprint(alignment) != spec['original_neural_alignment_sha256']:
        raise ValueError('Changed exact paid TRAIN inputs')
    samples, x, y, qids = table_source.training_table(rows, trees, lookup)
    records = [{'id': r['id'], 'path': path, 'subject': r['subject'],
        'fold': int(hashlib.sha256(r['id'].encode()).hexdigest()[:8], 16) % 3} for r, path in samples]
    valid = np.array([bool(trees[r['id']][path]['answer']) for r, path in samples])
    if len(rows) != 512 or sum(r['subject'] == 'math' for r in rows) != 256 or x.shape != (1536, 130):
        raise ValueError('Require exact pooled512 and math256 training cohorts')
    data = {'records': records, 'x': x, 'y': y, 'valid': valid}
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], str((args.base_fit/'complete.json').resolve()): io.sha(args.base_fit/'complete.json')},
        'paths': {n: str(getattr(args, n).resolve()) for n in ('base_fit', 'output')},
        'records_sha256': guarded.fingerprint(records), 'features_sha256': guarded.array_sha(x),
        'labels_sha256': guarded.array_sha(y), 'valid_sha256': guarded.array_sha(valid),
        'settings': list(SETTINGS), 'forest_trees': TREES, 'prior_weight': .01,
        'held_math_questions': 256, 'pooled_control_questions': 512, 'control_C': .1,
        'outer_folds': 3, 'inner_folds': 3, 'all_prefixes_grouped_by_question': True,
        'screen': 'Lower joint loss AND mean-three marginal Brier on all held math questions',
        'new_neural_calls': 0, 'cloud_calls': 0, 'development_or_test_read': False,
        'controller_accuracy_claim': False}
    return data, record


def run(args, data, record):
    p, q = np.full((1536, 8), np.nan), np.full((1536, 8), np.nan); receipts = []
    for fold in range(3):
        held, old, new, receipt = outer(data, fold); p[held] = old; q[held] = new; receipts.append(receipt)
        print({'outer_fold': fold, 'selected': receipt['selected']}, flush=True)
    math = np.array([r['subject'] == 'math' for r in data['records']]); indices = np.flatnonzero(math)
    if not np.isfinite(p[math]).all() or np.isfinite(q[~math]).any(): raise ValueError('Whole math scope required')
    records = [data['records'][i] for i in indices]; ids = sorted({r['id'] for r in records})
    groups = [[i for i, r in enumerate(records) if r['id'] == qid] for qid in ids]
    old, new = losses(data['y'][math], p[math]), losses(data['y'][math], q[math])
    draws = np.random.default_rng(238031).integers(0, 256, (2000, 256)); reports = {}
    for name in old:
        delta = np.array([(new[name][g]-old[name][g]).mean() for g in groups])
        reports[name] = {'pooled_control': float(old[name].mean()), 'math_specialized': float(new[name].mean()),
            'delta': float(delta.mean()), 'paired_question_CI95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()}
    passed = all(reports[n]['delta'] < 0 for n in ('joint_log_loss', 'mean_three_marginal_Brier'))
    io.npz(args.output/'private-oof.npz', {'indices': indices, 'control': p[math], 'specialized': q[math]})
    io.write(args.output/'fold-receipts.json', receipts)
    io.write(args.output/'summary.json', {'schema': SCHEMA, 'reports': reports, 'screen_passed': passed,
        'math_questions': 256, 'new_neural_calls': 0, 'cloud_calls': 0, 'development_or_test_read': False,
        'controller_accuracy_claim': False, 'paired_CI_scope': 'Descriptive, fitted folds fixed'})
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'screen_passed': passed, 'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}})
    print({'screen_passed': passed, 'reports': reports}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'run'))
    for n in ('base_fit', 'output'): p.add_argument('--'+n.replace('_', '-'), type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with guarded.training_only_reads(), guarded.guarded.no_model_calls(), threadpool_limits(limits=1):
        data, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered math diagnostic')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior outputs')
            run(args, data, record)


if __name__ == '__main__': main()
