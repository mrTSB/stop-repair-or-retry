"""Fixed sparse-text joint-state forecasts; no POMDP runtime changes yet."""

from dataclasses import dataclass
import hashlib
import json
import time
import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import repairability_pomdp as core

SEED = 2026112311
VARIANTS = ('numeric_C1', 'word12_C1', 'char35_C1')
TEXT_PARAMETERS = {'lowercase': False, 'max_features': 8192, 'min_df': 1,
                   'max_df': 1.0, 'sublinear_tf': False, 'use_idf': True,
                   'smooth_idf': True, 'norm': 'l2', 'binary': False,
                   'strip_accents': None, 'stop_words': None,
                   'token_pattern': r'(?u)\b\w\w+\b'}
MODEL_PARAMETERS = {'C': 1., 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 1e-4,
                    'fit_intercept': True, 'class_weight': None, 'random_state': SEED}
PATHS = ('', 'repair', 'retry')
METRICS = ('joint_log_loss', 'joint_brier', 'current_brier', 'repair_brier', 'retry_brier')
PANELS = ('all', 'root', 'continuation', 'repair', 'retry', *[
    subject + '_' + stage for subject in ('math', 'chemistry', 'physics') for stage in ('all', 'root', 'continuation')])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def fold_for(qid):
    return int(hashlib.sha256(str(qid).encode()).hexdigest()[:8], 16) % 3


def paid_text(row, path, tree):
    """Whitelist original question and the current paid response ancestry."""
    if path not in PATHS:
        raise ValueError('Text observations are only available at the three decision nodes')
    values = ['QUESTION:\n' + row['question'], 'ROOT RESPONSE:\n' + tree['']['text']]
    if path:
        values.extend(['PAID ACTION:\n' + path, 'CURRENT RESPONSE:\n' + tree[path]['text']])
    return '\n\n'.join(values)


def table(rows, trees, neural_lookup):
    texts, numeric, labels, ids, paths, subjects = [], [], [], [], [], []
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate training question')
    for row in rows:
        qid = row['id']; tree = trees[qid]
        for path in PATHS:
            children = [(path + '/' if path else '') + action for action in ('repair', 'retry')]
            if any(p not in tree for p in (path, *children)):
                raise ValueError('Every question requires all seven plain paired responses')
            paid = {p: tree[p] for p in ('', path)}
            safe = {**row, 'answer': ''}
            texts.append(paid_text(safe, path, paid))
            numeric.append(core.observables(safe, path, paid, neural_lookup[qid, path]))
            labels.append(core.state(core.correct(tree[path], row['answer']),
                                     core.correct(tree[children[0]], row['answer']),
                                     core.correct(tree[children[1]], row['answer'])))
            ids.append(qid); paths.append(path); subjects.append(row['subject'])
    numeric = np.asarray(numeric, dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError('Nonfinite numeric observations')
    return {'texts': texts, 'numeric': numeric, 'labels': np.asarray(labels, dtype=int),
            'ids': np.asarray(ids), 'paths': np.asarray(paths), 'subjects': np.asarray(subjects)}


def vectorizer(variant):
    if variant == 'numeric_C1':
        return None
    if variant not in VARIANTS:
        raise ValueError('Use only the two prespecified sparse-text variants or numeric control')
    word = variant == 'word12_C1'
    parameters = {**TEXT_PARAMETERS, 'analyzer': 'word' if word else 'char',
                  'ngram_range': (1, 2) if word else (3, 5)}
    # The char analyzer does not tokenize words; recording token_pattern=None
    # makes that distinction explicit and avoids an unused-parameter warning.
    if not word:
        parameters['token_pattern'] = None
    return TfidfVectorizer(**parameters)


@dataclass
class Forecaster:
    variant: str
    scaler: object
    text: object
    classifier: object
    convergence_warnings: list

    def matrix(self, texts, numeric):
        numeric = sparse.csr_matrix(self.scaler.transform(np.asarray(numeric, dtype=float)))
        if self.text is None:
            return numeric
        return sparse.hstack((numeric, self.text.transform(texts)), format='csr')

    def predict_proba(self, texts, numeric):
        matrix = self.matrix(texts, numeric)
        probabilities = np.full((len(texts), 8), 1e-9)
        probabilities[:, self.classifier.classes_.astype(int)] += self.classifier.predict_proba(matrix)
        probabilities /= probabilities.sum(1, keepdims=True)
        return probabilities


def fit_forecaster(texts, numeric, labels, variant):
    scaler = StandardScaler().fit(numeric)
    numeric_matrix = sparse.csr_matrix(scaler.transform(numeric))
    text = vectorizer(variant)
    matrix = numeric_matrix if text is None else sparse.hstack((numeric_matrix, text.fit_transform(texts)), format='csr')
    classifier = LogisticRegression(**MODEL_PARAMETERS)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always', ConvergenceWarning)
        classifier.fit(matrix, labels)
    messages = [str(r.message) for r in recorded if issubclass(r.category, ConvergenceWarning)]
    return Forecaster(variant, scaler, text, classifier, messages)


def outer_prediction(data, variant, held_fold):
    folds = np.asarray([fold_for(qid) for qid in data['ids']])
    training, held = np.flatnonzero(folds != held_fold), np.flatnonzero(folds == held_fold)
    fit_ids, held_ids = set(data['ids'][training]), set(data['ids'][held])
    if not fit_ids or not held_ids or fit_ids & held_ids:
        raise ValueError('Require three disjoint whole-question folds')
    model = fit_forecaster([data['texts'][i] for i in training], data['numeric'][training], data['labels'][training], variant)
    held_texts = [data['texts'][i] for i in held]
    probabilities = model.predict_proba(held_texts, data['numeric'][held])
    transformed = model.matrix(held_texts, data['numeric'][held])
    receipt = {'fold': held_fold, 'fit_questions': len(fit_ids), 'held_questions': len(held_ids),
               'fit_rows': len(training), 'held_rows': len(held), 'question_overlap': 0,
               'fit_ids_sha256': fingerprint(sorted(fit_ids)), 'held_ids_sha256': fingerprint(sorted(held_ids)),
               'classes': model.classifier.classes_.tolist(), 'feature_columns': transformed.shape[1],
               'numeric_columns': data['numeric'].shape[1], 'vocabulary_size': 0 if model.text is None else len(model.text.vocabulary_),
               'vocabulary_sha256': None if model.text is None else fingerprint({term: int(index) for term, index in model.text.vocabulary_.items()}),
               'idf_sha256': None if model.text is None else array_sha(model.text.idf_),
               'scaler_mean_sha256': array_sha(model.scaler.mean_), 'scaler_scale_sha256': array_sha(model.scaler.scale_),
               'classifier_coefficients_sha256': array_sha(model.classifier.coef_),
               'classifier_intercept_sha256': array_sha(model.classifier.intercept_),
               'iterations': model.classifier.n_iter_.tolist(), 'convergence_warnings': model.convergence_warnings,
               'mean_nonzero_features': float(transformed.getnnz(axis=1).mean()),
               'mean_linear_dot_flops': float(2 * transformed.getnnz(axis=1).mean() * len(model.classifier.classes_)),
               'temperature': 1., 'additional_calibration': False}
    # One fixed, label-independent question subset per held fold measures the
    # added CPU head, including text transformation, on single-node requests.
    timed_ids = set(sorted(held_ids, key=lambda q: hashlib.sha256(('sparse-latency:' + q).encode()).hexdigest())[:16])
    elapsed = []
    for position, i in enumerate(held):
        if data['ids'][i] not in timed_ids:
            continue
        start = time.perf_counter()
        single = model.predict_proba([data['texts'][i]], data['numeric'][i:i+1])[0]
        elapsed.append(time.perf_counter() - start)
        np.testing.assert_allclose(single, probabilities[position], atol=1e-10)
    receipt['head_cpu_latency'] = {'nodes': len(elapsed), 'question_ids_sha256': fingerprint(sorted(timed_ids)),
        'median_ms': float(1000 * np.median(elapsed)), 'p95_ms': float(1000 * np.quantile(elapsed, .95)),
        'mean_ms': float(1000 * np.mean(elapsed)), 'neural_calls': 0,
        'scope': 'Numeric scaling, optional sparse text transform, and multinomial prediction; same hardware, sequential requests'}
    return held, probabilities, model, receipt


def losses(labels, probabilities):
    y = np.asarray(labels, dtype=int); p = np.asarray(probabilities)
    if p.shape != (len(y), 8) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1.):
        raise ValueError('Malformed eight-state held-question forecasts')
    return {'joint_log_loss': -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.)),
            'joint_brier': ((p - np.eye(8)[y]) ** 2).sum(1),
            'current_brier': (p @ core.CURRENT - core.CURRENT[y]) ** 2,
            'repair_brier': (p @ core.AFTER[0] - core.AFTER[0, y]) ** 2,
            'retry_brier': (p @ core.AFTER[1] - core.AFTER[1, y]) ** 2}


def panel_mask(data, panel):
    subjects, paths = data['subjects'], data['paths']
    subject, stage = panel.split('_', 1) if '_' in panel else (None, panel)
    mask = np.ones(len(paths), dtype=bool) if subject is None else subjects == subject
    if stage == 'root':
        mask &= paths == ''
    elif stage == 'continuation':
        mask &= paths != ''
    elif stage in ('repair', 'retry'):
        mask &= paths == stage
    elif stage != 'all':
        raise ValueError('Unknown preregistered panel')
    return mask


def diagnostics(data, predictions):
    by_model = {name: losses(data['labels'], values) for name, values in predictions.items()}
    reports = {}
    for panel in PANELS:
        mask = panel_mask(data, panel); qids = sorted(set(data['ids'][mask]))
        if not qids:
            raise ValueError('Empty registered subject/stage panel')
        groups = [np.flatnonzero(mask & (data['ids'] == qid)) for qid in qids]
        draws = np.random.default_rng(SEED).integers(0, len(qids), size=(2000, len(qids)))
        reports[panel] = {'questions': len(qids), 'decision_nodes': int(mask.sum()), 'models': {}, 'paired_differences': {}}
        for name, values in by_model.items():
            reports[panel]['models'][name] = {metric: float(np.mean([values[metric][g].mean() for g in groups])) for metric in METRICS}
            if name == 'numeric_C1':
                continue
            difference = {}
            for metric in METRICS:
                node_diff = values[metric] - by_model['numeric_C1'][metric]
                per_question = np.asarray([node_diff[g].mean() for g in groups])
                difference[metric] = {'point': float(per_question.mean()),
                    'ci95': np.quantile(per_question[draws].mean(1), [.025, .975]).tolist(), 'lower_is_better': True}
            reports[panel]['paired_differences'][name + '_minus_numeric_C1'] = difference
    return reports
