"""Correctness POMDP with grouped calibration and incoming-action-specific priors."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

import binary_pomdp as binary
import binary_core as core
import dynamic_grounded_pomdp as perception
import difficulty_pomdp as original
from scoring import correct


def cohort(path):
    return path if path in ('', 'repair', 'retry', 'guided') else 'continuation'


@dataclass
class Observer(perception.Observer):
    def features(self, row, path, logits, numeric):
        z = np.asarray(logits, dtype=float) / self.temperature
        z -= z.mean()
        return [*z, *numeric, *[float(cohort(path) == p) for p in ('', 'repair', 'retry', 'guided')],
                *[float(row['subject'] == s) for s in ('math', 'chemistry', 'physics', 'biology')]]

    def row_probabilities(self, row, path, tree):
        x = self.features(row, path, self.values(row, path, tree)[0], original.visible(row, path, tree))
        p = np.zeros(2)
        p[self.calibration.classes_.astype(int)] = self.calibration.predict_proba([x])[0]
        return p


@dataclass
class Planner(binary.Planner):
    def likelihood(self, row, path, tree):
        prior, validity = self.observation_cohorts[row['subject'], cohort(path)]
        if not tree[path]['answer']:
            return 1 - validity
        return self.classifier.row_probabilities(row, path, tree) / prior * validity


def fit(rows, trees, judges, directory, metadata, outputs, smoothing=32., late=True,
        semantic_review=True, context_weight=0.):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    directory = Path(directory)
    recipe = json.loads((directory / 'recipe.json').read_bytes())
    summary = json.loads((directory.parent / 'training-summary.json').read_bytes())
    observer = Observer(str(directory.resolve()), summary['temperature_selected_on_training_oof'])
    by_id = {r['id']: r for r in rows}
    lookup = {(r['id'], r['kind'], r['path']): i for i, r in enumerate(metadata)}
    training = [r for r in metadata if r['kind'] == 'solver' and r['train'] and r['valid']]

    def features(r):
        i = lookup[r['id'], r['kind'], r['path']]
        return observer.features(by_id[r['id']], r['path'], outputs['state'][i], r['numeric'])

    x = np.asarray([features(r) for r in training])
    y = np.asarray([r['label'] % 2 for r in training])
    folds = np.asarray([r['fold'] for r in training])

    def learner():
        return make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000))

    observer.calibration = learner().fit(x, y)
    solver_rows = [r for r in metadata if r['kind'] == 'solver']
    calibrated = {}
    for fold in (0, 1, 2):
        fitted = learner().fit(x[folds != fold], y[folds != fold])
        group = [r for r in solver_rows if r['fold'] == fold]
        probabilities = fitted.predict_proba(np.asarray([features(r) for r in group]))
        for r, value in zip(group, probabilities):
            calibrated[r['id'], r['path']] = value
    scores = np.asarray([calibrated[r['id'], p] for r in rows for p in trees[r['id']]])
    cohorts = {}
    for subject in {r['subject'] for r in rows}:
        for incoming in ('', 'repair', 'retry', 'guided'):
            count, valid = np.ones(2), np.ones(2)
            for r in metadata:
                if (r['kind'] == 'solver' and r['train'] and cohort(r['path']) == incoming
                        and by_id[r['id']]['subject'] == subject):
                    c = r['label'] % 2
                    count[c] += 1
                    valid[c] += int(r['valid'])
            validity = np.clip(valid / count, 1e-5, 1 - 1e-5)
            validity[1] = 1.
            cohorts[subject, incoming] = valid / valid.sum(), validity
    price = recipe['parameters'] / 1e9 * recipe['max_length'] / 3.09
    extra = {(r['id'], p): price if p in ('repair', 'retry', 'guided') else 0.
             for r in rows for p in trees[r['id']]}
    signals = None
    if semantic_review:
        signals = {}
        for row in rows:
            for path in ('', 'repair', 'retry', 'guided'):
                if (row['id'], path) not in judges:
                    continue
                kind = 'late-review' if path and late else 'original-review'
                p = perception.softmax(outputs['state'][lookup[row['id'], kind, path]], observer.temperature)
                signals[row['id'], path] = perception.review_signal(p, judges[row['id'], path], trees[row['id']][path])
    initial = core.fit(rows, trees, judges, leaf=0, smoothing=smoothing, late_feedback=late,
                       stage_review=True, classifier_override=observer, scores_override=scores,
                       extra_costs=extra, review_observer=observer if semantic_review else None,
                       review_signals=signals)
    if semantic_review:
        for d in initial.dynamics.values():
            d.review_costs += price
    receipt = {**initial.fit_receipt, 'state_count': 2, 'semantic_observer': True,
               'semantic_review': semantic_review, 'dynamic_padding': True,
               'parameters': recipe['parameters'], 'max_length': recipe['max_length'],
               'weights': recipe['weights'], 'grouped_folds': 3,
               'calibration': 'Question-grouped logistic calibration of neural logits and visible numeric features',
               'cohort_priors': 'Subject and actual incoming solver action, rather than pooled continuations',
               'encoder_planning_price': 'Conservative cap; actual unpadded inference slots are charged'}
    return Planner(observer, initial.dynamics, receipt,
                   review_observer=observer if semantic_review else None,
                   observation_cohorts=cohorts)


def evaluate(row, tree, judges, model, settings, family):
    return binary.evaluate(row, tree, judges, model, settings, family)
