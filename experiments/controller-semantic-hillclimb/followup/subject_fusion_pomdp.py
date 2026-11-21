"""Fuse grouped neural observations with subject-dependent calibration."""

from pathlib import Path
import json

import numpy as np

from fusion_pomdp import Observer as FusionObserver, fit as initial_fit, evaluate
import dynamic_grounded_pomdp as perception
import difficulty_pomdp as core


class Observer(FusionObserver):
    def calibration_features(self, row, path, logits):
        z = np.asarray(logits, dtype=float) / self.temperature
        z -= z.mean()
        subjects = [float(row['subject'] == s) for s in ('math', 'chemistry', 'physics')]
        return [*[value * flag for flag in subjects for value in z],
                *subjects, *[float(bool(path)) * flag for flag in subjects]]


def fit(rows, trees, judges, directory, metadata, outputs, smoothing=32., late=True,
        semantic_review=False, context_weight=0.):
    from sklearn.linear_model import LogisticRegression
    initial = initial_fit(rows, trees, judges, directory, metadata, outputs,
                          smoothing=smoothing, late=late, semantic_review=semantic_review,
                          context_weight=context_weight)
    observer = Observer(**vars(initial.classifier))
    by_id = {r['id']: r for r in rows}
    lookup = {(r['id'], r['kind'], r['path']): value for r, value in zip(metadata, outputs['state'])}
    training = [r for r in metadata if r['kind'] == 'solver' and r['train'] and r['valid']]
    x = [observer.calibration_features(by_id[r['id']], r['path'], lookup[r['id'], 'solver', r['path']])
         for r in training]
    observer.calibration = LogisticRegression(C=1., max_iter=1500).fit(x, [r['label'] for r in training])
    scores = []
    for row in rows:
        for path in trees[row['id']]:
            p = np.zeros(4)
            features = observer.calibration_features(row, path, lookup[row['id'], 'solver', path])
            p[observer.calibration.classes_.astype(int)] = observer.calibration.predict_proba([features])[0]
            scores.append(p)
    recipe = json.loads((Path(directory) / 'recipe.json').read_bytes())
    price = recipe['parameters'] / 1e9 * recipe['max_length'] / 3.09
    extra = {(row['id'], path): price if path in ('repair', 'retry', 'guided') else 0.
             for row in rows for path in trees[row['id']]}
    rebuilt = core.fit(rows, trees, judges, leaf=0, smoothing=smoothing, late_feedback=late,
                        stage_review=True, classifier_override=observer, scores_override=np.asarray(scores),
                        extra_costs=extra, review_observer=None)
    receipt = {**initial.fit_receipt, **rebuilt.fit_receipt,
               'subject_dependent_calibration': True,
               'calibration_inputs': 'Both excluded-question logit vectors, known subject, known decision stage',
               'calibration_training': 'Existing training labels only; no new encoder fitting'}
    return perception.Planner(observer, rebuilt.dynamics, receipt, review_observer=None,
                              context_weight=context_weight, observation_cohorts=initial.observation_cohorts)
