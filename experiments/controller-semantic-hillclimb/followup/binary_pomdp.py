"""Two-state POMDP with a trained correctness observer and paid LLM observations."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dynamic_grounded_pomdp as perception
from context_pomdp import REVIEW_WIDTH
import binary_core as core


@dataclass
class Observer(perception.Observer):
    classes_: object = field(default_factory=lambda: np.arange(2))

    def row_probabilities(self, row, path, tree):
        logits = self.values(row, path, tree)[0]
        features = self.calibration_features(row, path, logits)
        p = np.zeros(2)
        p[self.calibration.classes_.astype(int)] = self.calibration.predict_proba([features])[0]
        return p


@dataclass
class Planner(core.Planner):
    context_weight: float = 0.
    observation_cohorts: dict = field(default_factory=dict)

    def context(self, row, path, tree, stage, reviewed, weight):
        if weight != 0.:
            raise ValueError('The binary model uses its learned conditional transition tables')
        d = self.dynamics[row['subject']]
        return d.reviewed_transition[stage] if reviewed else d.transition[stage]

    def likelihood(self, row, path, tree):
        prior, validity = self.observation_cohorts[row['subject'], int(bool(path))]
        if not tree[path]['answer']:
            return 1 - validity
        return self.classifier.row_probabilities(row, path, tree) / prior * validity


def fit(rows, trees, judges, directory, metadata, outputs, smoothing=32., late=True,
        semantic_review=True, context_weight=0.):
    from sklearn.linear_model import LogisticRegression
    directory = Path(directory)
    summary = json.loads((directory.parent / 'training-summary.json').read_bytes())
    recipe = json.loads((directory / 'recipe.json').read_bytes())
    observer = Observer(str(directory.resolve()), summary['temperature_selected_on_training_oof'])
    lookup = {(r['id'], r['kind'], r['path']): v for r, v in zip(metadata, outputs['state'])}
    by_id = {r['id']: r for r in rows}
    calibration = [r for r in metadata if r['kind'] == 'solver' and r['train'] and r['valid']]
    x = [observer.calibration_features(by_id[r['id']], r['path'], lookup[r['id'], 'solver', r['path']])
         for r in calibration]
    observer.calibration = LogisticRegression(C=1., max_iter=1000).fit(x, [r['label'] % 2 for r in calibration])
    scores = []
    for row in rows:
        for path in trees[row['id']]:
            features = observer.calibration_features(row, path, lookup[row['id'], 'solver', path])
            p = np.zeros(2)
            p[observer.calibration.classes_.astype(int)] = observer.calibration.predict_proba([features])[0]
            scores.append(p)
    cohorts = {}
    for subject in {r['subject'] for r in rows}:
        for stage in (0, 1):
            count = np.ones(2)
            valid_count = np.ones(2)
            for r in metadata:
                if (r['kind'] == 'solver' and r['train'] and int(bool(r['path'])) == stage
                        and by_id[r['id']]['subject'] == subject):
                    state = r['label'] % 2
                    count[state] += 1
                    valid_count[state] += int(r['valid'])
            validity = np.clip(valid_count / count, 1e-5, 1 - 1e-5)
            validity[1] = 1.
            cohorts[subject, stage] = (valid_count / valid_count.sum(), validity)
    price = recipe['parameters'] / 1e9 * recipe['max_length'] / 3.09
    extra = {(r['id'], path): price if path in ('repair', 'retry', 'guided') else 0.
             for r in rows for path in trees[r['id']]}
    signals = None
    if semantic_review:
        signals = {}
        for row in rows:
            for path in ('', 'repair', 'retry', 'guided'):
                judge = judges.get((row['id'], path))
                if judge is None:
                    continue
                kind = 'late-review' if path and late else 'original-review'
                probability = perception.softmax(lookup[row['id'], kind, path], observer.temperature)
                signals[row['id'], path] = perception.review_signal(probability, judge, trees[row['id']][path])
    initial = core.fit(rows, trees, judges, leaf=0, smoothing=smoothing, late_feedback=late,
                       stage_review=True, classifier_override=observer, scores_override=np.asarray(scores),
                       extra_costs=extra, review_observer=observer if semantic_review else None,
                       review_signals=signals)
    if semantic_review:
        for dynamics in initial.dynamics.values():
            dynamics.review_costs += price
    receipt = {**initial.fit_receipt, 'state_count': 2, 'semantic_observer': True,
               'semantic_review': semantic_review, 'dynamic_padding': True,
               'parameters': recipe['parameters'], 'max_length': recipe['max_length'],
               'weights': recipe['weights'], 'grouped_folds': 3,
               'current_transitions': 'learned correctness transition tables',
               'encoder_planning_price': 'conservative cap; actual unpadded slots are charged'}
    return Planner(observer, initial.dynamics, receipt,
                   review_observer=observer if semantic_review else None,
                   observation_cohorts=cohorts)


def evaluate(row, tree, judges, model, settings, family):
    if settings.get('continuous_filter') and settings.get('plan_reviews') and family == 'C':
        raise ValueError('Future-review planning uses the exact finite observation model')
    return perception.evaluate(row, tree, judges, model, settings, family)
