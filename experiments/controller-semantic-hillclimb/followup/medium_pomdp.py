"""Hybrid with a brief initial judge and a learned, deeper final-checkpoint review.

The two compact-solver-call horizon, plain actions, and ordinary B reference
remain fixed. The optional stage-one guided repair uses the paid independent
review and its separately declared answer-first compact response template.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dynamic_grounded_pomdp as four
import difficulty_pomdp as core
from context_pomdp import mentor_comparison
from scoring import correct
import binary_pomdp as binary


def merged_observations(metadata, outputs, additional, predictions):
    if len(metadata) != len(outputs['state']) or len(additional) != len(predictions['state']):
        raise ValueError('Neural metadata and prediction counts differ')
    def key(row):
        return row['id'], row['kind'], row['path']
    replaced = {key(r) for r in additional}
    if len(replaced) != len(additional):
        raise ValueError('Duplicate supplemental observation')
    indices = [i for i, r in enumerate(metadata) if key(r) not in replaced]
    result = [metadata[i] for i in indices] + additional
    scores = {name: np.concatenate((outputs[name][indices], predictions[name]), axis=0)
              for name in ('state', 'plain', 'guided')}
    if len({key(r) for r in result}) != len(result):
        raise ValueError('Duplicate merged observation')
    return result, scores


def root_mentor(model, rows, trees, judges, metadata, outputs, smoothing):
    """Only root review comparisons can inform a subsequent decision.

    Stage-one repairs are forced submissions, so mixing their different review
    protocol into the root reuse likelihood would introduce model mismatch.
    """
    lookup = {(r['id'], r['kind'], r['path']): v for r, v in zip(metadata, outputs['state'])}
    for subject, d in model.dynamics.items():
        counts = np.zeros_like(d.mentor)
        pool = np.ones((4, 3))
        for row in rows:
            if row['subject'] != subject or (row['id'], '') not in judges:
                continue
            qid = row['id']
            tree = trees[qid]
            hard = int(np.mean([correct(tree[p], row['answer']) for p in core.FIXED]) < .6)
            for path in ('guided', 'retry'):
                if path not in tree:
                    continue
                features = model.classifier.calibration_features(row, path, lookup[qid, 'solver', path])
                probabilities = np.zeros(4)
                calibration = model.classifier.calibration
                probabilities[calibration.classes_.astype(int)] = calibration.predict_proba([features])[0]
                observed = core.observation(probabilities, bool(tree[path]['answer']))
                state = 2 * hard + int(correct(tree[path], row['answer']))
                comparison = mentor_comparison(judges[qid, ''], tree[path])
                counts[observed, state, comparison] += 1
                pool[state, comparison] += 1
        pool /= pool.sum(-1, keepdims=True)
        counts += smoothing * pool[None]
        d.mentor = counts / counts.sum(-1, keepdims=True)


def fit(rows, trees, judges, directory, metadata, outputs, smoothing=32.,
        semantic_review=True, state_count=4):
    module = {2: binary, 4: four}[state_count]
    model = module.fit(rows, trees, judges, directory, metadata, outputs,
                       smoothing=smoothing, late=True, semantic_review=semantic_review,
                       context_weight=0.)
    if state_count == 4:
        root_mentor(model, rows, trees, judges, metadata, outputs, smoothing)
    model.fit_receipt.update(
        medium_late_review=True, medium_training_questions=192,
        late_solver_template='Final first, at most35justification words,128output tokens',
        initial_review='Unchanged ordinary brief protocol',
        late_review='Independent1024thought +512forced JSON; all phases charged',
        mentor_estimation='root reviewed continuations only', state_count=state_count)
    return model


def evaluate(row, tree, judges, model, settings, family):
    if settings.get('continuous_filter') and settings.get('plan_reviews') and family == 'C':
        raise ValueError('Future review backups require finite observations')
    return four.evaluate(row, tree, judges, model, settings, family)
