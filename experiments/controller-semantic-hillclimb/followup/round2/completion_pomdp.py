"""Science POMDP with a separately paid, terminal one-token completion action.

This is a new action protocol, not an unchanged-gate result. The 16 latent
states describe correctness now, after ordinary repair, after ordinary retry,
and after Complete. Complete consumes one of the two continuation calls and
submits its generated letter; probabilities from that paid call are never
available before selecting it. Math must use a separate retained A policy.
"""

from dataclasses import dataclass
import hashlib

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import repairability_pomdp as plain

STATES = np.arange(16)
CURRENT = (STATES >> 3) & 1
AFTER = np.asarray([(STATES >> 2) & 1, (STATES >> 1) & 1, STATES & 1])
ACTIONS = ('stop', 'repair', 'retry', 'complete')
N_OBS = 30
normalize = plain.normalize


def state(current, repair, retry, complete):
    return 8 * int(current) + 4 * int(repair) + 2 * int(retry) + int(complete)


def mask_invalid(p, valid):
    p = np.asarray(p, dtype=float).copy()
    if p.shape != (16,) or not np.isfinite(p).all():
        raise ValueError('Expected sixteen finite state probabilities')
    if not valid:
        p[CURRENT == 1] = 0.
        if not p.sum():
            p[CURRENT == 0] = 1.
    return normalize(p)


def probabilities(classifier, features):
    p = np.full(16, 1e-9)
    p[classifier.classes_.astype(int)] += classifier.predict_proba([features])[0]
    return normalize(p)


def observation(p, valid):
    """Retain fixability information even when the current answer is invalid."""
    p = mask_invalid(p, valid)
    gains = AFTER @ p
    best = int(np.argmax(gains))
    quality = int(gains[best] >= .5)
    if not valid:
        return 24 + 2 * best + quality
    current_bin = int(np.searchsorted([.3, .7, .97], float(p @ CURRENT)))
    return 6 * current_bin + 2 * best + quality


@dataclass
class Dynamics:
    prior: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    cost: np.ndarray

    def values(self, belief, remaining, penalty):
        if remaining not in (1, 2) or penalty < 0:
            raise ValueError('Two-call horizon and nonnegative compute penalty required')
        belief = normalize(belief)
        result = [float(belief @ CURRENT)]
        # At the next observation, each continuation has a terminal outcome.
        alpha = np.vstack([CURRENT, *[
            AFTER[a] - penalty * self.cost[1, a] / 1000 for a in range(3)]])
        for action in range(3):
            price = penalty * float(belief @ self.cost[2 - remaining, action]) / 1000
            if remaining == 1 or action == 2:
                future = float(belief @ AFTER[action])
            else:
                predicted = belief @ self.transition[action]
                future = float((alpha @ (predicted[:, None] * self.emission[action])).max(0).sum())
            result.append(future - price)
        return np.asarray(result)


@dataclass
class Planner:
    classifier: object
    forecaster: object
    dynamics: dict
    fit_receipt: dict

    def predicted(self, row, path, tree):
        if row['subject'] not in ('chemistry', 'physics') or path not in ('', 'repair', 'retry'):
            raise ValueError('This science policy observes only nonterminal plain prefixes')
        paid = {p: tree[p] for p in ('', path)}
        public = {**row, 'answer': ''}
        neural = self.classifier.values(public, path, paid)
        return mask_invalid(probabilities(self.forecaster, plain.observables(public, path, paid, neural)),
                            bool(paid[path]['answer']))


def classifier_for(x, y, regularization):
    if len(np.unique(y)) == 1:
        return DummyClassifier(strategy='prior').fit(x, y)
    return make_pipeline(StandardScaler(),
        LogisticRegression(C=regularization, max_iter=1000)).fit(x, y)


def fit(rows, trees, observer, neural_lookup, smoothing=32., regularization=.1):
    if not rows or any(r['subject'] not in ('chemistry', 'physics') for r in rows):
        raise ValueError('Completion fitting uses science training questions only')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate training question')
    labels, samples, features, states, folds = {}, [], [], [], []
    for row in rows:
        qid = row['id']
        for path in ('', 'repair', 'retry'):
            prefix = path + '/' if path else ''
            children = [prefix + action for action in ACTIONS[1:]]
            if not all(p in trees[qid] for p in (path, *children)):
                raise ValueError('Every science question needs all seven plain plus three Complete states')
            s = state(plain.correct(trees[qid][path], row['answer']), *[
                plain.correct(trees[qid][p], row['answer']) for p in children])
            labels[qid, path] = s
            samples.append((row, path)); states.append(s)
            folds.append(int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 3)
            features.append(plain.observables({**row, 'answer': ''}, path, trees[qid], neural_lookup[qid, path]))
    x, y, folds = np.asarray(features), np.asarray(states), np.asarray(folds)
    if set(folds) != {0, 1, 2}:
        raise ValueError('All three question-held forecast folds must be represented')
    oof = np.full((len(y), 16), 1e-9)
    for fold in range(3):
        train, held = folds != fold, folds == fold
        model = classifier_for(x[train], y[train], regularization)
        oof[np.ix_(held, model.classes_.astype(int))] += model.predict_proba(x[held])
    oof /= oof.sum(1, keepdims=True)
    forecaster = classifier_for(x, y, regularization)
    signals = {(row['id'], path): observation(oof[i], bool(trees[row['id']][path]['answer']))
               for i, (row, path) in enumerate(samples)}
    support = np.asarray([[[float(CURRENT[t] == AFTER[a, s]) for t in STATES]
                           for s in STATES] for a in (0, 1)])
    emission_support = np.ones((2, 16, N_OBS))
    emission_support[:, CURRENT == 1, 24:] = 0.
    encoder_cost = observer._recipe['parameters'] / 1e9 * observer._recipe['max_length'] / 3.09

    def counts(group):
        prior = np.ones(16)
        transition = .1 * support.copy()
        emission = .1 * emission_support.copy()
        prices, number = np.zeros((2, 3, 16)), np.zeros((2, 3, 16))
        for row in group:
            qid = row['id']; root = labels[qid, '']
            prior[root] += 1
            for action, child in enumerate(('repair', 'retry')):
                nxt = labels[qid, child]
                if CURRENT[nxt] != AFTER[action, root]:
                    raise ValueError('Paired transitions changed current-answer truth')
                transition[action, root, nxt] += 1
                emission[action, nxt, signals[qid, child]] += 1
            for stage, paths in ((0, ('',)), (1, ('repair', 'retry'))):
                for path in paths:
                    s = labels[qid, path]
                    for action, edge in enumerate(ACTIONS[1:]):
                        child = (path + '/' if path else '') + edge
                        node = trees[qid][child]
                        price = 4 * (node['input_tokens'] + node['output_tokens']) / 3.09
                        if stage == 0 and edge != 'complete':
                            price += encoder_cost
                        prices[stage, action, s] += price
                        number[stage, action, s] += 1
        return prior, transition, emission, prices, number

    def row_prob(x):
        return x / x.sum(-1, keepdims=True)

    pooled = counts(rows); dynamics = {}
    for subject in sorted({r['subject'] for r in rows}):
        local = counts([r for r in rows if r['subject'] == subject])
        prior = normalize(local[0] + smoothing * normalize(pooled[0]))
        transition = row_prob(local[1] + smoothing * row_prob(pooled[1]))
        emission = row_prob(local[2] + smoothing * row_prob(pooled[2]))
        pooled_cost = np.divide(pooled[3], pooled[4], out=np.full_like(pooled[3], 650.), where=pooled[4] > 0)
        cost = (local[3] + 4 * pooled_cost) / (local[4] + 4)
        dynamics[subject] = Dynamics(prior, transition, emission, cost)
    return Planner(observer, forecaster, dynamics, {
        'schema': 'science-terminal-completion-pomdp-v1', 'state_count': 16,
        'state_bits': ['current_correct', 'repair_correct', 'retry_correct', 'complete_correct'],
        'paired_questions': len(rows), 'decision_nodes': len(y), 'question_grouped_folds': 3,
        'regularization': regularization, 'smoothing': smoothing,
        'actions': list(ACTIONS), 'maximum_additional_solver_calls': 2,
        'complete_is_terminal': True, 'paid_scoring_features_used': False,
        'observation_recipe': 'Paid old-observer logits plus existing visible/numeric features; invalid fixability retained',
        'future_observations': 'Exact finite-observation two-step Bellman backup',
        'actual_counterfactuals_accessed_at_inference': False})


def evaluate(row, tree, judges, model, settings, family):
    if family != 'A' or row['subject'] not in ('chemistry', 'physics'):
        raise ValueError('Completion POMDP is a science-only A with no LLM review')
    settings = settings.get('by_subject', {}).get(row['subject'], settings)
    d, penalty = model.dynamics[row['subject']], settings['penalty']
    path, visited, decisions = '', [''], []
    public = {**row, 'answer': ''}
    belief = model.predicted(public, '', tree)
    for remaining in (2, 1):
        action = int(d.values(belief, remaining, penalty).argmax())
        decisions.append({'path': path, 'belief': belief.tolist(), 'action': ACTIONS[action],
                          'queried': False, 'information_value': 0.})
        if action == 0:
            break
        path = (path + '/' if path else '') + ACTIONS[action]
        if path not in tree:
            raise plain.MissingObservation('solver', path)
        visited.append(path)
        if action == 3 or remaining == 1:
            break
        p = model.predicted(public, path, tree)
        obs = observation(p, bool(tree[path]['answer']))
        belief = mask_invalid((belief @ d.transition[action - 1]) * d.emission[action - 1, :, obs],
                              bool(tree[path]['answer']))
    item = plain.outcome(row, tree, {}, path, visited, [], decisions)
    paid_observed = {decision['path']: tree[decision['path']] for decision in decisions}
    tokens, calls, slots, parameters = model.classifier.charges(public, paid_observed, {}, decisions, include_reviews=False)
    extra = parameters * slots / 3.09
    item.update(verifier_tokens=tokens, verifier_calls=calls, verifier_token_slots=slots,
                verifier_parameters_billions=parameters, completion_calls=int(path.endswith('complete')),
                completion_is_new_action=True)
    item['compute'] += extra
    item['total_parameter_compute'] += extra
    return item
