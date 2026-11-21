"""Eight-state POMDP over correctness and the two hidden next-action outcomes.

State bits are current correctness, repair correctness, and retry correctness.
Paired outcomes supervise fitting only. Decisions see paid response features
and predicted state probabilities, never the actual counterfactual outcomes.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from difficulty_pomdp import visible, MissingObservation
from context_pomdp import outcome
from scoring import correct
from numeric_consistency import observations as numeric_observations

STATES = np.arange(8)
CURRENT = (STATES >> 2) & 1
AFTER = np.asarray([(STATES >> 1) & 1, STATES & 1])
N_OBS = 31


def normalize(x):
    x = np.maximum(np.asarray(x, dtype=float), 0.)
    return x / x.sum() if x.sum() else np.full(len(x), 1. / len(x))


def state(c, repair, retry):
    return 4 * int(c) + 2 * int(repair) + int(retry)


def observables(row, path, tree, neural):
    paths = [''] + ([path] if path else [])
    paid = {p: tree[p] for p in paths}
    logits = np.concatenate([np.asarray(x, dtype=float) for x in neural[:3]])
    return [*np.clip(logits, -20., 20.), *visible({**row, 'answer': ''}, path, paid),
            *numeric_observations(paid[path]['text']), float(bool(path)),
            *[float(row['subject'] == s) for s in ('math', 'chemistry', 'physics')]]


def probabilities(classifier, features):
    p = np.full(8, 1e-9)
    p[classifier.classes_.astype(int)] += classifier.predict_proba([features])[0]
    return normalize(p)


def observation(p, valid):
    if not valid:
        return N_OBS - 1
    c = float(p @ CURRENT)
    gains = AFTER @ p
    return (int(np.searchsorted([.2, .5, .8, .97], c)) * 2 + int(gains[1] > gains[0])) * 3 + int(np.searchsorted([.3, .7], gains.max()))


@dataclass
class Dynamics:
    prior: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    cost: np.ndarray

    def values(self, belief, remaining, penalty):
        values = [float(belief @ CURRENT)]
        alpha = np.vstack([CURRENT, AFTER[0] - penalty * self.cost[1, 0] / 1000,
                           AFTER[1] - penalty * self.cost[1, 1] / 1000])
        for action in (0, 1):
            price = penalty * float(belief @ self.cost[2 - remaining, action]) / 1000
            if remaining == 1:
                future = float(belief @ AFTER[action])
            elif remaining == 2:
                predicted = belief @ self.transition[action]
                # Probability-weighted post-observation alpha values. This is
                # the exact finite-observation, two-step Bellman backup.
                future = float((alpha @ (predicted[:, None] * self.emission[action])).max(0).sum())
            else:
                raise ValueError('This POMDP has exactly two further solver calls')
            values.append(future - price)
        return np.asarray(values)


@dataclass
class Planner:
    classifier: object
    forecaster: object
    dynamics: dict
    fit_receipt: dict

    def predicted(self, row, path, tree):
        paid = {p: tree[p] for p in ('', path)}
        values = self.classifier.values({**row, 'answer': ''}, path, paid)
        p = probabilities(self.forecaster, observables(row, path, paid, values))
        if not tree[path]['answer']:
            p[CURRENT == 1] = 0.
            p = normalize(p)
        return p


def fit(rows, trees, observer, neural_lookup, smoothing=16., regularization=1.):
    samples, states, features, folds = [], [], [], []
    labels = {}
    for row in rows:
        qid = row['id']
        for path in ('', 'repair', 'retry'):
            children = [(path + '/' if path else '') + a for a in ('repair', 'retry')]
            if not all(p in trees[qid] for p in (path, *children)):
                raise ValueError('Every fitting question needs all seven paired plain states')
            s = state(correct(trees[qid][path], row['answer']),
                      correct(trees[qid][children[0]], row['answer']),
                      correct(trees[qid][children[1]], row['answer']))
            labels[qid, path] = s
            samples.append((row, path))
            states.append(s)
            folds.append(int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 3)
            features.append(observables(row, path, trees[qid], neural_lookup[qid, path]))
    x, y, folds = np.asarray(features), np.asarray(states), np.asarray(folds)
    oof = np.zeros((len(y), 8))
    for fold in (0, 1, 2):
        fit_mask, held = folds != fold, folds == fold
        model = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=1000)).fit(x[fit_mask], y[fit_mask])
        oof[np.ix_(held, model.classes_.astype(int))] = model.predict_proba(x[held])
    forecaster = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=1000)).fit(x, y)
    signals = {(r['id'], p): observation(oof[i], bool(trees[r['id']][p]['answer']))
               for i, (r, p) in enumerate(samples)}
    support = np.asarray([[[float(CURRENT[t] == AFTER[a, s]) for t in STATES] for s in STATES] for a in (0, 1)])
    emission_support = np.ones((2, 8, N_OBS))
    emission_support[:, CURRENT == 1, -1] = 0.
    encoder_cost = observer._recipe['parameters'] / 1e9 * observer._recipe['max_length'] / 3.09

    def counts(group):
        prior = np.ones(8)
        transition = .1 * support.copy()
        emission = .1 * emission_support.copy()
        prices, number = np.zeros((2, 2, 8)), np.zeros((2, 2, 8))
        for row in group:
            qid = row['id']; root = labels[qid, '']
            prior[root] += 1
            for action, parent in enumerate(('repair', 'retry')):
                nxt = labels[qid, parent]
                assert CURRENT[nxt] == AFTER[action, root]
                transition[action, root, nxt] += 1
                emission[action, nxt, signals[qid, parent]] += 1
            for stage, paths in ((0, ('',)), (1, ('repair', 'retry'))):
                for path in paths:
                    s = labels[qid, path]
                    for action, edge in enumerate(('repair', 'retry')):
                        child = (path + '/' if path else '') + edge
                        node = trees[qid][child]
                        price = 4 * (node['input_tokens'] + node['output_tokens']) / 3.09
                        price += encoder_cost if stage == 0 else 0.
                        prices[stage, action, s] += price
                        number[stage, action, s] += 1
        return prior, transition, emission, prices, number

    def row_prob(x):
        return x / x.sum(-1, keepdims=True)

    global_counts = counts(rows)
    dynamics = {}
    for subject in sorted({r['subject'] for r in rows}):
        local = counts([r for r in rows if r['subject'] == subject])
        prior = normalize(local[0] + smoothing * normalize(global_counts[0]))
        transition = row_prob(local[1] + smoothing * row_prob(global_counts[1]))
        emission = row_prob(local[2] + smoothing * row_prob(global_counts[2]))
        pooled_price = np.divide(global_counts[3], global_counts[4],
                                 out=np.full_like(global_counts[3], 650.), where=global_counts[4] > 0)
        cost = (local[3] + 4 * pooled_price) / (local[4] + 4)
        dynamics[subject] = Dynamics(prior, transition, emission, cost)
    return Planner(observer, forecaster, dynamics,
                   {'state_count': 8, 'state_bits': ['current_correct', 'repair_correct', 'retry_correct'],
                    'paired_questions': len(rows), 'decision_nodes': len(y), 'grouped_forecast_folds': 3,
                    'calibration_C': regularization, 'smoothing': smoothing,
                    'actual_counterfactuals_accessed_at_inference': False,
                    'solver_budget': 'Two additional calls; latest answer submitted',
                    'future_observations': 'Learned finite observation distribution; exact two-step backup'})


def evaluate(row, tree, judges, model, settings, family):
    if family != 'A':
        raise ValueError('The repairability-only controller acquires no LLM reviews')
    settings = settings.get('by_subject', {}).get(row['subject'], settings)
    d = model.dynamics[row['subject']]
    penalty = settings['penalty']
    path, visited, decisions = '', [''], []
    belief = model.predicted(row, '', tree)
    for remaining in (2, 1):
        action = int(d.values(belief, remaining, penalty).argmax())
        decisions.append({'path': path, 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                          'queried': False, 'information_value': 0.})
        if not action:
            break
        edge = ('repair', 'retry')[action - 1]
        path = (path + '/' if path else '') + edge
        if path not in tree:
            raise MissingObservation('solver', path)
        visited.append(path)
        if remaining == 2:
            p = model.predicted(row, path, tree)
            obs = observation(p, bool(tree[path]['answer']))
            belief = normalize((belief @ d.transition[action - 1]) * d.emission[action - 1, :, obs])
            if not tree[path]['answer']:
                belief[CURRENT == 1] = 0.
                belief = normalize(belief)
    item = outcome(row, tree, {}, path, visited, [], decisions)
    tokens, calls, slots, parameters = model.classifier.charges(row, tree, {}, decisions, include_reviews=False)
    extra = parameters * slots / 3.09
    item.update(verifier_tokens=tokens, verifier_calls=calls, verifier_token_slots=slots,
                verifier_parameters_billions=parameters)
    item['compute'] += extra
    item['total_parameter_compute'] += extra
    return item
