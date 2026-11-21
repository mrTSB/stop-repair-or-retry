"""Optional second critical review after a guided answer, with one action left.

The critical-review latent state is H/current/original-mentor/critical-mentor
(16 states). Prior modes keep their eight-state meanings. No mentor truth is
observed at runtime, and the original mentor bit persists through every action.
"""

from dataclasses import dataclass, field
from collections import Counter
import numpy as np

import stronger_base_hybrid as base_core
import mentor_pomdp as mentor_core
import repairability_pomdp as action_core

STATES = np.arange(16)
REWARD = (STATES >> 2) & 1
N_SIGNAL = 8


def signal(review, current):
    if not review['answer']:
        return 7
    if not review['parse_valid']:
        return 6
    confidence = int(np.searchsorted([.2, .8], review['p_correct']))
    agreement = int(mentor_core.equal(review['answer'], current['answer']))
    return 2 * confidence + agreement


def observation_support():
    mask = np.ones((16, N_SIGNAL))
    for s in STATES:
        current, critical = (s >> 2) & 1, s & 1
        for obs in range(6):
            agreement = obs % 2
            if (agreement and current != critical) or (not agreement and current == critical == 1):
                mask[s, obs] = 1e-4
        if critical:
            mask[s, 7] = 0.
    return mask


def terminal_support():
    # H, original mentor and critical mentor are persistent. Only C changes.
    return np.asarray([[float((s & 11) == (t & 11)) for t in STATES] for s in STATES])


@dataclass
class CriticalDynamics:
    lift: np.ndarray
    review: np.ndarray
    transition: np.ndarray
    costs: np.ndarray
    review_costs: np.ndarray
    _alphas: dict = field(default_factory=dict, repr=False)

    def alpha(self, penalty):
        if penalty not in self._alphas:
            self._alphas[penalty] = np.asarray([REWARD, *[
                self.transition[a] @ REWARD - penalty * self.costs[a] / 1000 for a in (0, 1)]])
        return self._alphas[penalty]

    def values(self, belief16, penalty):
        return self.alpha(penalty) @ belief16

    def query_values(self, beliefs8, penalty):
        masses = np.atleast_2d(beliefs8) @ self.lift
        joint = masses[:, :, None] * self.review[None]
        values = np.einsum('as,nso->nao', self.alpha(penalty), joint)
        return values.max(1).sum(1) - penalty * (masses @ self.review_costs) / 1000

    def query_value(self, belief8, penalty):
        return float(self.query_values(belief8, penalty)[0])


@dataclass
class Planner:
    primary: object
    critical: dict
    fit_receipt: dict

    @property
    def base(self):
        return self.primary.base

    @property
    def mentor(self):
        return self.primary.mentor


def fit_critical(records, smoothing=16.):
    """Each record is one complete registered critical-review science pair."""
    if (len(records) != 64 or len({r['id'] for r in records}) != 64 or
            Counter(r['subject'] for r in records) != {'chemistry': 32, 'physics': 32}):
        raise ValueError('Critical fitting requires all64 unique registered science pairs')
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError('Invalid critical-model smoothing')
    obs_support, trans_support = observation_support(), terminal_support()

    def counts(subject=None):
        prior = np.ones((8, 2)) * .5
        review = .1 * obs_support.copy()
        terminal = np.broadcast_to(.1 * trans_support, (2, 16, 16)).copy()
        cost_sum = np.zeros((2, 16)); number = np.zeros((2, 16))
        review_sum, review_number = np.zeros(16), np.zeros(16)
        for r in records:
            if subject is not None and r['subject'] != subject:
                continue
            s = 2 * r['state8'] + r['critical_correct']
            prior[r['state8'], r['critical_correct']] += 1
            review[s, r['signal']] += 1
            review_sum[s] += r['review_compute']; review_number[s] += 1
            for a in (0, 1):
                t = (s & ~4) | (4 * int(r['outcomes'][a]))
                terminal[a, s, t] += 1
                cost_sum[a, s] += r['solver_compute'][a]; number[a, s] += 1
        return prior, review, terminal, cost_sum, number, review_sum, review_number

    def probabilities(v):
        return v / v.sum(-1, keepdims=True)

    global_counts = counts(); result = {}
    for subject in ('chemistry', 'physics'):
        local = counts(subject)
        conditional = probabilities(local[0] + smoothing * probabilities(global_counts[0]))
        lift = np.zeros((8, 16))
        for s in range(8):
            lift[s, 2 * s:2 * s + 2] = conditional[s]
        review = probabilities(local[1] + smoothing * probabilities(global_counts[1]))
        transition = probabilities(local[2] + smoothing * probabilities(global_counts[2]))
        # Empty state cells use global observed action means, not invented labels.
        defaults = np.asarray([np.mean([r['solver_compute'][a] for r in records]) for a in (0, 1)])[:, None]
        pooled = np.divide(global_counts[3], global_counts[4],
                           out=np.broadcast_to(defaults, (2, 16)).copy(), where=global_counts[4] > 0)
        costs = (local[3] + 4 * pooled) / (local[4] + 4)
        default_review = np.mean([r['review_compute'] for r in records])
        review_pooled = np.divide(global_counts[5], global_counts[6], out=np.full(16, default_review),
                                  where=global_counts[6] > 0)
        prices = (local[5] + 4 * review_pooled) / (local[6] + 4)
        np.testing.assert_allclose(lift @ REWARD, mentor_core.REWARD)
        result[subject] = CriticalDynamics(lift, review, transition, costs, prices)
    return result


def continuation_value(d, critical, belief8, first_action, penalty, threshold):
    plain = float((d.alpha(first_action, penalty) @ belief8).max())
    if critical is None or first_action != 0:
        return plain
    return max(plain, critical.query_value(belief8, penalty) - threshold * float(belief8.sum()))


def root_values(d, critical, belief8, review_signal, penalty, threshold):
    values = [float(belief8 @ mentor_core.REWARD)]
    for action in (0, 1):
        predicted = belief8 @ d.reviewed_transition[review_signal, action]
        masses = (predicted[:, None] * d.emission[action]).T
        plain = (masses @ d.alpha(action, penalty).T).max(1)
        if critical is not None and action == 0:
            plain = np.maximum(plain, critical.query_values(masses, penalty) - threshold * masses.sum(1))
        future = float(plain.sum())
        values.append(future - penalty * float(belief8 @ d.first_costs[action]) / 1000)
    return np.asarray(values)


def root_information(d, critical, belief8, penalty, threshold):
    return float(sum(root_values(d, critical, belief8 * d.review[:, s], s, penalty, threshold).max()
                     for s in range(d.review.shape[-1])) - penalty * float(belief8 @ d.review_costs) / 1000)


def evaluate(row, tree, judges, model, settings, family):
    if family != 'C':
        raise ValueError('Second-review planning is a POMDP+LLM hybrid')
    selected = settings.get('by_subject', {}).get(row['subject'], settings)
    penalty = selected['penalty']; threshold = selected.get('critical_threshold', 0.)
    primary = model.primary; safe = {**row, 'answer': ''}
    critical = model.critical.get(row['subject'])
    base, base_settings = base_core.resolve_base(primary.base, primary.base_module, primary.base_settings, row['subject'])
    primary.prepare_observer(base.classifier)
    root = base.predicted(safe, '', base_core.prefix(tree, ''))
    ledger = {}; base_core.record_observation(ledger, base.classifier, safe, '', tree)
    belief = action_core.normalize(root @ primary.transfer[row['subject']])
    d = primary.mentor.dynamics[row['subject']]
    information = root_information(d, critical, belief, penalty, threshold) - base.dynamics[row['subject']].values(root, 2, penalty).max()
    if information <= selected.get('threshold', 0.) + 1e-9:
        fixed = {**base_settings, 'penalty': penalty}
        function = base_core.continuous.evaluate if 'posterior_weight' in fixed else action_core.evaluate
        item = function(safe, tree, {}, base, fixed, 'A')
        item['correct'] = int(action_core.correct(tree[item['path']], row['answer']))
        item['decisions'][0]['information_value'] = float(information)
        for decision in item['decisions']:
            base_core.record_observation(ledger, base.classifier, safe, decision['path'], tree)
        return base_core.bill(item, ledger, {}, row)
    if (row['id'], '') not in judges:
        raise action_core.MissingObservation('judge', '')
    root_review = judges[row['id'], '']; acquired = ['']
    probability = None
    if primary.review_observer is not None:
        primary.prepare_observer(primary.mentor.classifier)
        probability = mentor_core.perception.softmax(primary.mentor.classifier.values(safe, '', base_core.prefix(tree, ''), root_review)[0],
                                                      primary.mentor.classifier.temperature)
        base_core.record_observation(ledger, primary.mentor.classifier, safe, '', tree, root_review)
    rs = mentor_core.signal(root_review, tree[''], probability)
    belief = action_core.normalize(belief * d.review[:, rs])
    action = int(root_values(d, critical, belief, rs, penalty, threshold).argmax())
    decisions = [{'path': '', 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                  'queried': True, 'information_value': float(information)}]
    path, visited = '', ['']
    if action:
        first = action - 1; path = 'guided' if first == 0 else 'retry'
        if path not in tree:
            raise action_core.MissingObservation('solver', path)
        visited.append(path); primary.prepare_observer(primary.mentor.classifier)
        obs = primary.mentor.regular.observed(safe, path, base_core.prefix(tree, path))
        base_core.record_observation(ledger, primary.mentor.classifier, safe, path, tree)
        likelihood = d.emission[first, :, obs * 3 + mentor_core.comparison(root_review, tree[path])].copy()
        if not tree[path]['answer']:
            likelihood[mentor_core.REWARD == 1] = 0.
        belief = action_core.normalize((belief @ d.reviewed_transition[rs, first]) * likelihood)
        plain = d.alpha(first, penalty) @ belief
        critical_value = critical.query_value(belief, penalty) if critical is not None and first == 0 else -np.inf
        query = critical_value - plain.max() > threshold + 1e-9
        if query:
            if (row['id'], 'guided') not in judges:
                raise action_core.MissingObservation('judge', 'guided')
            new_review = judges[row['id'], 'guided']; acquired.append('guided')
            belief16 = action_core.normalize((belief @ critical.lift) * critical.review[:, signal(new_review, tree[path])])
            action = int(critical.values(belief16, penalty).argmax())
            decision_belief = belief16
        else:
            action = int(plain.argmax()); decision_belief = belief
        decisions.append({'path': path, 'belief': decision_belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                          'queried': bool(query), 'information_value': float(critical_value - plain.max()) if np.isfinite(critical_value) else None,
                          'critical_mentor_mode': bool(query)})
        if action:
            path += '/' + ('critical-repair' if query and action == 1 else 'reuse' if action == 1 else 'retry')
            if path not in tree:
                raise action_core.MissingObservation('solver', path)
            visited.append(path)
    item = mentor_core.outcome(row, tree, judges, path, visited, acquired, decisions)
    return base_core.bill(item, ledger, judges, row)
