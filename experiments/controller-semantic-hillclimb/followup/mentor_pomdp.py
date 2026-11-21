"""Hybrid POMDP tracking current correctness and persistent paid-mentor correctness.

The hidden state is (question difficulty, current correctness, mentor
correctness). One paid root review can be reused by the second compact repair.
The two further solver-call budget is unchanged. No mentor answer is submitted
directly, and no inference-time ground truth is available.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from banks import committed
from root_bank import view as root_view
from reuse_pairs import PROTOCOL as REUSE_PROTOCOL
from reasoning_pilot import short_repair
from context_pomdp import equal, encode, outcome
from scoring import correct
import dynamic_grounded_pomdp as perception
import difficulty_pomdp as core
import extended_acquire as old

N_REVIEW = 32
N_OBS = core.N_OBS
REWARD = np.array([0., 0., 1., 1., 0., 0., 1., 1.])


def comparison(judge, node):
    if not judge['answer'] or not node['answer']:
        return 0
    return 2 if equal(judge['answer'], node['answer']) else 1


def signal(judge, current, probability=None):
    if not judge['answer']:
        return 31
    if not judge['parse_valid']:
        return 30
    return encode(judge, current) if probability is None else perception.review_signal(probability, judge, current)


def state(base, mentor):
    return 2 * base + mentor


def normalize(x):
    total = x.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError('Impossible joint mentor observation')
    return x / total


def reuse_view(rows, trees, root_observations, reuse_observations, validate=False):
    import hashlib
    local, reviews = root_view(rows, trees, root_observations, validate=validate)
    by_id = {r['id']: r for r in rows}
    known = set()
    for item in reuse_observations:
        qid, path, node = item['id'], item['path'], item['response']
        if (qid, path) in known or qid not in by_id or item['kind'] != 'solver':
            raise ValueError('Duplicate or unexpected mentor-reuse response')
        known.add((qid, path))
        if path not in ('guided/reuse', 'retry/reuse') or node['protocol'] != REUSE_PROTOCOL:
            raise ValueError('Unexpected mentor-reuse protocol')
        parent = path.rsplit('/', 1)[0]
        if parent not in local[qid] or (qid, '') not in reviews:
            raise ValueError('Mentor reuse lacks its paid prefix')
        if validate:
            prompt = short_repair({**by_id[qid], 'answer': ''}, local[qid][parent], reviews[qid, ''])
            logical = 'repair/repair' if parent == 'guided' else 'retry/repair'
            seed = int(hashlib.sha256(f'{old.SEED}:{qid}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
            assert node['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
            assert node['seed'] == seed and node['output_cap'] == 128 and node['model_digest'] == old.MODEL_DIGEST
        local[qid][path] = node
    return local, reviews


@dataclass
class JointDynamics:
    prior: np.ndarray
    initial: np.ndarray
    transition: np.ndarray
    review: np.ndarray
    reviewed_transition: np.ndarray
    emission: np.ndarray
    last_transition: np.ndarray
    first_costs: np.ndarray
    last_costs: np.ndarray
    review_costs: np.ndarray
    mentor: np.ndarray
    _alphas: dict = field(default_factory=dict, repr=False)

    @property
    def reviewed_emission(self):
        return self.emission

    def alpha(self, first_action, penalty):
        key = first_action, penalty
        if key not in self._alphas:
            self._alphas[key] = np.array([REWARD, *[
                self.last_transition[first_action, a] @ REWARD - penalty * self.last_costs[first_action, a] / 1000
                for a in (0, 1)]])
        return self._alphas[key]

    def values(self, belief, review_signal, penalty):
        # Homogeneity keeps the review-signal probability in an unnormalised
        # posterior, avoiding an extra normalisation in expected information.
        values = np.array([belief @ REWARD, 0., 0.])
        for action in (0, 1):
            predicted = belief @ self.reviewed_transition[review_signal, action]
            masses = (predicted[:, None] * self.emission[action]).T
            future = (masses @ self.alpha(action, penalty).T).max(1).sum()
            values[action + 1] = future - penalty * (belief @ self.first_costs[action]) / 1000
        return values

    def information_value(self, belief, penalty):
        total = sum(self.values(belief * self.review[:, r], r, penalty).max() for r in range(N_REVIEW))
        return float(total - penalty * (belief @ self.review_costs) / 1000)


@dataclass
class Planner:
    classifier: object
    dynamics: dict
    fit_receipt: dict
    regular: object
    review_observer: object = None


def fit(rows, trees, judges, directory, metadata, outputs, smoothing=16.,
        semantic_review=False, late=False, context_weight=0.):
    plain_trees = {qid: {p: n for p, n in tree.items() if not p.endswith('/reuse')} for qid, tree in trees.items()}
    regular = perception.fit(rows, plain_trees, judges, directory, metadata, outputs,
                              smoothing=smoothing, late=False, semantic_review=False, context_weight=0.)
    recipe = json.loads((Path(directory) / 'recipe.json').read_bytes())
    encoder_price = recipe['parameters'] / 1e9 * recipe['max_length'] / 3.09
    lookup = {(r['id'], r['kind'], r['path']): value for r, value in zip(metadata, outputs['state'])}
    paired = [r for r in rows if (r['id'], '') in judges]
    if len(paired) != 192:
        raise ValueError('Use all192matched root training questions')
    hard = {r['id']: int(np.mean([correct(trees[r['id']][p], r['answer']) for p in core.FIXED]) < .6) for r in rows}

    def base_state(row, path):
        return 2 * hard[row['id']] + int(correct(trees[row['id']][path], row['answer']))

    def observed(row, path):
        features = regular.classifier.calibration_features(row, path, lookup[row['id'], 'solver', path])
        p = np.zeros(4)
        p[regular.classifier.calibration.classes_.astype(int)] = regular.classifier.calibration.predict_proba([features])[0]
        return core.observation(p, bool(trees[row['id']][path]['answer']))

    records = []
    for row in paired:
        qid = row['id']; judge = judges[qid, '']; tree = trees[qid]
        mentor_correct = int(correct(judge, row['answer']))
        probability = (perception.softmax(lookup[qid, 'original-review', ''], regular.classifier.temperature)
                       if semantic_review else None)
        rsignal = signal(judge, tree[''], probability)
        s = state(base_state(row, ''), mentor_correct)
        first, ends = [], []
        for action, parent in enumerate(('guided', 'retry')):
            if parent + '/reuse' not in tree:
                raise ValueError('Finish all384mentor-reuse counterfactuals')
            next_s = state(base_state(row, parent), mentor_correct)
            first.append((next_s, observed(row, parent) * 3 + comparison(judge, tree[parent]),
                          (tree[parent]['input_tokens'] + tree[parent]['output_tokens']) * 4 / 3.09 + encoder_price))
            last = []
            for child in (parent + '/reuse', parent + '/retry'):
                node = tree[child]
                last.append((state(base_state(row, child), mentor_correct),
                             (node['input_tokens'] + node['output_tokens']) * 4 / 3.09))
            ends.append(last)
        price = (judge['input_tokens'] + judge['output_tokens']) * 3 / 3.09 + (encoder_price if semantic_review else 0.)
        records.append((row['subject'], s, rsignal, first, ends, price))

    # Preserve difficulty and mentor truth across every solver action.
    support = np.zeros((8, 8))
    for s in range(8):
        for c in (0, 1):
            support[s, state(2 * (s // 4) + c, s % 2)] = 1.
    review_mask = np.ones((8, N_REVIEW))
    obs_mask = np.ones((8, N_OBS * 3))
    for s in range(8):
        c, m = (s // 2) % 2, s % 2
        for r in range(30):
            agree = r % 2
            if (agree and c != m) or (not agree and c == m == 1):
                review_mask[s, r] = 1e-4
        if m:
            review_mask[s, 31] = 0.
        for o in range(N_OBS):
            for e in range(3):
                if (c and o == 0) or (o == 0 and e != 0) or (e == 2 and c != m) or (c == m == 1 and e != 2):
                    obs_mask[s, o * 3 + e] = 1e-4

    def counts(group):
        k = np.ones((4, 2)); r = .5 * review_mask.copy()
        t0 = np.zeros((N_REVIEW, 2, 8, 8))
        t1 = np.broadcast_to(.5 * support, (2, 2, 8, 8)).copy()
        emission = np.broadcast_to(.1 * obs_mask, (2, 8, N_OBS * 3)).copy()
        first_sum = np.zeros((2, 8)); first_n = np.zeros_like(first_sum)
        last_sum = np.zeros((2, 2, 8)); last_n = np.zeros_like(last_sum)
        judge_sum = np.zeros(8); judge_n = np.zeros(8)
        for _, s, rs, first, ends, price in group:
            k[s // 2, s % 2] += 1; r[s, rs] += 1
            judge_sum[s] += price; judge_n[s] += 1
            for action, (nxt, obs, cost) in enumerate(first):
                t0[rs, action, s, nxt] += 1
                emission[action, nxt, obs] += 1
                first_sum[action, s] += cost; first_n[action, s] += 1
                for a, (final, cost) in enumerate(ends[action]):
                    t1[action, a, nxt, final] += 1
                    last_sum[action, a, nxt] += cost; last_n[action, a, nxt] += 1
        return k, r, t0, t1, emission, first_sum, first_n, last_sum, last_n, judge_sum, judge_n

    def probabilities(a):
        return a / a.sum(-1, keepdims=True)

    global_counts = counts(records)
    dynamics = {}
    for subject, rd in regular.dynamics.items():
        local = counts([r for r in records if r[0] == subject])
        k = probabilities(local[0] + smoothing * probabilities(global_counts[0]))
        prior = normalize((rd.prior[:, None] * k).reshape(8))
        initial = np.repeat(rd.initial, 2, axis=0)
        plain = np.zeros((2, 2, 8, 8))
        for stage in (0, 1):
            for action in (0, 1):
                for s in range(8):
                    for target in range(4):
                        plain[stage, action, s, state(target, s % 2)] = rd.transition[stage, action, s // 2, target]
        review = probabilities(local[1] + smoothing * probabilities(global_counts[1]))
        # Review-specific kernels shrink toward the actual reviewed kernel,
        # not toward the unreviewed repair kernel.
        pooled0 = probabilities(global_counts[2].sum(0) + support[None])
        t0 = probabilities(local[2] + smoothing * pooled0[None])
        t1 = probabilities(local[3] + smoothing * probabilities(global_counts[3]))
        emission = probabilities(local[4] + smoothing * probabilities(global_counts[4]))

        def price(local_sum, local_n, global_sum, global_n, default):
            pooled = np.divide(global_sum, global_n, out=np.full_like(global_sum, default), where=global_n > 0)
            return (local_sum + 4. * pooled) / (local_n + 4.)

        first_costs = price(local[5], local[6], global_counts[5], global_counts[6], 650.)
        last_costs = price(local[7], local[8], global_counts[7], global_counts[8], 650.)
        review_costs = price(local[9], local[10], global_counts[9], global_counts[10], 2800.)
        mentor = emission.reshape(2, 8, N_OBS, 3).sum(2)
        dynamics[subject] = JointDynamics(prior, initial, plain, review, t0, emission, t1,
                                           first_costs, last_costs, review_costs, mentor)
    receipt = {**regular.fit_receipt, 'state_count': 8, 'mentor_correctness_hidden': True,
               'maximum_logical_reviews': 1, 'review_availability': 'Root only; retained proposal reused without another judge call',
               'hidden_states': [f'H{h}/current{c}/mentor{m}' for h in (0, 1) for c in (0, 1) for m in (0, 1)],
               'semantic_review': semantic_review, 'paired_root_questions': 192,
               'paired_reuse_outcomes': 384, 'neural_training': 'Existing excluded-question observer outputs only',
               'equality_model': 'Joint neural-observation and candidate/proposal comparison likelihood; paid root information updated once',
               'soft_structural_mass': 1e-4, 'solver_budget': 'At most two additional compact calls'}
    return Planner(regular.classifier, dynamics, receipt, regular,
                   review_observer=regular.classifier if semantic_review else None)


def evaluate(row, tree, judges, model, settings, family):
    if family != 'C' or settings.get('plan_reviews'):
        raise ValueError('Joint mentor hybrid has one optional root review')
    rd = model.regular.dynamics[row['subject']]; d = model.dynamics[row['subject']]
    penalty = settings['penalty'] * (settings.get('science_weight', 1.) if row['domain'] == 'science' else 1.)
    o = model.regular.observed(row, '', tree)
    if settings.get('continuous_root', False):
        b4 = rd.update(rd.prior, model.regular.likelihood(row, '', tree))
        conditional = d.prior.reshape(4, 2) / rd.prior[:, None]
        belief = normalize((b4[:, None] * conditional).reshape(8))
    else:
        belief = normalize(d.prior * d.initial[:, o])
    if not tree['']['answer'] or not tree[''].get('valid', True):
        belief[REWARD == 1] = 0.
        belief = normalize(belief)
    b4 = belief.reshape(4, 2).sum(1)
    values = rd.values(b4, 2, penalty)
    information = d.information_value(belief, penalty) - values.max()
    query = information > settings.get('threshold', 0.) + 1e-9
    judge = None; acquired = []; rs = None
    if query:
        if (row['id'], '') not in judges:
            raise core.MissingObservation('judge', '')
        judge = judges[row['id'], '']; acquired.append('')
        probability = None
        if model.review_observer is not None:
            probability = perception.softmax(model.classifier.values(row, '', tree, judge)[0], model.classifier.temperature)
        rs = signal(judge, tree[''], probability)
        belief = normalize(belief * d.review[:, rs])
        values = d.values(belief, rs, penalty)
    action = int(values.argmax())
    decisions = [{'path': '', 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                  'queried': bool(query), 'information_value': float(information), 'tracked_mentor': True}]
    path = ''; visited = ['']
    if action:
        first_action = action - 1
        path = 'guided' if query and first_action == 0 else ('repair', 'retry')[first_action]
        if path not in tree:
            raise core.MissingObservation('solver', path)
        visited.append(path)
        o = model.regular.observed(row, path, tree)
        if query:
            e = comparison(judge, tree[path])
            likelihood = d.emission[first_action, :, o * 3 + e].copy()
            if not tree[path]['answer'] or not tree[path].get('valid', True):
                likelihood[REWARD == 1] = 0.
            belief = normalize((belief @ d.reviewed_transition[rs, first_action]) * likelihood)
            values = d.alpha(first_action, penalty) @ belief
        else:
            likelihood = np.repeat(rd.emission[0, first_action, :, o], 2)
            belief = normalize((belief @ d.transition[0, first_action]) * likelihood)
            values = rd.values(belief.reshape(4, 2).sum(1), 1, penalty)
        action = int(values.argmax())
        decisions.append({'path': path, 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                          'queried': False, 'information_value': None, 'tracked_mentor': True})
        if action:
            edge = 'reuse' if query and action == 1 else ('repair', 'retry')[action - 1]
            path = path + '/' + edge
            if path not in tree:
                raise core.MissingObservation('solver', path)
            visited.append(path)
    item = outcome(row, tree, judges, path, visited, acquired, decisions)
    tokens, calls, slots, parameters = model.classifier.charges(
        row, tree, judges, decisions, include_reviews=model.review_observer is not None)
    extra = parameters * slots / 3.09
    item.update(verifier_tokens=tokens, verifier_calls=calls, verifier_token_slots=slots,
                verifier_parameters_billions=parameters)
    item['compute'] += extra; item['total_parameter_compute'] += extra
    return item
