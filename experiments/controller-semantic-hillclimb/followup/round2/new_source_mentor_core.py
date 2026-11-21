"""Explicit128-pair variant of the frozen retained-mentor fitter.

The inherited fit, truth-preserving transfer and build functions retain their
numerical operations. Only names, cohort guards and count receipts differ.
Tests compare these definitions with the declared parent transformations.
No frozen runtime implementation is changed; pickles retain parent classes.
"""

import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from mentor_pomdp import (Planner as MentorPlanner, JointDynamics, N_REVIEW, N_OBS,
                          state, signal, comparison, normalize, core, perception, correct)
from hybrid_repairability import (Planner as HybridPlanner, action_core, mentor_core, FIXED)

def fit_new(rows, trees, judges, directory, metadata, outputs, smoothing=16.,
        semantic_review=False, late=False, context_weight=0.):
    plain_trees = {qid: {p: n for p, n in tree.items() if not p.endswith('/reuse')} for qid, tree in trees.items()}
    regular = perception.fit(rows, plain_trees, judges, directory, metadata, outputs,
                              smoothing=smoothing, late=False, semantic_review=False, context_weight=0.)
    recipe = json.loads((Path(directory) / 'recipe.json').read_bytes())
    encoder_price = recipe['parameters'] / 1e9 * recipe['max_length'] / 3.09
    lookup = {(r['id'], r['kind'], r['path']): value for r, value in zip(metadata, outputs['state'])}
    paired = [r for r in rows if (r['id'], '') in judges]
    if len(paired) != 128:
        raise ValueError('Use all128new-source matched root training questions')
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
                raise ValueError('Finish all256new-source mentor-reuse counterfactuals')
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
               'semantic_review': semantic_review, 'paired_root_questions': 128,
               'paired_reuse_outcomes': 256, 'neural_training': 'Existing excluded-question observer outputs only',
               'equality_model': 'Joint neural-observation and candidate/proposal comparison likelihood; paid root information updated once',
               'soft_structural_mass': 1e-4, 'solver_budget': 'At most two additional compact calls'}
    return MentorPlanner(regular.classifier, dynamics, receipt, regular,
                   review_observer=regular.classifier if semantic_review else None)



def fit_transfer_new(rows, trees, judges, smoothing=16.):
    paired = [r for r in rows if (r['id'], '') in judges]
    if len(paired) != 128:
        raise ValueError('Use all128registered new-source mentor-training questions')
    support = np.asarray([[float(action_core.CURRENT[s] == mentor_core.REWARD[t])
                           for t in range(8)] for s in range(8)])
    records = []
    for row in paired:
        qid = row['id']; tree = trees[qid]
        c = int(correct(tree[''], row['answer']))
        source = action_core.state(c, correct(tree['repair'], row['answer']), correct(tree['retry'], row['answer']))
        hard = int(np.mean([correct(tree[p], row['answer']) for p in FIXED]) < .6)
        target = mentor_core.state(2 * hard + c, correct(judges[qid, ''], row['answer']))
        records.append((row['subject'], source, target))
    global_counts = .1 * support.copy()
    for _, source, target in records:
        global_counts[source, target] += 1
    pooled = global_counts / global_counts.sum(1, keepdims=True)
    result = {}
    for subject in sorted({r['subject'] for r in rows}):
        counts = .1 * support.copy()
        for sub, source, target in records:
            if sub == subject:
                counts[source, target] += 1
        counts += smoothing * pooled
        result[subject] = counts / counts.sum(1, keepdims=True)
        np.testing.assert_allclose(result[subject] @ mentor_core.REWARD, action_core.CURRENT, atol=1e-12)
    return result



def build_new(base, mentor, rows, trees, judges, smoothing=16.):
    recipe_a = hashlib.sha256((Path(base.classifier.directory) / 'recipe.json').read_bytes()).hexdigest()
    recipe_m = hashlib.sha256((Path(mentor.classifier.directory) / 'recipe.json').read_bytes()).hexdigest()
    if recipe_a != recipe_m:
        raise ValueError('The two hybrid modes must use exactly the same neural weights and input profile')
    return HybridPlanner(base, mentor, fit_transfer_new(rows, trees, judges, smoothing),
                   {'state_count': 16, 'observed_modes': ['unqueried_action_fixability', 'queried_retained_mentor'],
                    'states_per_observed_mode': 8, 'base_A': base.fit_receipt, 'mentor': mentor.fit_receipt,
                    'transfer_smoothing': smoothing, 'maximum_reviews': 1,
                    'observer_recipe_sha256': recipe_a,
                    'solver_budget': 'At most two additional compact calls',
                    'shared_neural_passes_require_identical_assets': True})

