"""Question-balanced320-root retained-review POMDP, using cached TRAIN logits.

No neural inference, cloud request, development selection or held-out record
read occurs here. The runtime classes remain the existing finite-horizon
POMDP classes. This is explicitly a new weighted calibration/counting variant.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mentor_pomdp as mentor
import hybrid_repairability as hybrid
from prepare_mixed_mentor import fingerprint
import expanded_mentor_spec as spec

core, perception, correct = mentor.core, mentor.perception, mentor.correct


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def normalize(values):
    values = np.asarray(values, dtype=float)
    total = values.sum(-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(total <= 0):
        raise ValueError('Invalid probability table support')
    return values / total


@dataclass
class ConstantCalibration:
    classes_: np.ndarray

    def predict_proba(self, values):
        return np.ones((len(values), 1))


def fit_calibrator(features, labels, records):
    x, y = np.asarray(features, dtype=float), np.asarray(labels, dtype=int)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(records) or not np.isfinite(x).all():
        raise ValueError('Invalid calibration feature alignment')
    if not set(y.tolist()) <= {0, 1, 2, 3}:
        raise ValueError('Unexpected four-state calibration labels')
    weights = np.asarray(spec.question_weights(records))
    if len(set(y.tolist())) == 1:
        model = ConstantCalibration(np.asarray([int(y[0])]))
    else:
        model = LogisticRegression(C=1., max_iter=1000, random_state=spec.SEED)
        model.fit(x, y, sample_weight=weights)
    return model, spec.weight_receipt(records, weights.tolist())


def predict_calibrator(model, values):
    probabilities = np.zeros((len(values), 4))
    probabilities[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(values)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1.):
        raise ValueError('Invalid calibrated probabilities')
    return probabilities


def fit_observer(rows, trees, metadata, outputs, observer):
    """Fit weighted current-state calibration and predict whole-question OOF.

Terminal cache rows never enter this fit or its forecasts. All-invalid
questions remain present in dynamics and are explicitly counted below.
"""
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate training question')
    state = np.asarray(outputs['state'])
    if state.shape != (len(metadata), 4) or not np.isfinite(state).all():
        raise ValueError('Changed neural state dimensions')
    indices = [i for i, r in enumerate(metadata) if r['kind'] == 'solver'
               and r['path'] in spec.CALIBRATION_PATHS]
    records = []
    features = []
    hard = {qid: int(np.mean([correct(trees[qid][p], row['answer']) for p in core.FIXED]) < .6)
            for qid, row in by_id.items()}
    seen = set()
    for i in indices:
        record = metadata[i]; qid, path = record['id'], record['path']
        if qid not in by_id or (qid, path) in seen:
            raise ValueError('Unexpected/duplicate decision-node calibration input')
        seen.add((qid, path))
        row, node = by_id[qid], trees[qid][path]
        label = 2 * hard[qid] + int(correct(node, row['answer']))
        if (record['label'], record['valid'], record['train'], record['prompt_sha256']) != (
                label, bool(node['answer']), True, node['prompt_sha256']):
            raise ValueError('Calibration labels/validity/prompt changed')
        records.append({**record, 'subject': row['subject']})
        features.append(observer.calibration_features({**row, 'answer': ''}, path, state[i]))
    expected = {(r['id'], p) for r in rows for p in spec.CALIBRATION_PATHS if p in trees[r['id']]}
    if seen != expected:
        raise ValueError('Incomplete decision-node observations')
    x = np.asarray(features, dtype=float)
    y = np.asarray([r['label'] for r in records], dtype=int)
    valid = np.asarray([r['valid'] for r in records], dtype=bool)
    folds = np.asarray([spec.calibration_fold(r['id']) for r in records])
    ids = np.asarray([r['id'] for r in records])
    oof = np.full((len(records), 4), np.nan)
    receipts = []
    for fold in range(3):
        train, held = valid & (folds != fold), folds == fold
        training = [r for r, chosen in zip(records, train) if chosen]
        fit_ids, held_ids = set(ids[train].tolist()), set(ids[held].tolist())
        if fit_ids & held_ids or not held_ids:
            raise ValueError('Invalid whole-question calibration split')
        fitted, weighting = fit_calibrator(x[train], y[train], training)
        oof[held] = predict_calibrator(fitted, x[held])
        receipts.append({'fold': fold, 'training_ids': sorted(fit_ids), 'held_ids': sorted(held_ids),
            'weights': weighting, 'question_overlap': 0, 'fit_feature_sha256': array_digest(x[train]),
            'fit_label_sha256': array_digest(y[train]), 'predictions_sha256': array_digest(oof[held])})
    if not np.isfinite(oof).all():
        raise ValueError('Missing excluded-question predictions')
    training = [r for r, chosen in zip(records, valid) if chosen]
    observer.calibration, weighting = fit_calibrator(x[valid], y[valid], training)
    eligible = {r['id'] for r in training}
    invalid_ids = sorted(set(by_id) - eligible)
    receipt = {'schema': spec.SCHEMA, 'fit_module': __name__, 'C': 1., 'max_iter': 1000,
        'seed': spec.SEED, 'mask': list(spec.CALIBRATION_PATHS), 'fit_valid_only': True,
        'terminal_rows_fit': 0, 'weights': weighting, 'folds': receipts,
        'all_invalid_questions': len(invalid_ids), 'all_invalid_ids_sha256': fingerprint(invalid_ids),
        'all_invalid_subject_counts': dict(Counter(by_id[q]['subject'] for q in invalid_ids)),
        'features_sha256': array_digest(x), 'labels_sha256': array_digest(y),
        'metadata_sha256': fingerprint(records), 'oof_sha256': array_digest(oof),
        'neural_training_question_exclusion': 'Must be bound by the variant input identity; not inferred from calibration folds',
        'calibration_question_exclusion': 'Three whole-question folds; weights use fit-fold rows only',
        'nested_neural_training_claim': False, 'development_or_test_used': False,
        'runtime_calibrator': 'All eligible TRAIN questions; weight mass equals independent eligible questions',
        'terminal_forecasts_used_for_emissions': False}
    return observer, {(r['id'], r['path']): p for r, p in zip(records, oof)}, receipt, records, oof


def plain_regular(rows, trees, observer, oof, encoder_price, calibration_receipt):
    """Use exactly seven plain paths; every table uses question-sized counts.

At stage1 a question has two possible incoming parents. Each receives1/2
weight per action so coverage never increases a question's total mass.
Terminal observation likelihoods are fixed unused support; their labels
contribute only to final-transition and cost estimates.
"""
    dynamics, cohorts = {}, {}
    hard = {r['id']: int(np.mean([correct(trees[r['id']][p], r['answer']) for p in core.FIXED]) < .6)
            for r in rows}
    for subject in sorted({r['subject'] for r in rows}):
        group = [r for r in rows if r['subject'] == subject]
        prior = np.ones(4); initial = np.full((4, core.N_OBS), .5)
        transition = np.zeros((2, 2, 4, 4))
        for h in (0, 1):
            transition[:, :, 2*h:2*h+2, 2*h:2*h+2] = 1.
        emission = np.full((2, 2, 4, core.N_OBS), .5)
        price_sum = np.zeros((2, 2, 4)); price_n = np.zeros_like(price_sum)
        cohort_count = np.ones((2, 4)); valid_count = np.ones((2, 4))
        for row in group:
            qid, tree = row['id'], trees[row['id']]
            if not set(core.PLAIN) <= set(tree):
                raise ValueError('Missing one of the seven plain controls')
            state = lambda p: 2 * hard[qid] + int(correct(tree[p], row['answer']))
            observed = lambda p: core.observation(oof[qid, p], bool(tree[p]['answer']))
            prior[state('')] += 1
            initial[state(''), observed('')] += 1
            for stage, parents in enumerate((('',), ('repair', 'retry'))):
                weight = 1. / len(parents)
                for parent in parents:
                    s = state(parent)
                    cohort_count[stage, s] += weight
                    valid_count[stage, s] += weight * bool(tree[parent]['answer'])
                    for action, edge in enumerate(('repair', 'retry')):
                        child = edge if not parent else parent + '/' + edge
                        node, nxt = tree[child], state(child)
                        transition[stage, action, s, nxt] += weight
                        if stage == 0:
                            emission[stage, action, nxt, observed(child)] += weight
                        price_sum[stage, action, s] += weight * (
                            (node['input_tokens'] + node['output_tokens']) * 4 / 3.09
                            + (encoder_price if stage == 0 else 0.))
                        price_n[stage, action, s] += weight
        transition = normalize(transition)
        initial[1::2, 0] = 0
        emission[:, :, 1::2, 0] = 0
        initial, emission = normalize(initial), normalize(emission)
        costs = np.divide(price_sum, price_n, out=np.full_like(price_sum, 550.), where=price_n > 0)
        # Review fields in this unqueried helper are deliberately uninformative.
        # The queried mode below has its own jointly fitted retained-review tables.
        review = np.full((2, core.N_OBS, 4, mentor.N_REVIEW), 1. / mentor.N_REVIEW)
        reviewed = np.repeat(transition[:, None], mentor.N_REVIEW, axis=1)
        comparison = np.full((core.N_OBS, 4, 3), 1. / 3)
        dynamics[subject] = core.Dynamics(core.normalize(prior), initial, transition, emission, costs,
            review, reviewed, emission.copy(), costs.copy(), np.zeros((2, 4)), comparison)
        for stage in (0, 1):
            validity = np.clip(valid_count[stage] / cohort_count[stage], 1e-5, 1-1e-5)
            validity[[1, 3]] = 1.
            cohorts[subject, stage] = (valid_count[stage] / valid_count[stage].sum(), validity)
    receipt = {'schema': spec.SCHEMA, 'training_questions': len(rows),
        'plain_paths': list(core.PLAIN), 'guided_in_unreviewed_counts': False,
        'stage1_question_parent_mass': .5, 'initial_and_first_emissions_question_excluded': True,
        'terminal_emissions': 'Fixed unused support; no terminal forecast or observer price',
        'calibration': calibration_receipt, 'review_model': 'Unused; queried mode fitted separately'}
    return perception.Planner(observer, dynamics, receipt, review_observer=None,
                              context_weight=0., observation_cohorts=cohorts)


def masks():
    support = np.zeros((8, 8))
    for s in range(8):
        for c in (0, 1):
            support[s, mentor.state(2 * (s // 4) + c, s % 2)] = 1.
    review_mask = np.ones((8, mentor.N_REVIEW))
    obs_mask = np.ones((8, mentor.N_OBS * 3))
    for s in range(8):
        c, m = (s // 2) % 2, s % 2
        for signal in range(30):
            agree = signal % 2
            if (agree and c != m) or (not agree and c == m == 1):
                review_mask[s, signal] = 1e-4
        if m:
            review_mask[s, 31] = 0.
        for obs in range(mentor.N_OBS):
            for equal in range(3):
                if ((c and obs == 0) or (obs == 0 and equal != 0)
                    or (equal == 2 and c != m) or (c == m == 1 and equal != 2)):
                    obs_mask[s, obs * 3 + equal] = 1e-4
    return support, review_mask, obs_mask


def paired_records(rows, trees, judges, oof, encoder_price, review_signals, review_encoder):
    records = []
    for row in rows:
        qid, tree = row['id'], trees[row['id']]
        if (qid, '') not in judges:
            continue
        judge = judges[qid, '']
        truth = int(correct(judge, row['answer']))
        hard = int(np.mean([correct(tree[p], row['answer']) for p in core.FIXED]) < .6)
        state = lambda p: mentor.state(2*hard + int(correct(tree[p], row['answer'])), truth)
        first, ends = [], []
        for parent in ('guided', 'retry'):
            node = tree[parent]
            obs = core.observation(oof[qid, parent], bool(node['answer']))
            first.append((state(parent), obs*3 + mentor.comparison(judge, node),
                (node['input_tokens'] + node['output_tokens'])*4/3.09 + encoder_price))
            last = []
            for child in (parent + '/reuse', parent + '/retry'):
                node = tree[child]
                last.append((state(child), (node['input_tokens'] + node['output_tokens'])*4/3.09))
            ends.append(last)
        cost = (judge['input_tokens'] + judge['output_tokens'])*3/3.09
        if review_encoder:
            cost += encoder_price
        records.append({'id': qid, 'subject': row['subject'], 'state': state(''),
            'signal': review_signals[qid], 'first': first, 'ends': ends, 'review_cost': cost})
    return records


def paired_counts(records, pooled=False):
    """Literal4:1 pooled subject weights; local evidence always unit/question."""
    support, review_mask, obs_mask = masks()
    k = np.ones((4, 2)); review = .5 * review_mask.copy()
    first = np.zeros((mentor.N_REVIEW, 2, 8, 8))
    last = np.broadcast_to(.5 * support, (2, 2, 8, 8)).copy()
    emission = np.broadcast_to(.1 * obs_mask, (2, 8, mentor.N_OBS*3)).copy()
    first_sum = np.zeros((2, 8)); first_n = np.zeros_like(first_sum)
    last_sum = np.zeros((2, 2, 8)); last_n = np.zeros_like(last_sum)
    review_sum = np.zeros(8); review_n = np.zeros(8)
    for row in records:
        weight = spec.POOLED[row['subject']] if pooled else 1.
        s, signal = row['state'], row['signal']
        if not 0 <= signal < mentor.N_REVIEW:
            raise ValueError('Invalid root review signal')
        k[s//2, s%2] += weight
        review[s, signal] += weight
        review_sum[s] += weight*row['review_cost']; review_n[s] += weight
        for action, (nxt, obs, price) in enumerate(row['first']):
            if not support[s, nxt]:
                raise ValueError('First action changed persistent hidden truth')
            first[signal, action, s, nxt] += weight
            emission[action, nxt, obs] += weight
            first_sum[action, s] += weight*price; first_n[action, s] += weight
            for a, (final, price) in enumerate(row['ends'][action]):
                if not support[nxt, final]:
                    raise ValueError('Last action changed persistent hidden truth')
                last[action, a, nxt, final] += weight
                last_sum[action, a, nxt] += weight*price; last_n[action, a, nxt] += weight
    return k, review, first, last, emission, first_sum, first_n, last_sum, last_n, review_sum, review_n


def reviewed_dynamics(regular, records, smoothing):
    support = masks()[0]
    pooled = paired_counts(records, True)
    dynamics = {}
    for subject, rd in regular.dynamics.items():
        local = paired_counts([r for r in records if r['subject'] == subject], False)
        relationship = normalize(local[0] + smoothing*normalize(pooled[0]))
        prior = core.normalize((rd.prior[:, None] * relationship).reshape(8))
        initial = np.repeat(rd.initial, 2, axis=0)
        plain = np.zeros((2, 2, 8, 8))
        for s in range(8):
            for target in range(4):
                plain[:, :, s, mentor.state(target, s % 2)] = rd.transition[:, :, s//2, target]
        review = normalize(local[1] + smoothing*normalize(pooled[1]))
        first = normalize(local[2] + smoothing*normalize(pooled[2].sum(0) + support[None])[None])
        last = normalize(local[3] + smoothing*normalize(pooled[3]))
        emission = normalize(local[4] + smoothing*normalize(pooled[4]))
        def price(offset, default):
            mean = np.divide(pooled[offset], pooled[offset+1],
                out=np.full_like(pooled[offset], default), where=pooled[offset+1] > 0)
            return (local[offset] + 4.*mean) / (local[offset+1] + 4.)
        dynamics[subject] = mentor.JointDynamics(prior, initial, plain, review, first, emission, last,
            price(5, 650.), price(7, 650.), price(9, 2800.),
            emission.reshape(2, 8, mentor.N_OBS, 3).sum(2))
    return dynamics


def fit_mentor(rows, trees, judges, regular, oof, metadata, outputs, encoder_price,
               smoothing=8., semantic_review=False, review_signals=None):
    spec.validate_counts(rows, {qid for qid, path in judges if path == ''})
    if any(path for qid, path in judges):
        raise ValueError('Expanded model permits one root logical review only')
    if smoothing not in spec.SMOOTHING:
        raise ValueError('Unregistered smoothing')
    if review_signals is not None and semantic_review:
        raise ValueError('Explicit metadata signals must not buy a review neural pass')
    if review_signals is None:
        lookup = {(r['id'], r['kind'], r['path']): v for r, v in zip(metadata, outputs['state'])}
        review_signals = {}
        for qid, path in judges:
            p = (perception.softmax(lookup[qid, 'original-review', ''], regular.classifier.temperature)
                 if semantic_review else None)
            review_signals[qid] = mentor.signal(judges[qid, ''], trees[qid][''], p)
        observation_name = 'semantic-review' if semantic_review else 'reported-review'
    else:
        observation_name = 'nested-question-excluded-paid-metadata'
    if set(review_signals) != {qid for qid, path in judges}:
        raise ValueError('Root signals do not cover exactly320 paid pairs')
    records = paired_records(rows, trees, judges, oof, encoder_price, review_signals, semantic_review)
    receipt = {'schema': spec.SCHEMA, 'fit_module': __name__, 'training_questions': 512,
        'paired_root_questions': 320, 'paired_reuse_outcomes': 640,
        'state_count': 8, 'maximum_logical_reviews': 1, 'semantic_review': semantic_review,
        'review_observation': observation_name, 'pooled_subject_weights': spec.POOLED,
        'subject_local_question_weight': 1., 'smoothing': smoothing,
        'calibration': regular.fit_receipt['calibration'], 'plain_counts': regular.fit_receipt,
        'first_emission_question_excluded': True, 'terminal_emission_or_neural_cost': False,
        'pair_records_sha256': fingerprint(records), 'solver_budget': 'At most two additional compact calls',
        'pure_more_data_ablation': False, 'neural_training': False, 'development_or_test_used': False}
    model = mentor.Planner(regular.classifier, reviewed_dynamics(regular, records, smoothing), receipt,
        regular, review_observer=regular.classifier if semantic_review else None)
    return model, records


def fit_transfer(rows, trees, judges, smoothing=8.):
    spec.validate_counts(rows, {qid for qid, path in judges if path == ''})
    support = np.asarray([[float(hybrid.action_core.CURRENT[s] == hybrid.mentor_core.REWARD[t])
                           for t in range(8)] for s in range(8)])
    records = []
    for row in rows:
        qid, tree = row['id'], trees[row['id']]
        if (qid, '') not in judges:
            continue
        current = int(correct(tree[''], row['answer']))
        source = hybrid.action_core.state(current, correct(tree['repair'], row['answer']),
                                          correct(tree['retry'], row['answer']))
        hard = int(np.mean([correct(tree[p], row['answer']) for p in core.FIXED]) < .6)
        target = mentor.state(2*hard + current, correct(judges[qid, ''], row['answer']))
        records.append((row['subject'], source, target))
    pooled = .1*support.copy()
    for subject, source, target in records:
        pooled[source, target] += spec.POOLED[subject]
    pooled = normalize(pooled)
    result = {}
    for subject in spec.SUBJECTS:
        local = .1*support.copy()
        for sub, source, target in records:
            if subject == sub:
                local[source, target] += 1.
        result[subject] = normalize(local + smoothing*pooled)
        np.testing.assert_allclose(result[subject] @ hybrid.mentor_core.REWARD,
                                   hybrid.action_core.CURRENT, atol=1e-12)
    return result


def build(base, fitted, rows, trees, judges, smoothing):
    recipe_a = hashlib.sha256((Path(base.classifier.directory)/'recipe.json').read_bytes()).hexdigest()
    recipe_m = hashlib.sha256((Path(fitted.classifier.directory)/'recipe.json').read_bytes()).hexdigest()
    if recipe_a != recipe_m:
        raise ValueError('Initial construction requires identical old-V1 assets; strongest-A replacement is separate')
    return hybrid.Planner(base, fitted, fit_transfer(rows, trees, judges, smoothing),
        {'schema': spec.SCHEMA, 'fit_module': __name__, 'state_count': 16, 'states_per_observed_mode': 8,
         'observed_modes': ['unqueried_action_fixability', 'queried_retained_mentor'],
         'base_A': base.fit_receipt, 'mentor': fitted.fit_receipt, 'maximum_reviews': 1,
         'transfer_smoothing': smoothing, 'pooled_subject_weights': spec.POOLED,
         'subject_local_question_weight': 1., 'observer_recipe_sha256': recipe_a,
         'solver_budget': 'At most two additional compact calls',
         'shared_neural_passes_require_identical_assets': True})
