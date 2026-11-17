"""Fit a weighted probability update and optimize retry selection under a token cap."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from arithmetic import correction
from study_data import index_responses, prior_pairs, prior_test, probe_sample, question_rows, records, validate_response
from evaluate import check_disjoint
from features import SHORT_INPUT_OFFSETS, paired as pair_features, probability_features
from gate import Gate, accept_retry, average_precision, baseline, dispatch, fold_ids, outcome, profile, sigmoid
from scoring import correct


def load_fit(source):
    archive = prior_pairs()
    prior = prior_test()
    missing = [row for row in prior if row['short'] is None]
    extra = index_responses(missing, records(source, 'prior-retries'))
    for row in missing:
        row['short'] = extra[row['id']]['short']
        validate_response(row, row['compact'], row['short'])
    probes = probe_sample(archive)
    signals = index_responses([row for row, _ in probes], records(source, 'train-probes'))
    traces, weights = [], []
    for row, weight in probes:
        sample = signals[row['id']]
        fresh = sample['compact']
        validate_response(row, fresh, row['short'])
        if sample['weight'] != weight or any(fresh[k] != row['compact'][k]
                                             for k in ('text', 'input_tokens', 'output_tokens', 'finish', 'answer')):
            raise ValueError(f'Probe differs from the archived answer for {row["id"]}')
        traces.append({**row, 'compact': fresh})
        weights.append(weight)
    dev = question_rows(source, 'dev')
    replies = index_responses(dev, records(source, 'dev-pairs'))
    for row in dev:
        row.update(replies[row['id']])
        validate_response(row, row['compact'], row['short'])
    check_disjoint(archive, prior, dev, question_rows(source, 'test'))
    return archive, traces, np.array(weights), dev, prior


def margin_view(rows):
    result = []
    for row in rows:
        comp = row['compact']
        features = np.asarray(comp['margin']['features'], dtype=float)
        if features.shape != (17,) or not np.isfinite(features).all():
            raise ValueError('Invalid token-margin observation')
        math = row['domain'] == 'math'
        invalid = not comp['answer']
        result.append([math, invalid, *features, *(features[:5] * math), *(features[9:13] * invalid)])
    return np.array(result, dtype=float)


def margin_pair_view(rows, first=None):
    if first is None:
        first = margin_view(rows)
    extra = []
    for row in rows:
        comp, retry = row['compact'], row['short']
        options = comp['margin']['alternatives']
        previous, next_ = options.get(comp['answer'], 0.0), options.get(retry['answer'], 0.0)
        extra.append([previous, next_, next_ - previous, bool(next_), bool(previous and next_ > previous)])
    return np.column_stack((first, np.array(extra, dtype=float)))


def confidence_view(rows):
    result = []
    for row in rows:
        comp = row['compact']
        observed = np.array(probability_features(comp.get('confidence'), bool(comp['answer'])))
        math = row['domain'] == 'math'
        result.append([math, not comp['answer'], *observed, *(observed[[2, 3, 14, 15]] * math)])
    return np.array(result, dtype=float)


def pair_view(rows, first=None):
    if first is None:
        first = confidence_view(rows)
    extra = []
    for row in rows:
        initial, retry = row['compact']['answer'], row['short']['answer']
        scores = row['short']['evidence']
        a, b = scores.get(initial, 0), scores.get(retry, 0)
        extra.append([a, b, b - a, scores.get('mean_logprob') or 0, float(row['domain'] == 'math')])
    return np.column_stack((first, np.array(extra, dtype=float)))


def logit(probability):
    value = np.clip(probability, 1e-5, 1 - 1e-5)
    return np.log(value / (1 - value))


@dataclass
class ProbabilityUpdate:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    @classmethod
    def fit(cls, x, y, probability, weights, strength):
        if len(x) == 0:
            return cls(np.zeros(x.shape[1]), np.ones(x.shape[1]), np.zeros(x.shape[1] + 1))
        w = weights / weights.mean()
        mean = np.average(x, axis=0, weights=w)
        scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w)).clip(min=0.12)
        design = np.column_stack((np.ones(len(x)), ((x - mean) / scale).clip(-6, 6)))
        coef = np.zeros(design.shape[1])
        penalty = np.full(len(coef), strength)
        penalty[0] = 2
        ridge = np.diag(penalty)
        offset = logit(probability)
        for _ in range(25):
            p = sigmoid(offset + design @ coef)
            curvature = w * np.maximum(p * (1 - p), 1e-5)
            grad = design.T @ (w * (p - y)) + penalty * coef
            step = np.linalg.solve(design.T @ (curvature[:, None] * design) + ridge, grad)
            coef -= step.clip(-1.5, 1.5)
            if np.max(np.abs(step)) < 1e-5:
                break
        return cls(mean, scale, coef)

    def predict(self, x, probability, share):
        normalized = ((x - self.mean) / self.scale).clip(-6, 6)
        return sigmoid(logit(probability) + share * (self.coef[0] + normalized @ self.coef[1:]))

    def as_dict(self):
        return {k: value.tolist() for k, value in vars(self).items()}


@dataclass
class Learner:
    base: Gate
    updates: dict
    compare: ProbabilityUpdate

    def predict(self, states, signals, share):
        before = self.base.predict(states)
        if share:
            for name, fit in self.updates.items():
                before[name] = fit.predict(signals, before[name], share)
        before['harm'] *= states.valid
        return before

    def comparison(self, features, signals, share):
        p = self.base.comparison(features)
        return self.compare.predict(signals, p, share) if share else p

    def as_dict(self):
        return {'base': self.base.as_dict(), 'updates': {k: value.as_dict() for k, value in self.updates.items()},
                'compare': self.compare.as_dict()}

    @classmethod
    def restore(cls, record):
        def update(item):
            return ProbabilityUpdate(**{k: np.array(v) for k, v in item.items()})

        return cls(Gate.restore(record['base']), {k: update(value) for k, value in record['updates'].items()},
                   update(record['compare']))


def teacher_targets(x, actual, reference, weights, groups, eligible):
    """Produce soft labels from token margins with each labeled probe left out."""
    taught = np.array(reference, copy=True)
    for fold in range(5):
        use = (groups != fold) & eligible
        held = groups == fold
        teacher = ProbabilityUpdate.fit(x[use], actual[use], reference[use], weights[use], 14)
        taught[held] = teacher.predict(x[held], reference[held], .75)
    return .65 * actual + .35 * taught


def fit(data, probe, before, after, teacher_before, teacher_after, weights, strength, margin):
    base = Gate.fit(data, 'linear', strength, margin)
    folds = fold_ids(data.first)
    lookup = dict(zip(data.first.ids, folds))
    groups = np.array([lookup[key] for key in probe.first.ids])
    oof = {name: np.empty(len(groups)) for name in ('can_fix', 'fix', 'harm')}
    post = np.empty(len(groups))
    for fold in range(5):
        seen = groups == fold
        trained = Gate.fit(data.take(folds != fold), 'linear', strength, margin)
        for name, p in trained.predict(probe.first.take(seen)).items():
            if name in oof:
                oof[name][seen] = p
        post[seen] = trained.comparison(probe.comparison[seen])

    exclusive = probe.first.valid & probe.short_valid & probe.disagree & (probe.short_correct ^ probe.compact_correct)
    y = probe.short_correct
    pair_targets = teacher_targets(teacher_after, y, post, weights, groups, exclusive)
    comparator = ProbabilityUpdate.fit(after[exclusive], pair_targets[exclusive], post[exclusive], weights[exclusive], 14)
    calibrated = post.copy()
    for fold in range(5):
        seen = groups == fold
        proper = exclusive & ~seen
        fit_one = ProbabilityUpdate.fit(after[proper], pair_targets[proper], post[proper], weights[proper], 14)
        calibrated[seen] = fit_one.predict(after[seen], post[seen], .75)
    choose = accept_retry(probe.first.valid, probe.short_valid, probe.disagree, calibrated, margin)
    can_fix = ~probe.compact_correct & probe.short_correct
    targets = {'can_fix': can_fix, 'fix': choose & can_fix, 'harm': choose & probe.compact_correct & ~probe.short_correct}
    eligible = np.ones(len(groups), dtype=bool)
    updates = {
        name: ProbabilityUpdate.fit(before, teacher_targets(teacher_before, actual, oof[name], weights, groups, eligible),
                                    oof[name], weights, 14)
        for name, actual in targets.items()
    }
    return Learner(base, updates, comparator)


def maximize_value(value, upper_cost, budget):
    """Maximize expected net fixes over the whole batch, with integer upper costs."""
    active = np.flatnonzero((value > 0.003) & (upper_cost <= budget))
    choices = np.zeros(len(value), dtype=bool)
    if not active.size:
        return choices
    best = np.zeros(budget + 1)
    taken = np.zeros((len(active), budget + 1), dtype=bool)
    for j, index in enumerate(active):
        cost = upper_cost[index]
        with_item = best[:-cost] + value[index]
        better = with_item > best[cost:] + 1e-11
        taken[j, cost:] = better
        best[cost:][better] = with_item[better]
    remaining = int(np.argmax(best))
    for j in range(len(active) - 1, -1, -1):
        if taken[j, remaining]:
            index = active[j]
            choices[index] = True
            remaining -= upper_cost[index]
    return choices


def allocate(states, estimates, settings, verified=None):
    math = states.domains == 'math'
    invalid = ~states.valid
    prompt = states.prompt_tokens + np.where(math, SHORT_INPUT_OFFSETS['math'], SHORT_INPUT_OFFSETS['science'])
    upper = prompt + np.where(math, 32, 1)
    budget = int(prompt[invalid].sum() * settings['credit'])
    gain = estimates['fix'] - settings['harm_weight'] * estimates['harm']
    gain += .2 * np.maximum(estimates['can_fix'] - estimates['fix'], 0)
    gain[invalid] = settings['invalid_weight'] * estimates['can_fix'][invalid]
    if verified is not None:
        gain[verified] = 0
    return maximize_value(gain, upper, budget)


def evaluate_old(dev, previous):
    old = Gate.restore(previous['model'])
    query = dispatch(dev.first, old.predict(dev.first), previous['settings'])
    return outcome(dev, query, old.comparison(dev.comparison), old.margin)


def exact_results(rows):
    alternative = [correction(row) for row in rows]
    exists = np.array([bool(answer) for answer in alternative])
    correct_exact = np.array([correct({'answer': answer}, row['answer']) for answer, row in zip(alternative, rows)])
    return exists, correct_exact


def add_arithmetic(result, eligible, grades):
    return {**result, 'correct': np.where(eligible, grades, result['correct']).astype(int)}


def select(rows, probes, weights, development, earlier_development, previous):
    train = pair_features(rows)
    checked = pair_features(probes)
    dev = pair_features(development)
    first_probe = confidence_view(probes)
    first_dev = confidence_view(development)
    after_probe, after_dev = pair_view(probes, first_probe), pair_view(development, first_dev)
    teacher_probe = margin_view(probes)
    teacher_pairs = margin_pair_view(probes, teacher_probe)
    invalid = baseline(dev)
    earlier = pair_features(earlier_development)
    first_earlier = confidence_view(earlier_development)
    after_earlier = pair_view(earlier_development, first_earlier)
    earlier_baseline = baseline(earlier)
    verified, exact_grades = exact_results(development)
    earlier_verified, earlier_grades = exact_results(earlier_development)
    verified_control = add_arithmetic(invalid, verified, exact_grades)
    verified_earlier_control = add_arithmetic(earlier_baseline, earlier_verified, earlier_grades)
    old = evaluate_old(dev, previous)
    reference = profile(dev, old, invalid)
    baseline_retry_tokens = (dev.short_tokens * invalid['query']).sum()
    best = None
    records = []
    for strength, margin in product((24, 64), (.44, .55, .66)):
        model = fit(train, checked, first_probe, after_probe, teacher_probe, teacher_pairs, weights, strength, margin)
        for share in (0.0, .75):
            p = model.predict(dev.first, first_dev, share)
            post = model.comparison(dev.comparison, after_dev, share)
            earlier_p = model.predict(earlier.first, first_earlier, share)
            earlier_post = model.comparison(earlier.comparison, after_earlier, share)
            for credit, invalid_weight, harm_weight in product((.60, .72, .82, .91), (1., 1.55, 2.15), (1., 1.6)):
                settings = {'credit': credit, 'invalid_weight': invalid_weight, 'harm_weight': harm_weight, 'distilled_share': share}
                result = outcome(dev, allocate(dev.first, p, settings, verified), post, margin)
                result = add_arithmetic(result, verified, exact_grades)
                metrics = profile(dev, result, invalid)
                improvement_from_routing = profile(dev, result, verified_control)
                earlier_result = outcome(earlier, allocate(earlier.first, earlier_p, settings, earlier_verified), earlier_post, margin)
                earlier_result = add_arithmetic(earlier_result, earlier_verified, earlier_grades)
                earlier_metrics = profile(earlier, earlier_result, verified_earlier_control)
                accuracy_gain, saved = metrics['accuracy_gain'], metrics['tokens_saved']
                retry_savings = saved * len(dev.first.ids) / baseline_retry_tokens if baseline_retry_tokens else 0
                # Prefer a bigger gain and a larger saving than the locked prior gate.
                dominates_old = accuracy_gain > reference['accuracy_gain'] and saved > reference['tokens_saved']
                dual = accuracy_gain > 0 and saved > 0
                stable = earlier_metrics['accuracy_gain'] >= 0
                routing_gain = improvement_from_routing['accuracy_gain']
                rank = (dual and stable and routing_gain > 0 and retry_savings >= .18,
                        dual and stable and routing_gain > 0, dual and stable and dominates_old,
                        dual and stable and retry_savings >= .18, dual and stable, dual,
                        round(accuracy_gain, 8), saved, -share)
                metrics['retry_work_savings'] = float(retry_savings)
                metrics['fix_average_precision'] = average_precision(p['can_fix'], ~dev.compact_correct & dev.short_correct)
                metrics['accuracy_gain_vs_arithmetic_and_invalid'] = routing_gain
                records.append({'strength': strength, 'comparison_margin': margin, 'settings': settings,
                                'metrics': metrics, 'earlier_development_vs_arithmetic_and_invalid': earlier_metrics})
                if best is None or rank > best[0]:
                    best = (rank, model, settings, metrics, earlier_metrics)
    _, model, settings, metrics, earlier_metrics = best
    return {'model': model.as_dict(), 'settings': settings, 'dev': metrics, 'previous_gate_dev': reference,
            'earlier_development_vs_arithmetic_and_invalid': earlier_metrics,
            'arithmetic_and_invalid_dev': profile(dev, verified_control, invalid),
            'invalid_flag_average_precision': average_precision((~dev.first.valid).astype(float), ~dev.compact_correct & dev.short_correct),
            'candidates': records}
