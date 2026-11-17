"""Calibrate retry fixes and regressions, then allocate the retry budget."""

from dataclasses import dataclass
import hashlib
from itertools import product

import numpy as np

from features import CONFIDENCE_WIDTH, RETRY_WIDTH, SHORT_INPUT_OFFSETS
from planner import Ridge


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class Logistic:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(cls, x, y, strength):
        mean = x.mean(axis=0)
        scale = x.std(axis=0).clip(min=0.08)
        design = np.column_stack((np.ones(len(x)), ((x - mean) / scale).clip(-6, 6)))
        weights = np.zeros(design.shape[1])
        weights[0] = np.log((y.sum() + 0.5) / (len(y) - y.sum() + 0.5))
        if np.all(y == y[0]):
            return cls(mean, scale, weights)
        regularizer = np.full(len(weights), strength)
        regularizer[0] = 0.1
        identity = np.diag(regularizer)
        for _ in range(24):
            p = sigmoid(design @ weights)
            curvature = np.maximum(p * (1 - p), 1e-5)
            gradient = design.T @ (p - y) + regularizer * weights
            step = np.linalg.solve(design.T @ (curvature[:, None] * design) + identity, gradient)
            weights -= step.clip(-1.5, 1.5)
            if np.max(np.abs(step)) < 1e-5:
                break
        return cls(mean, scale, weights)

    def predict(self, x):
        normalized = ((x - self.mean) / self.scale).clip(-6, 6)
        return sigmoid(self.coefficients[0] + normalized @ self.coefficients[1:])

    def as_dict(self):
        return {k: v.tolist() for k, v in vars(self).items()}


def fold_ids(first, total=5):
    folds = np.zeros(len(first.ids), dtype=int)
    for domain in np.unique(first.domains):
        group = np.flatnonzero(first.domains == domain)
        ranked = sorted(group, key=lambda i: hashlib.sha256(('paired-uplift:' + first.ids[i]).encode()).digest())
        folds[ranked] = np.arange(len(ranked)) % total
    return folds


def basis(raw, kind, *, comparison=False):
    if kind == 'no_confidence':
        if comparison:
            return np.column_stack((raw[:, :-(CONFIDENCE_WIDTH + RETRY_WIDTH)], raw[:, -RETRY_WIDTH:]))
        return raw[:, :-CONFIDENCE_WIDTH]
    if kind == 'context':
        return np.column_stack((raw, raw[:, 4:] * raw[:, [0]], raw[:, 4:] * raw[:, [3]], np.maximum(raw[:, 4:] - 0.5, 0)))
    return raw


def fit_comparator(data, kind, strength):
    relevant = data.first.valid & data.short_valid & data.disagree & (data.compact_correct ^ data.short_correct)
    if not relevant.any():
        raise ValueError('No observed exclusive wins to train the answer comparator')
    x = basis(data.comparison, kind, comparison=True)
    return Logistic.fit(x[relevant], data.short_correct[relevant], strength)


def accept_retry(first_valid, other_valid, disagree, probability, margin):
    return other_valid & disagree & (~first_valid | (probability > margin))


@dataclass
class Gate:
    kind: str
    strength: float
    margin: float
    compare: Logistic
    correct: Logistic
    can_fix: Logistic
    accepts_fix: Logistic
    accepts_harm: Logistic
    direct_fix: Logistic
    direct_harm: Logistic
    cost: Ridge

    @classmethod
    def fit(cls, data, kind, strength, margin):
        states, outcomes = data.first, data.compact_correct
        x = basis(states.raw, kind)
        compare_x = basis(data.comparison, kind, comparison=True)
        # A query model sees every exercise. Its labels use a held-out comparator for each exercise.
        group = fold_ids(states)
        post = np.empty(len(outcomes))
        for fold in range(5):
            take = group != fold
            comparator = fit_comparator(data.take(take), kind, strength)
            post[~take] = comparator.predict(compare_x[~take])
        switched = accept_retry(states.valid, data.short_valid, data.disagree, post, margin)
        fixes = switched & ~outcomes & data.short_correct
        harms = switched & outcomes & ~data.short_correct
        wrong = ~outcomes
        return cls(kind, strength, margin, fit_comparator(data, kind, strength),
                   Logistic.fit(x[states.valid], outcomes[states.valid], strength),
                   Logistic.fit(x[wrong], data.short_correct[wrong], strength),
                   Logistic.fit(x[wrong], fixes[wrong], strength),
                   Logistic.fit(x[outcomes], harms[outcomes], strength),
                   Logistic.fit(x, fixes, strength), Logistic.fit(x, harms, strength),
                   Ridge.fit(x, (data.short_tokens - states.prompt_tokens) / 128, strength))

    def predict(self, states):
        x = basis(states.raw, self.kind)
        p_correct = self.correct.predict(x) * states.valid
        can_fix = (1 - p_correct) * self.can_fix.predict(x)
        conditional_fix = (1 - p_correct) * self.accepts_fix.predict(x)
        conditional_harm = p_correct * self.accepts_harm.predict(x)
        outputs = {'can_fix': can_fix, 'fix': conditional_fix, 'harm': conditional_harm,
                   'direct_fix': self.direct_fix.predict(x), 'direct_harm': self.direct_harm.predict(x) * states.valid,
                   'cost': (states.prompt_tokens + self.cost.predict(x) * 128).clip(min=1)}
        return outputs

    def comparison(self, features):
        return self.compare.predict(basis(features, self.kind, comparison=True))

    def as_dict(self):
        return {key: value.as_dict() if isinstance(value, (Ridge, Logistic)) else value for key, value in vars(self).items()}

    @classmethod
    def restore(cls, payload):
        values = dict(payload)
        for key, value in payload.items():
            if isinstance(value, dict):
                estimator = Ridge if key == 'cost' else Logistic
                values[key] = estimator(**{k: np.array(v) for k, v in value.items()})
        return cls(**values)


def dispatch(states, estimates, settings):
    """Reallocate a fraction of the baseline retry budget, protecting predicted invalid-answer fixes."""
    invalid = ~states.valid
    math = states.domains == 'math'
    prompt = states.prompt_tokens + np.where(math, SHORT_INPUT_OFFSETS['math'], SHORT_INPUT_OFFSETS['science'])
    max_cost = prompt + np.where(math, 32, 1)
    cost = np.where(math, np.clip(estimates['cost'], prompt, max_cost), max_cost)
    # The baseline spends at least its retry input tokens. Charge an upper bound
    # to the gate so a longer-than-expected math expression cannot exceed the budget.
    budget = int(settings['credit'] * prompt[invalid].sum())
    request = np.zeros(len(cost), dtype=bool)
    # Missing initial answers cannot be harmed by replacing them with a parsed retry.
    invalid_probability = estimates['can_fix']
    target = settings['reserve'] * invalid_probability[invalid].sum()
    recovered = 0.0
    for i in np.argsort(-invalid_probability / cost, kind='stable'):
        if recovered >= target:
            break
        if invalid[i] and max_cost[i] <= budget:
            request[i] = True
            budget -= max_cost[i]
            recovered += invalid_probability[i]

    fix = estimates['direct_fix'] if settings['target'] == 'direct' else estimates['fix']
    harm = estimates['direct_harm'] if settings['target'] == 'direct' else estimates['harm']
    probability = np.where(invalid, invalid_probability, fix)
    gain = probability - settings['harm_weight'] * harm
    gain += settings['opportunity'] * np.where(invalid, 0, np.maximum(estimates['can_fix'] - probability, 0))
    for i in np.argsort(-gain / cost, kind='stable'):
        if not request[i] and gain[i] > 0.005 and max_cost[i] <= budget:
            request[i] = True
            budget -= max_cost[i]
    return request


def outcome(data, query, compare, margin):
    accepted = query & accept_retry(data.first.valid, data.short_valid, data.disagree, compare, margin)
    final = np.where(accepted, data.short_correct, data.compact_correct)
    return {'correct': final.astype(int), 'tokens': data.first.compact_tokens + query * data.short_tokens,
            'query': query, 'accepted': accepted}


def baseline(data):
    return outcome(data, ~data.first.valid, np.zeros(len(data.first.ids)), 1)


def profile(data, result, control):
    return {'n': len(data.first.ids), 'accuracy': float(result['correct'].mean()),
            'accuracy_gain': float((result['correct'] - control['correct']).mean()),
            'tokens': float(result['tokens'].mean()), 'tokens_saved': float((control['tokens'] - result['tokens']).mean()),
            'queries': int(result['query'].sum()),
            'actual_fixes_queried': int((result['query'] & ~data.compact_correct & data.short_correct).sum()),
            'successful_fixes': int(((result['correct'] == 1) & ~data.compact_correct).sum()),
            'regressions': int(((result['correct'] == 0) & data.compact_correct).sum())}


def average_precision(scores, actual):
    sorted_indices = np.argsort(-scores, kind='stable')
    ordered = scores[sorted_indices]
    groups = np.r_[np.flatnonzero(np.diff(ordered)), len(scores) - 1]
    positives = actual[sorted_indices].cumsum()[groups]
    if not positives[-1]:
        return None
    return float(np.dot(np.diff(np.r_[0, positives]) / positives[-1], positives / (groups + 1)))


def fix_audit(data, estimates):
    actual = ~data.compact_correct & data.short_correct
    return {'events': int(actual.sum()), 'rate': float(actual.mean()),
            'gate_average_precision': average_precision(estimates['can_fix'], actual),
            'invalid_flag_average_precision': average_precision((~data.first.valid).astype(float), actual),
            'brier_score': float(np.mean((estimates['can_fix'] - actual) ** 2)),
            'initial_invalid': int((~data.first.valid).sum())}


def fit_and_select(train, dev):
    folds = fold_ids(train.first)
    controls = [baseline(batch) for batch in (train, dev)]
    best = None
    ablation = None
    designs = [('linear', 14.0), ('linear', 48.0), ('context', 32.0), ('no_confidence', 24.0)]
    for (kind, strength), margin in product(designs, (.44, .55, .66)):
        model = Gate.fit(train, kind, strength, margin)
        intended = model.predict(dev.first)
        oof = {name: np.empty(len(train.first.ids)) for name in intended}
        comparison = np.empty(len(train.first.ids))
        for fold in range(5):
            in_fold = folds == fold
            fitted = Gate.fit(train.take(~in_fold), kind, strength, margin)
            for name, estimate in fitted.predict(train.first.take(in_fold)).items():
                oof[name][in_fold] = estimate
            comparison[in_fold] = fitted.comparison(train.comparison[in_fold])
        dev_comparison = model.comparison(dev.comparison)
        gates = product((.76, .86, .94), (.68, .85), (1., 1.5), (0., .30), ('conditional', 'direct'))
        for credit, reserve, harm_weight, opportunity, target in gates:
            settings = {'credit': credit, 'reserve': reserve, 'harm_weight': harm_weight,
                        'opportunity': opportunity, 'target': target}
            t, d = [profile(data, outcome(data, dispatch(data.first, messages, settings), post, margin), reference)
                    for data, messages, post, reference in zip((train, dev), (oof, intended), (comparison, dev_comparison), controls)]
            gain = min(t['accuracy_gain'], d['accuracy_gain'])
            saving = min(t['tokens_saved'], d['tokens_saved'])
            rank = (gain > 0 and saving > 0, saving > 0, round(gain, 8),
                    round(t['accuracy_gain'] + d['accuracy_gain'], 8), saving)
            item = (rank, model, settings, t, d, oof, intended)
            if best is None or rank > best[0]:
                best = item
            if kind == 'no_confidence' and (ablation is None or rank > ablation[0]):
                ablation = item

    def serialize(selected):
        _, model, settings, training, development, oof, intended = selected
        return {'model': model.as_dict(), 'settings': settings,
                'train_oof': training, 'dev': development,
                'fix_audit': {'train_oof': fix_audit(train, oof), 'dev': fix_audit(dev, intended)}}

    return {'selected': serialize(best), 'without_token_confidence': serialize(ablation)}
