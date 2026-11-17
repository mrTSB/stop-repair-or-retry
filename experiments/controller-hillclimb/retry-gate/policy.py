"""Learn which replayed solver call to buy and which observed answer to submit."""

from dataclasses import dataclass
import hashlib
from itertools import product

import numpy as np

from planner import Ridge
from scoring import canonical_answer
from signals import compact_features, question_features, recovery, short_features


def join(*blocks):
    return np.concatenate(blocks, axis=1)


@dataclass
class Batch:
    ids: tuple
    domain: np.ndarray
    questions: np.ndarray
    compact: np.ndarray
    short: np.ndarray
    comparison: np.ndarray
    answers: np.ndarray
    costs: np.ndarray
    input_tokens: np.ndarray
    correct: np.ndarray

    def take(self, mask):
        arrays = [value[mask] for value in list(vars(self).values())[1:]]
        return Batch(tuple(key for key, keep in zip(self.ids, mask) if keep), *arrays)


def prepare(rows):
    questions, compact, short, comparison, answers, costs, inputs, labels = ([] for _ in range(8))
    for row in rows:
        domain, question = row['domain'], row['question']
        initial, retry = row['compact'], row['short']
        candidate, support = recovery(domain, question, initial)
        given = [canonical_answer(r) if r else '' for r in (initial['answer'], candidate.answer, retry['answer'])]
        evidence = retry['evidence']
        questions.append(question_features(domain, row['subject'], question))
        compact.append(compact_features(domain, question, initial, candidate, support))
        short.append(short_features(domain, retry))
        comparison.append([
            support.get(given[2].upper(), 0.0), evidence.get(given[0].upper(), 0.0),
            evidence.get(given[1].upper(), 0.0),
            bool(candidate.answer and given[1] == given[2]),
        ])
        answers.append(given)
        inputs.append([x['input_tokens'] for x in (initial, retry)])
        costs.append([x['input_tokens'] + x['output_tokens'] for x in (initial, retry)])
        gold = canonical_answer(row['answer'])
        labels.append([int(bool(a) and a == gold) for a in given])
    return Batch(tuple(row['id'] for row in rows), np.array([row['domain'] for row in rows]),
                 np.array(questions, dtype=float), np.array(compact, dtype=float), np.array(short, dtype=float),
                 np.array(comparison, dtype=float), np.array(answers, dtype=str),
                 np.array(costs), np.array(inputs), np.array(labels, dtype=int))


def fold_ids(data, count=5):
    folds = np.zeros(len(data.ids), dtype=int)
    for domain in np.unique(data.domain):
        indices = np.flatnonzero(data.domain == domain)
        ordered = sorted(indices, key=lambda i: hashlib.sha256(data.ids[i].encode()).digest())
        folds[ordered] = np.arange(len(ordered)) % count
    return folds


def expand(features, kind):
    if kind == 'linear':
        return features
    # Subject-specific slopes and hinge features model different error regimes with regularization.
    by_domain = [features[:, 4:] * features[:, [index]] for index in range(4)]
    if kind == 'threshold':
        return join(features, *by_domain, *(features[:, 4:] > limit for limit in (.25, .85, 1.8, 4)))
    return join(features, *by_domain, np.maximum(features[:, 4:] - 0.5, 0.0))


def views(data, kind):
    question = expand(data.questions, kind)
    compact = expand(join(data.questions, data.compact), kind)
    short = expand(join(data.questions, data.short), kind)
    compared = expand(join(data.questions, data.compact, data.short, data.comparison), kind)
    return question, compact, short, compared


def keep_index(data, parsing):
    alternative = (data.answers[:, 1] != '')
    if parsing == 'invalid':
        alternative &= data.answers[:, 0] == ''
    if parsing == 'original':
        alternative[:] = False
    return alternative.astype(int)


def compared_index(data, keep, estimate, margin):
    rows = np.arange(len(keep))
    compact = data.answers[rows, keep]
    short = data.answers[:, 2]
    use_retry = (short != '') & (short != compact) & ((compact == '') | (estimate > margin))
    return np.where(use_retry, 2, keep)


def pair_fit(data, features, keep, strength):
    indices = np.arange(len(keep))
    different = ((data.answers[indices, keep] != '') & (data.answers[:, 2] != '')
                 & (data.answers[indices, keep] != data.answers[:, 2]))
    paired = data.correct[:, 2] - data.correct[indices, keep]
    if different.sum() < 2:
        different[:] = True
    return Ridge.fit(features[different], paired[different], strength)


@dataclass
class Learner:
    front: Ridge
    after_compact: Ridge
    after_short: Ridge
    compare: Ridge
    parsing: str
    kind: str
    margin: float

    @classmethod
    def fit(cls, data, parsing, kind, strength, margin, opportunity=0.0):
        question, compact, short, compared = views(data, kind)
        keep = keep_index(data, parsing)
        rows = np.arange(len(keep))
        correctness = data.correct[rows, keep]
        crossfit = np.empty(len(keep))
        groups = fold_ids(data)
        for group in range(5):
            fit = groups != group
            pair = pair_fit(data.take(fit), compared[fit], keep[fit], strength)
            crossfit[~fit] = pair.predict(compared[~fit])
        selected = compared_index(data, keep, crossfit, margin)
        final = data.correct[rows, selected]
        before = np.column_stack([data.correct[:, 2] - correctness, (data.costs[:, 1] - data.costs[:, 0]) / 128])
        missed_short = (data.correct[:, 2] == 1) & (final == 0) & (correctness == 0)
        missed_compact = (correctness == 1) & (final == 0) & (data.correct[:, 2] == 0)
        after_compact = np.column_stack([final - correctness + opportunity * missed_short,
                                         (data.costs[:, 1] - data.input_tokens[:, 0]) / 128])
        after_short = np.column_stack([final - data.correct[:, 2] + opportunity * missed_compact,
                                       (data.costs[:, 0] - data.input_tokens[:, 1]) / 128])
        return cls(Ridge.fit(question, before, strength), Ridge.fit(compact, after_compact, strength),
                   Ridge.fit(short, after_short, strength), pair_fit(data, compared, keep, strength), parsing, kind, margin)

    def predict(self, data):
        question, compact, short, compared = views(data, self.kind)
        return np.column_stack([self.front.predict(question), self.after_compact.predict(compact),
                                self.after_short.predict(short), self.compare.predict(compared)])

    def as_dict(self):
        return {key: value.as_dict() if isinstance(value, Ridge) else value for key, value in vars(self).items()}


def first_calls(predictions, fraction, cost_weight):
    gain, delta = predictions[:, 0], predictions[:, 1]
    value = gain - cost_weight * delta
    selected = np.zeros(len(value), dtype=bool)
    ranked = np.argsort(-value, kind='stable')[:int(fraction * len(value))]
    selected[ranked] = value[ranked] > 0
    return selected


def allocate(data, predictions, first_short, fraction, minimum):
    """Spend only estimated work credited by the validity rule and first-call savings."""
    compact_seen = ~first_short
    gain = np.where(first_short, predictions[:, 4], predictions[:, 2])
    cost = np.where(first_short, data.input_tokens[:, 1] + predictions[:, 5] * 128,
                    data.input_tokens[:, 0] + predictions[:, 3] * 128).clip(min=1)
    predicted_saved = (-predictions[first_short, 1] * 128).sum()
    invalid_cost = cost[compact_seen & (data.answers[:, 0] == '')].sum()
    budget = fraction * max(0, invalid_cost + predicted_saved)
    requested = np.zeros(len(cost), dtype=bool)
    for i in np.argsort(-gain / cost, kind='stable'):
        if gain[i] > minimum and cost[i] <= budget:
            requested[i] = True
            budget -= cost[i]
    return requested


def decide(data, predictions, settings):
    first_short = first_calls(predictions, settings['first_short'], settings['front_price'])
    retry = allocate(data, predictions, first_short, settings['credit'], settings['minimum'])
    keep = keep_index(data, settings['parsing'])
    compared = compared_index(data, keep, predictions[:, 6], settings['margin'])
    return np.where(retry, compared, np.where(first_short, 2, keep)), first_short, retry


def measure(data, selected, first_short, retry):
    row = np.arange(len(selected))
    first = first_short.astype(int)
    tokens = data.costs[row, first] + retry * data.costs[row, 1 - first]
    return {'correct': data.correct[row, selected], 'tokens': tokens, 'selected': selected,
            'first_short': first_short, 'retry': retry}


def baselines(data):
    no = np.zeros(len(data.ids), dtype=bool)
    invalid = data.answers[:, 0] == ''
    recovered = keep_index(data, 'invalid')
    unresolved = invalid & (recovered == 0)
    return {
        'compact_only': measure(data, no.astype(int), no, no),
        'retry_on_invalid': measure(data, np.where(invalid, 2, 0), no, invalid),
        'recover_then_retry_on_invalid': measure(data, np.where(unresolved, 2, recovered), no, unresolved),
    }


def profile(record, base):
    return {'accuracy': float(record['correct'].mean()), 'tokens': float(record['tokens'].mean()),
            'accuracy_gain': float((record['correct'] - base['correct']).mean()),
            'tokens_saved': float((base['tokens'] - record['tokens']).mean()),
            'retry_count': int(record['retry'].sum()), 'short_first_count': int(record['first_short'].sum())}


def selection(train, dev):
    groups = fold_ids(train)
    references = [baselines(data)['retry_on_invalid'] for data in (train, dev)]
    selected = {}
    for parsing, kind, strength, margin, opportunity in product(
            ('original', 'invalid', 'trace'), ('linear', 'context', 'threshold'), (12, 48, 160), (-0.02, 0.06), (0.0, .45)):
        model = Learner.fit(train, parsing, kind, strength, margin, opportunity)
        oof = np.empty((len(train.ids), 7))
        for fold in range(5):
            mask = groups == fold
            crossfit = Learner.fit(train.take(~mask), parsing, kind, strength, margin, opportunity)
            oof[mask] = crossfit.predict(train.take(mask))
        dev_predictions = model.predict(dev)
        front = [(0.0, .32)] + list(product((.12, .25, .45), (.12, .32, .60)))
        for (first_short, front_price), credit, minimum in product(front, (.60, .78, .90), (0.0, .035)):
            family = 'compact_first' if first_short == 0 else 'routed_first'
            settings = {'parsing': parsing, 'kind': kind, 'strength': strength, 'margin': margin,
                        'first_short': first_short, 'front_price': front_price, 'credit': credit, 'minimum': minimum,
                        'opportunity': opportunity}
            t, d = [profile(measure(data, *decide(data, expected, settings)), base)
                    for data, expected, base in zip((train, dev), (oof, dev_predictions), references)]
            gain, saved = min(t['accuracy_gain'], d['accuracy_gain']), min(t['tokens_saved'], d['tokens_saved'])
            rank = (gain > 0 and saved > 0, saved > 0, gain, t['accuracy_gain'] + d['accuracy_gain'], saved)
            incumbent = selected.get(family)
            if incumbent is None or rank > incumbent[0]:
                selected[family] = (rank, {'config': settings, 'train_oof': t, 'dev': d, 'model': model.as_dict()})
    return {name: item[1] for name, item in selected.items()}
