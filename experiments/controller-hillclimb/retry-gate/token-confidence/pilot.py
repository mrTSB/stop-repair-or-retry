"""Compare gates with and without compact-answer probabilities on matched dev data."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from probe import sample
from evaluate import fresh, jsonl, published, validate_response
from policy import Learner, baselines, decide, fold_ids, measure, prepare, profile, selection


def load(rows, logs):
    records = jsonl(logs if logs.exists() else logs.with_suffix('.jsonl.gz'))
    by_id = {entry['id']: entry for entry in records}
    if len(by_id) != len(records) or set(by_id) != {row['id'] for row in rows}:
        raise ValueError('Missing or duplicate confidence records')
    packed, features = [], []
    for row in rows:
        entry = by_id[row['id']]
        validate_response(row, entry['compact'], row['short'])
        packed.append({**row, 'compact': entry['compact']})
        probability = list(entry['confidence'])
        if not entry['compact']['answer']:
            probability[:4] = [0, 0, 0, 0]
        features.append(probability)
    return prepare(packed), np.array(features, dtype=float), sum(bool(entry['matches_archive']) for entry in records)


def augment(batch, features):
    # Log odds retain differences between answer tokens that all round to 0.99.
    probability = np.clip(features[:, [1, 2, 4, 5, 7, 8, 10, 11]], 1e-7, 1 - 1e-7)
    odds = np.log(probability / (1 - probability)) / 4
    odds[features[:, 0] == 0, :2] = 0
    batch.compact = np.column_stack((batch.compact, features, odds))


def limits(data, confidence, quantile):
    output = {}
    for domain in ('math', 'science'):
        valid = (data.domain == domain) & (confidence[:, 0] > 0)
        output[domain] = float(np.quantile(confidence[valid, 2], quantile)) if valid.any() else 0.0
    return output


def constrain(data, probability, predictions, cutoffs, invalid_bonus):
    """Spend retries primarily on unparseable or low-confidence initial answers."""
    changed = predictions.copy()
    valid = data.answers[:, 0] != ''
    changed[~valid, 2] += invalid_bonus
    for domain, cutoff in cutoffs.items():
        certain = (data.domain == domain) & valid & (probability[:, 0] > 0) & (probability[:, 2] > cutoff)
        changed[certain, 2] = -np.inf
    return changed


def guarded(train, train_probability, dev, dev_probability):
    group = fold_ids(train)
    refs = [baselines(data)['retry_on_invalid'] for data in (train, dev)]
    best = None
    for kind, strength, margin in product(('linear', 'context'), (16, 64, 160), (-.02, .04)):
        model = Learner.fit(train, 'invalid', kind, strength, margin, .3)
        oof = np.empty((len(train.ids), 7))
        for fold in range(5):
            use = group != fold
            pair = Learner.fit(train.take(use), 'invalid', kind, strength, margin, .3)
            oof[~use] = pair.predict(train.take(~use))
        dev_predictions = model.predict(dev)
        for quantile, bonus in product((.15, .30, .50), (0.0, .06, .12)):
            train_predictions = np.empty_like(oof)
            for fold in range(5):
                use = group != fold
                cutoffs = limits(train.take(use), train_probability[use], quantile)
                train_predictions[~use] = constrain(train.take(~use), train_probability[~use], oof[~use], cutoffs, bonus)
            cutoffs = limits(train, train_probability, quantile)
            test_predictions = constrain(dev, dev_probability, dev_predictions, cutoffs, bonus)
            for credit in (.65, .80, .92):
                settings = {'parsing': 'invalid', 'kind': kind, 'strength': strength, 'margin': margin,
                            'first_short': 0.0, 'front_price': .32, 'credit': credit, 'minimum': 0.0,
                            'quantile': quantile, 'cutoffs': cutoffs, 'invalid_bonus': bonus}
                t, d = [profile(measure(data, *decide(data, p, settings)), ref)
                        for data, p, ref in zip((train, dev), (train_predictions, test_predictions), refs)]
                gain = min(t['accuracy_gain'], d['accuracy_gain'])
                savings = min(t['tokens_saved'], d['tokens_saved'])
                rank = (gain > 0 and savings > 0, savings > 0, gain, t['accuracy_gain'] + d['accuracy_gain'], savings)
                if best is None or rank > best[0]:
                    best = (rank, {'config': settings, 'train_oof': t, 'dev': d, 'model': model.as_dict()})
    return best[1]


def options(result):
    return {name: {key: value for key, value in chosen.items() if key != 'model'}
            for name, chosen in result.items()}


def auc(score, correct, mask):
    positive = score[mask & correct]
    negative = score[mask & ~correct]
    delta = positive[:, None] - negative
    return float(((delta > 0) + .5 * (delta == 0)).mean()) if delta.size else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--signals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-per-domain', type=int, default=80)
    args = parser.parse_args()
    sources = [('train', sample(published('train') + fresh(args.data, 'train'), args.train_per_domain)),
               ('dev', published('dev'))]
    observations = [load(rows, args.signals / f'{split}.jsonl') for split, rows in sources]
    train, dev = [item[0] for item in observations]
    regular = selection(train, dev)
    audit = {}
    for (split, _), (batch, features, exact) in zip(sources, observations):
        audit[split] = {'n': len(batch.ids), 'exact_archive_matches': exact, 'answer_confidence_observed': int(features[:, 0].sum()),
                        'answer_confidence_auc': auc(features[:, 1], batch.correct[:, 0].astype(bool), features[:, 0] > 0)}
        augment(batch, features)
    augmented = selection(train, dev)
    restricted = guarded(train, observations[0][1], dev, observations[1][1])
    result = {'audit': audit, 'without_confidence': options(regular), 'with_confidence': options(augmented),
              'guarded_confidence': options({'compact_first': restricted})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
