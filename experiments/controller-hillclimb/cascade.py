"""Evaluate a compact-first controller on a separate, extended response cache."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from collect import DATA, canonical_answer, prompt as short_prompt, trees
from compact import MODELS, parse_answer, prompt
from hillclimb import FOLDS, folds, new_run, paired, save, sha, signals_from
from observations import Attempt, Context, basic, observed, trace
from planner import FIXED, SOLVER_PARAMETERS_B, Data, Ridge, outcomes, prepare, utility
from scoring import correct


COMPACT = 'qwen3.5:4b'
PATHS = ('compact_stop', 'compact_short_retry', 'compact_retry', 'compact_retry_repair', 'compact_retry_retry')
SHORT_PATHS = ('short_stop', 'short_compact_retry', 'short_retry', 'short_retry_repair', 'short_retry_retry')


@dataclass
class Cache:
    data: Data
    initial: np.ndarray
    followup: np.ndarray
    valid: np.ndarray
    correct: np.ndarray
    original: np.ndarray
    compact: np.ndarray
    model_digest: str

    def take(self, mask):
        return Cache(self.data.take(mask), self.initial[mask], self.followup[mask], self.valid[mask],
                     self.correct[mask], self.original[mask], self.compact[mask], self.model_digest)


def short_features(record, domain):
    evidence = record['evidence']
    if domain == 'math':
        logprob = evidence['mean_logprob']
        return [math.exp(logprob) if logprob is not None else 0.0, 0.0, 0.0]
    ordered = sorted(evidence.values(), reverse=True)
    best, runner_up = (ordered + [0.0, 0.0])[:2]
    return [best, best - runner_up, sum(value * value for value in ordered)]


def read(source, split, short_first=False):
    rows = trees(split)
    file = source / f'compact-4b-{split}.jsonl'
    with file.open(encoding='utf-8') as stream:
        outputs = list(map(json.loads, stream))
    records = {record['id']: record for record in outputs}
    direct = signals_from(source, split, ['4b'])[COMPACT]
    ids = {row['id'] for row in rows}
    digests = {record['model_digest'] for record in outputs + list(direct.values())}
    if len(outputs) != len(records) or set(records) != ids or set(direct) != ids or len(digests) != 1:
        raise ValueError(f'Wrong or incomplete compact responses for {split}')
    data = prepare(rows, {})
    initial, followup, valid, scores, original, compact = [], [], [], [], [], []
    for row in rows:
        first = records[row['id']]
        short = direct[row['id']]
        expected = hashlib.sha256(prompt(row['domain'], row['question']).encode()).hexdigest()
        if (first['model'] != COMPACT or first['prompt_sha256'] != expected
                or first['answer'] != parse_answer(row['domain'], first['text'])
                or first['input_tokens'] <= 0 or not 0 <= first['output_tokens'] <= 128):
            raise ValueError(f'Invalid compact response for {row["id"]}')
        formatted, limit = short_prompt(row['domain'], row['question'])
        if (short['prompt_sha256'] != hashlib.sha256(formatted.encode()).hexdigest()
                or short['input_tokens'] <= 0 or not 0 <= short['output_tokens'] <= limit):
            raise ValueError(f'Invalid short retry for {row["id"]}')
        extra = []
        truncated = first['finish'] == 'length'
        if short_first:
            first, short = short, first
            extra = short_features(first, row['domain'])
            truncated = row['domain'] == 'math' and first['finish'] == 'length'
        root = Attempt(canonical_answer(first['answer']), 'missing', bool(first['answer']),
                       truncated, first['text'], first['output_tokens'])
        context = Context(row['domain'], row['subject'], row['question'], (root,), '')
        retry = observed(row, '').history[0]
        later = Context(row['domain'], row['subject'], row['question'], (root, retry), 'retry')
        initial.append(basic(context) + trace(context) + extra)
        followup.append(basic(later) + trace(later) + extra)
        valid.append(root.valid)
        idx = len(valid) - 1
        scores.append([correct(first, row['answer']), correct(short, row['answer']), *data.correct[idx, :3]])
        old = data.tokens[idx]
        original.append([(0, 0), (0, 0), old[0], old[0] + old[1], old[0] + old[2]])
        consumed = np.array([first['input_tokens'], first['output_tokens']])
        short_tokens = np.array([short['input_tokens'], short['output_tokens']])
        compact.append([consumed, consumed + short_tokens, consumed, consumed, consumed])
    return Cache(data, np.array(initial), np.array(followup), np.array(valid), np.array(scores, dtype=int),
                 np.array(original), np.array(compact), digests.pop())


def work(cache):
    return cache.original.sum(axis=2) + cache.compact.sum(axis=2) * MODELS[COMPACT] / SOLVER_PARAMETERS_B


def fit(cache, strength):
    costs = work(cache)
    gains = cache.correct
    initial = np.column_stack([gains[:, 1:] - gains[:, [0]], (costs[:, 1:] - costs[:, [0]]) / 1000])
    followup = np.column_stack([gains[:, 3:] - gains[:, [2]], (costs[:, 3:] - costs[:, [2]]) / 1000])
    return Ridge.fit(cache.initial, initial, strength), Ridge.fit(cache.followup, followup, strength)


def predict(models, cache):
    return models[0].predict(cache.initial), models[1].predict(cache.followup)


def decide(cache, predictions, penalty, gate):
    penalties = np.full(len(cache.valid), penalty)
    values = utility(predictions[0], penalties)
    root_values = np.column_stack([values[:, 0], values[:, 1], values[:, 2:].max(axis=1)])
    if gate in ('invalid_only', 'always_rescue'):
        root_values[~cache.valid, 0] = -np.inf
    if gate == 'invalid_only':
        root_values[cache.valid, 1:] = -np.inf
    path = root_values.argmax(axis=1)
    aftermath = utility(predictions[1], penalties).argmax(axis=1)
    return np.where(path == 2, path + aftermath, path)


def record(cache, paths):
    index = (np.arange(len(paths)), paths)
    return {'correct': cache.correct[index], 'work': work(cache)[index],
            'original': cache.original[index], 'compact': cache.compact[index], 'paths': paths}


def profile(cache, paths, short_first=False):
    result = record(cache, paths)
    if short_first:
        accuracy_reference = cache.data.correct[:, [FIXED['repair_once'], FIXED['repair_twice']]].mean(axis=0).max()
        compute_reference = cache.data.tokens[:, 0].sum(axis=1).mean()
    else:
        accuracy_reference = cache.correct[:, 0].mean()
        compute_reference = work(cache)[:, 1:3].mean(axis=0).min()
    accuracy = result['correct'].mean()
    return {'accuracy': float(accuracy), 'work': float(result['work'].mean()),
            'accuracy_gain': float(accuracy - accuracy_reference),
            'work_savings': float(1 - result['work'].mean() / compute_reference)}


def select(train, dev, short_first=False):
    group = folds(train.data)
    best = None
    strengths = (8, 32, 128, 512) if short_first else (8, 32, 128)
    penalties = (0, .05, .1, .15, .2, .3, .45, .65)
    if short_first:
        penalties = (.04, .06, .08, .1, .12, .15, .18, .2, .23, .26, .3, .35, .4, .5, .65, .85, 1)
    for strength in strengths:
        trained = fit(train, strength)
        oof = [np.empty_like(array) for array in predict(trained, train)]
        for fold in range(FOLDS):
            split = group == fold
            model = fit(train.take(~split), strength)
            for target, values in zip(oof, predict(model, train.take(split))):
                target[split] = values
        dev_predictions = predict(trained, dev)
        for penalty, gate in product(penalties, ('invalid_only', 'always_rescue', 'learned')):
            t = profile(train, decide(train, oof, penalty, gate), short_first)
            d = profile(dev, decide(dev, dev_predictions, penalty, gate), short_first)
            gain = min(t['accuracy_gain'], d['accuracy_gain'])
            savings = min(t['work_savings'], d['work_savings'])
            rank = (gain > 0 and savings > 0, savings > 0, round(gain, 8),
                    round(t['accuracy_gain'] + d['accuracy_gain'], 8), savings)
            if best is None or rank > best[0]:
                best = (rank, {'strength': strength, 'penalty': penalty, 'gate': gate}, t, d)
    _, config, train_profile, dev_profile = best
    return {'config': config, 'train_oof': train_profile, 'dev': dev_profile}


def phase_evidence(source):
    here = Path(__file__).parent
    files = [here / path for path in ('cascade.py', 'compact.py', 'collect.py', 'hillclimb.py', 'observations.py', 'planner.py')]
    files += [DATA.parent / 'scoring.py', *(DATA / f'{split}-trees.jsonl.gz' for split in ('train', 'dev', 'test'))]
    files += [source / f'{prefix}-{split}.jsonl' for prefix, split in
              product(('compact-4b', 'critic-v2-4b'), ('train', 'dev'))]
    return {path.name: sha(path) for path in files}


def summarize(cache, methods, paths):
    groups = {}
    for name, result in methods.items():
        groups[name] = {}
        for domain in ('all', 'math', 'science'):
            subset = cache.data.domain == domain if domain != 'all' else np.ones(len(cache.valid), dtype=bool)
            old = result['original'][subset]
            compact = result['compact'][subset]
            groups[name][domain] = {
                'n': int(subset.sum()), 'correct': int(result['correct'][subset].sum()),
                'accuracy': float(result['correct'][subset].mean()),
                'mean_original_input_tokens': float(old[:, 0].mean()),
                'mean_original_output_tokens': float(old[:, 1].mean()),
                'mean_compact_input_tokens': float(compact[:, 0].mean()),
                'mean_compact_output_tokens': float(compact[:, 1].mean()),
                'mean_total_tokens': float((old + compact).sum(axis=1).mean()),
                'mean_solver_equivalent_tokens': float(result['work'][subset].mean()),
                'mean_parameter_token_flops_proxy': float(2e9 * SOLVER_PARAMETERS_B * result['work'][subset].mean()),
            }
            if name not in FIXED:
                groups[name][domain]['paths'] = dict(Counter(paths[index] for index in result['paths'][subset]))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['tune', 'test'])
    parser.add_argument('--signals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--short-first', action='store_true', help='Start with the expression or one-token format')
    args = parser.parse_args()
    train = read(args.signals, 'train', args.short_first)
    dev = read(args.signals, 'dev', args.short_first)
    if train.model_digest != dev.model_digest or set(train.data.ids) & set(dev.data.ids):
        raise ValueError('Split overlap or changed compact weights')
    frozen = args.output / 'frozen.json'
    if args.phase == 'tune':
        chosen = select(train, dev, args.short_first)
        cfg = chosen['config']
        with new_run(args.output) as pending:
            save(pending / 'frozen.json', {'selected': chosen, 'short_first': args.short_first, 'model_digest': train.model_digest,
                                           'evidence': phase_evidence(args.signals),
                                           'models': [model.as_dict() for model in fit(train, cfg['strength'])]})
        print(json.dumps(chosen, indent=2))
        return

    locked = json.loads(frozen.read_text(encoding='utf-8'))
    if locked['evidence'] != phase_evidence(args.signals) or locked['short_first'] != args.short_first:
        raise ValueError('Experiment artifacts changed since freeze')
    config = locked['selected']['config']
    model = fit(train, config['strength'])
    if not all(np.allclose(value, stored[key]) for fit_model, stored in zip(model, locked['models'])
               for key, value in fit_model.as_dict().items()):
        raise ValueError('Frozen fitted weights do not match')
    test = read(args.signals, 'test', args.short_first)
    if test.model_digest != locked['model_digest'] or set(test.data.ids) & (set(train.data.ids) | set(dev.data.ids)):
        raise ValueError('Split overlap or changed compact weights in test')
    saved = args.output / 'test'
    n = len(test.valid)
    paths = SHORT_PATHS if args.short_first else PATHS
    records = {name: record(test, np.full(n, path)) for path, name in enumerate(paths)}
    records.update({f'invalid_{name}': record(test, np.where(test.valid, 0, path))
                    for path, name in enumerate(paths) if path})
    for name, path in FIXED.items():
        original = outcomes(test.data, np.full(n, path))
        records[name] = {**original, 'original': original['solver'], 'compact': original['critic']}
    controller = decide(test, predict(model, test), config['penalty'], config['gate'])
    records['controller'] = record(test, controller)
    summary = summarize(test, records, paths)
    rng = np.random.default_rng(20261116)
    comparisons = {name: paired(records['controller'], result, rng)
                   for name, result in records.items() if name != 'controller'}
    with new_run(saved) as pending:
        save(pending / 'summary.json', {'methods': summary, 'paired': comparisons, 'frozen_sha256': sha(frozen),
                                       'test_signals_sha256': {name: sha(args.signals / f'{name}-test.jsonl')
                                                               for name in ('compact-4b', 'critic-v2-4b')}})
        with (pending / 'per_question.jsonl').open('x', encoding='utf-8') as stream:
            for i, key in enumerate(test.data.ids):
                stream.write(json.dumps({'id': key, 'domain': test.data.domain[i], 'path': paths[controller[i]],
                                         'outcomes': {name: {'correct': int(r['correct'][i]),
                                                             'original_tokens': r['original'][i].tolist(),
                                                             'compact_tokens': r['compact'][i].tolist()}
                                                      for name, r in records.items()}}) + '\n')
    for name, groups in summary.items():
        stats = groups['all']
        print(f'{name:<24} {stats["accuracy"]:>7.2%}  {stats["mean_solver_equivalent_tokens"]:8.1f} solver-equivalent tokens')


if __name__ == '__main__':
    main()
