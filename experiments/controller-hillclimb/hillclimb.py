"""Tune and freeze cost-aware controllers, then replay on the published test trees."""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from collect import DATA, prompt, trees
from planner import FIXED, PATH_TOKENS, PLANS, SOLVER_PARAMETERS_B, Planner, decide, outcomes


PENALTIES = (0, .025, .05, .075, .1, .15, .2, .25, .3, .4, .55, .75, 1)
STRENGTHS = (8, 32, 128)
MARGINS = (.02, .05, .1, .2, .4, 2)
SCIENCE_WEIGHTS = (0.7, 1.0, 1.5)
MODELS = {'2b': 'qwen3.5:2b', '4b': 'qwen3.5:4b'}
FOLDS = 5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


@contextmanager
def new_run(path):
    if path.exists():
        raise ValueError(f'{path} already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{path.name}-', dir=path.parent) as directory:
        pending = Path(directory)
        yield pending
        pending.rename(path)


def signals_from(source, split, models):
    cache = {}
    for alias in models:
        file = source / f'critic-v2-{alias}-{split}.jsonl'
        with file.open(encoding='utf-8') as stream:
            records = list(map(json.loads, stream))
        if len({r['model_digest'] for r in records}) != 1 or any(r['model'] != MODELS[alias] for r in records):
            raise ValueError(f'Auxiliary models changed in {file}')
        by_id = {r['id']: r for r in records}
        if len(by_id) != len(records):
            raise ValueError(f'Duplicate auxiliary observations in {file}')
        cache[MODELS[alias]] = by_id
    return cache


def read_split(source, split, models):
    from planner import prepare

    rows = trees(split)
    signals = signals_from(source, split, models)
    ids = {row['id'] for row in rows}
    if len(rows) != len(ids) or any(set(by_id) != ids for by_id in signals.values()):
        raise ValueError(f'Duplicate or missing question observations in {split}')
    for by_id in signals.values():
        for row in rows:
            record = by_id[row['id']]
            formatted, limit = prompt(row['domain'], row['question'])
            if (record['prompt_sha256'] != hashlib.sha256(formatted.encode()).hexdigest()
                    or record['input_tokens'] <= 0 or not 0 <= record['output_tokens'] <= limit):
                raise ValueError(f'Invalid auxiliary accounting or prompt for {row["id"]}')
    return prepare(rows, signals)


def folds(data):
    result = np.zeros(len(data.ids), dtype=int)
    for domain in np.unique(data.domain):
        indices = np.flatnonzero(data.domain == domain)
        ordered = sorted(indices, key=lambda i: hashlib.sha256(('hillclimb:' + data.ids[i]).encode()).digest())
        result[ordered] = np.arange(len(ordered)) % FOLDS
    return result


def crossfit(train, dev, group, richness, strength, critic):
    full = Planner.fit(train, richness, strength, critic)
    predictions = [np.empty_like(x) for x in full.predict(train)]
    for fold in range(FOLDS):
        model = Planner.fit(train.take(group != fold), richness, strength, critic)
        for target, result in zip(predictions, model.predict(train.take(group == fold))):
            target[group == fold] = result
    return tuple(predictions), full.predict(dev)


def profile(data, records):
    fixed = [outcomes(data, np.full(len(data.ids), FIXED[name])) for name in ('repair_once', 'repair_twice')]
    work = records['work'].mean()
    accuracy = records['correct'].mean()
    return {
        'accuracy': float(accuracy), 'work': float(work),
        'accuracy_gain': float(accuracy - max(r['correct'].mean() for r in fixed)),
        'work_savings': float(1 - work / fixed[0]['work'].mean()),
        'savings_vs_repair_twice': float(1 - work / fixed[1]['work'].mean()),
        'critic_rate': float(np.count_nonzero(records['critic'].sum(axis=1)) / len(data.ids)),
    }


def rank(train, dev):
    gain = min(train['accuracy_gain'], dev['accuracy_gain'])
    savings = min(train['work_savings'], dev['work_savings'])
    twice_savings = min(train['savings_vs_repair_twice'], dev['savings_vs_repair_twice'])
    return (gain > 1e-9 and savings > 1e-9, gain > 1e-9 and twice_savings > 1e-9,
            savings > 1e-9, round(gain, 8),
            round(train['accuracy_gain'] + dev['accuracy_gain'], 8), savings)


def select(train, dev, models):
    group = folds(train)
    choices = {}
    for richness in ('confidence', 'trace'):
        for strength in STRENGTHS:
            base = crossfit(train, dev, group, richness, strength, '')
            for alias in ('', *models):
                critic = MODELS[alias] if alias else ''
                reviewed = crossfit(train, dev, group, richness, strength, critic) if critic else (None, None)
                options = [('A', 'none', 0)]
                if alias:
                    options = [('B', 'all', 2)] + [
                        ('C', domain, margin) for domain in ('all', 'math', 'science') for margin in MARGINS
                        if (domain, margin) != ('all', 2)
                    ]
                grids = product(options, PENALTIES, SCIENCE_WEIGHTS, (False, True) if critic else (False,))
                for (family, scope, margin), penalty, science_weight, accept in grids:
                    config = {'richness': richness, 'strength': strength, 'critic': critic,
                              'scope': scope, 'margin': margin, 'penalty': penalty,
                              'science_weight': science_weight, 'accept_agreement': accept}
                    stats = []
                    for data, baseline, extra in zip((train, dev), base, reviewed):
                        paths, queries = decide(data, config, baseline, extra)
                        stats.append(profile(data, outcomes(data, paths, queries, critic)))
                    train_stats, dev_stats = stats
                    candidate = (rank(train_stats, dev_stats), config, train_stats, dev_stats)
                    previous = choices.get(family)
                    if previous is None or candidate[0] > previous[0]:
                        choices[family] = candidate
    return {family: {'config': config, 'train_oof': train, 'dev': dev}
            for family, (_, config, train, dev) in choices.items()}


def model_for(train, config):
    args = (config['richness'], config['strength'])
    base = Planner.fit(train, *args)
    reviewed = Planner.fit(train, *args, config['critic']) if config['critic'] else None
    return base, reviewed


def replay(data, config, base, reviewed):
    paths, queries = decide(data, config, base.predict(data), reviewed.predict(data) if reviewed else None)
    return outcomes(data, paths, queries, config['critic'])


def oracle(data):
    paths = []
    for scores, costs in zip(data.correct, data.tokens.sum(axis=-1) @ PATH_TOKENS.T):
        valid = np.flatnonzero(scores)
        paths.append(valid[costs[valid].argmin()] if valid.size else 0)
    return outcomes(data, np.array(paths))


def describe(data, result):
    summary = {}
    for domain in ('all', 'math', 'science'):
        subset = data.domain == domain if domain != 'all' else np.ones(len(data.ids), dtype=bool)
        solver = result['solver'][subset]
        critic = result['critic'][subset]
        paths = result['paths'][subset]
        summary[domain] = {
            'n': int(subset.sum()), 'correct': int(result['correct'][subset].sum()),
            'accuracy': float(result['correct'][subset].mean()),
            'mean_solver_input_tokens': float(solver[:, 0].mean()),
            'mean_solver_output_tokens': float(solver[:, 1].mean()),
            'mean_critic_input_tokens': float(critic[:, 0].mean()),
            'mean_critic_output_tokens': float(critic[:, 1].mean()),
            'mean_total_tokens': float((solver + critic).sum(axis=1).mean()),
            'mean_solver_equivalent_tokens': float(result['work'][subset].mean()),
            'mean_parameter_token_flops_proxy': float(2e9 * SOLVER_PARAMETERS_B * result['work'][subset].mean()),
            'mean_solver_calls': float(PATH_TOKENS[paths].sum(axis=-1).mean()),
            'mean_critic_calls': float(np.count_nonzero(critic.sum(axis=1)) / subset.sum()),
            'submitted_paths': dict(Counter(PLANS[path] or 'initial' for path in paths)),
        }
    return summary


def evidence(source, models):
    here = Path(__file__).parent
    artifacts = [here / name for name in ('hillclimb.py', 'planner.py', 'observations.py', 'collect.py')]
    artifacts += [DATA.parent / 'scoring.py', *(DATA / f'{split}-trees.jsonl.gz' for split in ('train', 'dev', 'test'))]
    artifacts += [source / f'critic-v2-{alias}-{split}.jsonl' for alias in models for split in ('train', 'dev')]
    return {path.name: sha(path) for path in artifacts}


def paired(policy, reference, rng):
    difference = policy['correct'] - reference['correct']
    cost = policy['work'] - reference['work']
    sampling = rng.integers(len(difference), size=(5000, len(difference)))
    return {
        'accuracy_delta_pp': float(difference.mean() * 100),
        'accuracy_delta_pp_ci95': (np.quantile(difference[sampling].mean(axis=1), [.025, .975]) * 100).tolist(),
        'work_delta': float(cost.mean()),
        'work_delta_ci95': np.quantile(cost[sampling].mean(axis=1), [.025, .975]).tolist(),
        'work_savings_pct': float((1 - policy['work'].mean() / reference['work'].mean()) * 100),
        'paired_wins': int(np.count_nonzero(difference > 0)),
        'paired_losses': int(np.count_nonzero(difference < 0)),
    }


def archive(path, data, methods):
    summaries = {name: describe(data, result) for name, result in methods.items()}
    rng = np.random.default_rng(20261116)
    comparisons = {name: {fixed: paired(result, methods[fixed], rng) for fixed in ('repair_once', 'repair_twice')}
                   for name, result in methods.items() if name not in FIXED}
    save(path / 'summary.json', {'methods': summaries, 'paired': comparisons})
    with (path / 'per_question.jsonl').open('x', encoding='utf-8') as stream:
        for name, result in methods.items():
            for i, key in enumerate(data.ids):
                record = {'id': key, 'domain': data.domain[i], 'policy': name,
                          'path': PLANS[result['paths'][i]], 'correct': int(result['correct'][i]),
                          'solver_tokens': result['solver'][i].tolist(), 'critic_tokens': result['critic'][i].tolist(),
                          'solver_equivalent_tokens': float(result['work'][i])}
                stream.write(json.dumps(record) + '\n')
    for name, summary in summaries.items():
        aggregate = summary['all']
        print(f'{name:<20} {aggregate["accuracy"]:>7.2%}  {aggregate["mean_solver_equivalent_tokens"]:8.1f} solver-equivalent tokens')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['tune', 'test'])
    parser.add_argument('--signals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='*', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if args.phase == 'tune':
        train = read_split(args.signals, 'train', args.models)
        dev = read_split(args.signals, 'dev', args.models)
        if set(train.ids) & set(dev.ids):
            raise ValueError('Question IDs overlap between training and development')
        selected = select(train, dev, args.models)
        payload = {'models': args.models, 'evidence': evidence(args.signals, args.models), 'selected': selected}
        payload['fits'] = {}
        for name, item in selected.items():
            base, reviewed = model_for(train, item['config'])
            payload['fits'][name] = [base.as_dict(), reviewed.as_dict() if reviewed else None]
        payload['oracle'] = {split: describe(data, oracle(data)) for split, data in (('train', train), ('dev', dev))}
        with new_run(args.output) as pending:
            save(pending / 'frozen.json', payload)
        print(json.dumps(selected, indent=2))
    else:
        frozen_path = args.output / 'frozen.json'
        payload = json.loads(frozen_path.read_text(encoding='utf-8'))
        if payload['evidence'] != evidence(args.signals, payload['models']):
            raise ValueError('Training, development, or source artifacts changed after freeze')
        test_path = args.output / 'test'
        if test_path.exists():
            raise ValueError(f'{test_path} already exists')
        selected = payload['selected']
        used = {s['config']['critic'] for s in selected.values()}
        aliases = [alias for alias, model in MODELS.items() if model in used]
        train = read_split(args.signals, 'train', aliases)
        dev = read_split(args.signals, 'dev', aliases)
        for name, item in selected.items():
            base, reviewed = model_for(train, item['config'])
            fresh = [base.as_dict(), reviewed.as_dict() if reviewed else None]
            if not all(np.allclose(m[stage][key], saved[stage][key])
                       for m, saved in zip(fresh, payload['fits'][name]) if m is not None
                       for stage in ('root', 'followup') for key in m[stage]):
                raise ValueError(f'Fitted model changed after freeze for {name}')
        train_signals = signals_from(args.signals, 'train', aliases)
        test_signals = signals_from(args.signals, 'test', aliases)
        for model, signals in test_signals.items():
            if {r['model_digest'] for r in signals.values()} != {r['model_digest'] for r in train_signals[model].values()}:
                raise ValueError(f'Test auxiliary model changed for {model}')
        data = read_split(args.signals, 'test', aliases)
        if set(data.ids) & (set(train.ids) | set(dev.ids)):
            raise ValueError('Question IDs overlap across splits')
        results = {name: outcomes(data, np.full(len(data.ids), path)) for name, path in FIXED.items()}
        results.update({name: replay(data, item['config'], *model_for(train, item['config']))
                        for name, item in selected.items()})
        with new_run(test_path) as pending:
            archive(pending, data, results)
            save(pending / 'manifest.json', {
                'frozen_sha256': sha(frozen_path),
                'test_trees_sha256': sha(DATA / 'test-trees.jsonl.gz'),
                'test_signals_sha256': {alias: sha(args.signals / f'critic-v2-{alias}-test.jsonl') for alias in aliases},
            })


if __name__ == '__main__':
    main()
