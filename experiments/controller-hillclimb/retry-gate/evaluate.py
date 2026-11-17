"""Freeze gate settings and evaluate the original and newly sampled test suites."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from collect_pairs import MODEL, PARENT, read, stem
from collect import DATA, prompt as short_prompt
from compact import parse_answer, prompt as compact_prompt
from hillclimb import new_run, paired, save, sha
from planner import SOLVER_PARAMETERS_B, Ridge
from policy import Learner, baselines, decide, measure, prepare, selection


PUBLISHED = DATA.parent.parent / 'controller-hillclimb/data'
MODEL_DIGEST = '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd'
COMPUTE = 4 / SOLVER_PARAMETERS_B


def jsonl(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as source:
        return list(map(json.loads, source))


def validate_response(question, compact, short):
    domain, prompt = question['domain'], question['question']
    short_text, limit = short_prompt(domain, prompt)
    for record, formatted, cap in ((compact, compact_prompt(domain, prompt), 128), (short, short_text, limit)):
        digest = hashlib.sha256(formatted.encode()).hexdigest()
        if (record['model'] != MODEL or record['model_digest'] != MODEL_DIGEST or record['prompt_sha256'] != digest
                or record['input_tokens'] <= 0 or not 0 <= record['output_tokens'] <= cap):
            raise ValueError(f'Changed inference settings for {question["id"]}')
    if parse_answer(domain, compact['text']) != compact['answer']:
        raise ValueError(f'Changed compact parser for {question["id"]}')


def merge(questions, records):
    by_id = {record['id']: record for record in records}
    ids = {question['id'] for question in questions}
    if len(by_id) != len(records) or len(ids) != len(questions) or set(by_id) != ids:
        raise ValueError('Wrong or incomplete response coverage')
    result = []
    for row in questions:
        record = by_id[row['id']]
        compact, short = record['compact'], record['short']
        validate_response(row, compact, short)
        result.append({**row, 'compact': compact, 'short': short})
    return result


def published(split):
    left = jsonl(PUBLISHED / f'compact-4b-{split}.jsonl.gz')
    right = jsonl(PUBLISHED / f'critic-v2-4b-{split}.jsonl.gz')
    assert len(left) == len(right)
    other = {record['id']: record for record in right}
    if len(other) != len(right):
        raise ValueError('Duplicate short responses in published cache')
    responses = [{'id': record['id'], 'compact': record, 'short': other[record['id']]} for record in left]
    return merge(read(DATA / f'{split}-questions.jsonl'), responses)


def fresh(source, split):
    manifest = json.loads((source / 'splits/selection.json').read_bytes())
    questions = source / 'splits' / f'{split}-questions.jsonl'
    records = source / f'{split}-pairs.jsonl'
    if not records.exists():
        records = records.with_suffix('.jsonl.gz')
    selection = manifest['splits'][split]
    if sha(questions) != selection['sha256']:
        raise ValueError(f'Changed fresh {split} question split')
    rows = read(questions)
    if len(rows) != selection['count']:
        raise ValueError(f'Incomplete fresh {split} question split')
    return merge(rows, jsonl(records))


def check_disjoint(*splits):
    ids, texts = set(), set()
    for split in splits:
        split_ids = {row['id'] for row in split}
        split_text = {stem(row['question']) for row in split}
        if ids & split_ids or texts & split_text or len(split_ids) != len(split):
            raise ValueError('Overlap between controller splits')
        ids.update(split_ids)
        texts.update(split_text)


def inputs(source):
    here = Path(__file__).resolve().parent
    groups = {
        'source': [*here.glob('*.py'), *(PARENT / name for name in ('collect.py', 'compact.py', 'planner.py', 'hillclimb.py')),
                   DATA.parent / 'scoring.py'],
        'fresh': list((source / 'splits').glob('*.json*')),
        'published': [DATA / f'{split}-questions.jsonl' for split in ('train', 'dev', 'test')]
        + [PUBLISHED / f'{prefix}-{split}.jsonl.gz' for split in ('train', 'dev', 'test')
           for prefix in ('compact-4b', 'critic-v2-4b')],
    }
    digests = {f'{group}/{path.name}': sha(path) for group, paths in groups.items() for path in sorted(paths)}
    pairs = source / 'train-pairs.jsonl'
    saved = pairs.read_bytes() if pairs.exists() else gzip.decompress(pairs.with_suffix('.jsonl.gz').read_bytes())
    return digests | {'fresh/train-pairs.jsonl': hashlib.sha256(saved).hexdigest()}


def train(source):
    old_train, old_dev = published('train'), published('dev')
    extra = fresh(source, 'train')
    published_test = read(DATA / 'test-questions.jsonl')
    fresh_test = read(source / 'splits/test-questions.jsonl')
    check_disjoint(old_train, extra, old_dev, published_test, fresh_test)
    return prepare(old_train + extra), prepare(old_dev)


def restored(item):
    snapshots = dict(item)
    for kind in ('front', 'after_compact', 'after_short', 'compare'):
        snapshots[kind] = Ridge(**{key: np.array(value) for key, value in snapshots[kind].items()})
    return Learner(**snapshots)


def summarize(data, result):
    output = {}
    for domain in ('all', 'math', 'science'):
        mask = data.domain == domain if domain != 'all' else np.ones(len(data.ids), dtype=bool)
        output[domain] = {
            'n': int(mask.sum()), 'correct': int(result['correct'][mask].sum()),
            'accuracy': float(result['correct'][mask].mean()),
            'mean_tokens': float(result['tokens'][mask].mean()),
            'mean_solver_equivalent_tokens': float(result['tokens'][mask].mean() * COMPUTE),
            'short_first': int(result['first_short'][mask].sum()), 'retries': int(result['retry'][mask].sum()),
            'submitted': dict(Counter(('compact', 'recovered', 'short')[i] for i in result['selected'][mask])),
        }
    return output


def for_bootstrap(result):
    return {'correct': result['correct'], 'work': result['tokens'] * COMPUTE}


def oracle(data, base):
    recovered = data.correct[:, :2].max(axis=1).astype(bool)
    successes = int(recovered.sum())
    credit = int(np.sum(base['tokens'] - data.costs[:, 0]))
    costs = data.costs[(~recovered) & (data.correct[:, 2] == 1), 1]
    affordable = int(np.count_nonzero(np.cumsum(np.sort(costs)) < credit))
    return {'no_call_correct': successes, 'maximum_correct_at_lower_compute': successes + affordable,
            'retry_only_wins_available': len(costs)}


def evaluate(rows, locked, output):
    started = time.perf_counter()
    data = prepare(rows)
    results = baselines(data)
    reference = for_bootstrap(results['retry_on_invalid'])
    rng = np.random.default_rng(20261116)
    for name, chosen in locked['selected'].items():
        model = restored(chosen['model'])
        predictions = model.predict(data)
        results[name] = measure(data, *decide(data, predictions, chosen['config']))
    elapsed = time.perf_counter() - started
    details = {name: summarize(data, result) for name, result in results.items()}
    payload = {'methods': details, 'oracle_compact_first': oracle(data, results['retry_on_invalid']),
               'paired_vs_invalid': {name: paired(for_bootstrap(result), reference, rng)
                                     for name, result in results.items() if name != 'retry_on_invalid'}}
    with new_run(output) as pending:
        save(pending / 'summary.json', payload)
        with (pending / 'per_question.jsonl').open('w', encoding='utf-8') as stream:
            for i, key in enumerate(data.ids):
                outcomes = {name: {'correct': int(r['correct'][i]), 'tokens': int(r['tokens'][i]),
                                   'submit': int(r['selected'][i]), 'short_first': bool(r['first_short'][i]),
                                   'retry': bool(r['retry'][i])} for name, r in results.items()}
                stream.write(json.dumps({'id': key, 'domain': data.domain[i], 'outcomes': outcomes}) + '\n')
    for name, summary in details.items():
        stats = summary['all']
        print(f'{name:<32} {stats["accuracy"]:>7.2%} {stats["mean_solver_equivalent_tokens"]:>8.2f} '
              f'equivalent tokens, {stats["retries"]} retries')
    print(f'Feature extraction and inference CPU seconds: {elapsed:.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('tune', 'published', 'fresh'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frozen = args.output / 'frozen.json'
    if args.phase == 'tune':
        training, development = train(args.data)
        chosen = selection(training, development)
        with new_run(args.output) as pending:
            save(pending / 'frozen.json', {'evidence': inputs(args.data), 'train_count': len(training.ids),
                                           'dev_count': len(development.ids), 'selected': chosen})
        print(json.dumps({name: {k: v for k, v in item.items() if k != 'model'} for name, item in chosen.items()}, indent=2))
        return

    locked = json.loads(frozen.read_bytes())
    if locked['evidence'] != inputs(args.data):
        raise ValueError('Frozen inputs or policy source changed')
    rows = published('test') if args.phase == 'published' else fresh(args.data, 'test')
    evaluate(rows, locked, args.output / args.phase)


if __name__ == '__main__':
    main()
