"""Prepare disjoint questions and collect unchanged paired solver responses."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sys


PARENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PARENT), str(PARENT / 'token-confidence')]

from collect_pairs import DATA, MODEL, ROOT, SCIENCE, SCIENCE_SUFFIX, collect, read, request, stem
from compact import parse_answer, prompt
from evaluate import MODEL_DIGEST, check_disjoint
from probe import confidence


PREVIOUS = DATA.parent.parent / 'controller-hillclimb/retry-gate/data'
SEED = 2026111703


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def existing_questions():
    files = [DATA / f'{split}-questions.jsonl' for split in ('train', 'dev', 'test')]
    files += list((PREVIOUS / 'splits').glob('*-questions.jsonl'))
    files += list((ROOT / '.cache/mmlu-pro/prepared').glob('*.jsonl'))
    files += [p for p in (ROOT / 'runs/gsm8k-large').glob('*.jsonl') if p.stem in ('train', 'dev', 'test')]
    return [question for file in files for question in read(file)]


def gsm8k(path, split):
    import pyarrow as pa

    with pa.memory_map(str(path), 'r') as source:
        rows = pa.ipc.open_stream(source).read_all().to_pylist()
    return [{'id': f'gsm8k-{split}-{i:05d}', 'domain': 'math', 'subject': 'math',
             'question': row['question'], 'answer': row['answer'].rsplit('####', 1)[1].strip()}
            for i, row in enumerate(rows) if row['question'] and row['answer'] and '####' in row['answer']]


def science(path):
    import pyarrow.parquet as pq

    groups = {subject: [] for subject in SCIENCE}
    keys = 'ABCDEFGHIJ'
    columns = ['question_id', 'question', 'options', 'answer', 'category']
    for row in pq.read_table(path, columns=columns).to_pylist():
        subject, options, answer = row['category'], row['options'], row['answer']
        if (subject not in groups or not options or not 2 <= len(options) <= 10 or answer not in keys[:len(options)]
                or len({value.strip().casefold() for value in options}) != len(options)):
            continue
        choices = '\n'.join(f'{key}. {value.strip()}' for key, value in zip(keys, options))
        groups[subject].append({'id': f'mmlu-pro-{subject}-{row["question_id"]:05d}', 'domain': 'science',
                                'subject': subject, 'answer': answer,
                                'question': f'Question: {row["question"]}\n\nOptions:\n{choices}{SCIENCE_SUFFIX}'})
    return groups


def prepare(args):
    previous = existing_questions()
    used_ids = {row['id'] for row in previous}
    used_stems = {stem(row['question']) for row in previous}
    exclusions = json.loads((DATA.parent / 'provenance/selection-manifest.json').read_bytes())
    used_ids.update(exclusions['historical_math_test_exclusions'])
    used_ids.update(f'gsm8k-train-{item:05d}' for item in exclusions['historical_math_training_exclusions'])
    used_ids.update(exclusions['historical_science_exclusions'])
    used_ids.update(entry['id'] for entry in exclusions['source_quality_exclusions'])

    def take(pool, name, number):
        def key(row):
            return hashlib.sha256(f'{SEED}:{name}:{row["id"]}'.encode()).digest()

        selected = []
        for row in sorted(pool, key=key):
            normalized = stem(row['question'])
            if row['id'] in used_ids or normalized in used_stems:
                continue
            selected.append(row)
            used_ids.add(row['id'])
            used_stems.add(normalized)
            if len(selected) == number:
                return selected
        raise ValueError(f'Could only find {len(selected)} of {number} eligible {name} rows')

    math_train, math_test = gsm8k(args.math_train, 'train'), gsm8k(args.math_test, 'test')
    science_groups = science(args.science)
    samples = {}
    for name, size in [('test', args.test), ('dev', args.dev), ('train', args.train)]:
        rows = take(math_test if name == 'test' else math_train, name, size // 2)
        for subject in SCIENCE:
            rows.extend(take(science_groups[subject], name, size // 6))
        random.Random(f'{SEED}:{name}').shuffle(rows)
        samples[name] = rows
    check_disjoint(*samples.values())
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'seed': SEED, 'source_sha256': {key: sha(getattr(args, key)) for key in ('math_train', 'math_test', 'science')},
                'splits': {}}
    for name, rows in samples.items():
        path = args.output / f'{name}-questions.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        manifest['splits'][name] = {'n': len(rows), 'subjects': dict(Counter(row['subject'] for row in rows)), 'sha256': sha(path)}
    (args.output / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def compact_call(row):
    domain, question = row['domain'], row['question']
    formatted = prompt(domain, question)
    response = request('/api/generate', {
        'model': MODEL, 'prompt': formatted, 'raw': True, 'stream': False, 'logprobs': True,
        'options': {'temperature': 0, 'num_predict': 128, 'num_ctx': 4096},
    })
    if response.get('model') != MODEL or not response.get('done'):
        raise RuntimeError('Incomplete compact response')
    text = response['response']
    logprobs = response.get('logprobs', [])
    if text and not logprobs:
        raise RuntimeError('Compact response is missing token probabilities')
    return {'model': MODEL, 'model_digest': MODEL_DIGEST, 'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(),
            'text': text, 'answer': parse_answer(domain, text), 'finish': response['done_reason'],
            'input_tokens': int(response['prompt_eval_count']), 'output_tokens': int(response.get('eval_count', 0)),
            'confidence': confidence(text, logprobs, domain)}


def collect_answers(args):
    rows = read(args.questions)
    if args.action == 'retries':
        plan = json.loads(args.plan.read_bytes())
        if plan['question_sha256'] != sha(args.questions):
            raise ValueError('Retry plan was compiled for different questions')
        ids = set(plan['retry_union'])
        rows = [row for row in rows if row['id'] in ids]
        if len(rows) != len(ids):
            raise ValueError('Retry plan contains unknown or duplicate question IDs')
    models = {entry['name']: entry['digest'] for entry in request('/api/tags')['models']}
    if models[MODEL] != MODEL_DIGEST:
        raise ValueError('The local model digest differs from the original compact solver')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        previous = [json.loads(line) for line in stream]
        seen = {row['id'] for row in previous}
        if len(seen) != len(previous) or not seen <= {row['id'] for row in rows}:
            raise ValueError('Collection file contains duplicate or unexpected rows')
        for index, row in enumerate(rows, 1):
            if row['id'] not in seen:
                record = {'id': row['id']}
                if args.action != 'retries':
                    record['compact'] = compact_call(row)
                if args.action != 'first':
                    record['short'] = collect(MODEL, MODEL_DIGEST, row['domain'], row['question'])
                stream.write(json.dumps(record) + '\n')
                stream.flush()
            if index % 30 == 0 or index == len(rows):
                print(f'{args.questions.stem} ({args.action}) {index}/{len(rows)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    collection = parser.add_subparsers(dest='action', required=True)
    split = collection.add_parser('prepare')
    for key in ('math-train', 'math-test', 'science', 'output'):
        split.add_argument(f'--{key}', type=Path, required=True)
    for key, size in [('train', 480), ('dev', 120), ('test', 300)]:
        split.add_argument(f'--{key}', type=int, default=size)
    for action in ('pairs', 'first', 'retries'):
        run = collection.add_parser(action)
        run.add_argument('--questions', type=Path, required=True)
        run.add_argument('--output', type=Path, required=True)
        if action == 'retries':
            run.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        if any(size <= 0 or size % 6 for size in (args.train, args.dev, args.test)):
            parser.error('Split sizes must be positive multiples of six')
        prepare(args)
    else:
        collect_answers(args)


if __name__ == '__main__':
    main()
