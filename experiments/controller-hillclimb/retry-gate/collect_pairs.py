"""Prepare new question splits and collect the two unchanged Qwen3.5 calls."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import unicodedata

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))

from collect import DATA, ROOT, SCIENCE_SUFFIX, collect, request
from compact import parse_answer, prompt


MODEL = 'qwen3.5:4b'
SCIENCE = ('biology', 'chemistry', 'physics')
LETTERS = 'ABCDEFGHIJ'
SEED = 2026111602


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def stem(text):
    text = text.removeprefix('Question: ').split('\n\nOptions:\n', 1)[0]
    return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', text).casefold())


def order_key(row):
    return hashlib.sha256(f'{SEED}:{row["id"]}'.encode()).hexdigest()


def prepare(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    existing = [row for split in ('train', 'dev', 'test') for row in read(DATA / f'{split}-questions.jsonl')]
    for path in (ROOT / '.cache/mmlu-pro/prepared').glob('*.jsonl'):
        existing.extend(read(path))
    for path in (ROOT / 'runs/gsm8k-large').glob('[td]*.jsonl'):
        if path.stem in ('train', 'dev', 'test'):
            existing.extend(read(path))
    historical = json.loads((DATA.parent / 'provenance/selection-manifest.json').read_bytes())
    seen_ids = {row['id'] for row in existing}
    seen_ids.update(historical['historical_math_test_exclusions'])
    seen_ids.update(f'gsm8k-train-{item:05d}' for item in historical['historical_math_training_exclusions'])
    seen_ids.update(historical['historical_science_exclusions'])
    seen_ids.update(row['id'] for row in historical['source_quality_exclusions'])
    seen_stems = {stem(row['question']) for row in existing}

    math_rows = {}
    for split, path in [('train', args.math_train), ('test', args.math_test)]:
        with pa.memory_map(str(path), 'r') as source:
            table = pa.ipc.open_stream(source).read_all()
        math_rows[split] = [
            {'id': f'gsm8k-{split}-{i:05d}', 'domain': 'math', 'subject': 'math',
             'question': row['question'], 'answer': row['answer'].rsplit('####', 1)[1].strip()}
            for i, row in enumerate(table.to_pylist()) if '####' in row['answer']
        ]

    science_rows = {category: [] for category in SCIENCE}
    for row in pq.read_table(args.science, columns=['question_id', 'question', 'options', 'answer', 'category']).to_pylist():
        category, options, answer = row['category'], row['options'], row['answer']
        if (category not in SCIENCE or not 2 <= len(options) <= 10 or answer not in LETTERS[:len(options)]
                or len({item.strip().casefold() for item in options}) != len(options)):
            continue
        choices = '\n'.join(f'{key}. {value.strip()}' for key, value in zip(LETTERS, options))
        science_rows[category].append({
            'id': f'mmlu-pro-{category}-{row["question_id"]:05d}', 'domain': 'science', 'subject': category,
            'question': f'Question: {row["question"]}\n\nOptions:\n{choices}{SCIENCE_SUFFIX}', 'answer': answer,
        })

    def sample(pool, count):
        result = []
        for row in sorted(pool, key=order_key):
            key = stem(row['question'])
            if row['id'] in seen_ids or key in seen_stems:
                continue
            seen_ids.add(row['id'])
            seen_stems.add(key)
            result.append(row)
            if len(result) == count:
                return result
        raise ValueError(f'Insufficient fresh questions for sample of {count}')

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'seed': SEED, 'splits': {}, 'sources': {}}
    for name, path in [('gsm8k_train', args.math_train), ('gsm8k_test', args.math_test), ('mmlu_pro', args.science)]:
        manifest['sources'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for split, size in [('train', args.train), ('test', args.test)]:
        rows = sample(math_rows[split], size // 2)
        for category in SCIENCE:
            rows.extend(sample(science_rows[category], size // 6))
        random.Random(f'{SEED}:{split}').shuffle(rows)
        output = args.output / f'{split}-questions.jsonl'
        output.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        manifest['splits'][split] = {'count': len(rows), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    (args.output / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def compact_answer(digest, domain, question):
    formatted = prompt(domain, question)
    response = request('/api/generate', {
        'model': MODEL, 'prompt': formatted, 'raw': True, 'stream': False, 'logprobs': True,
        'options': {'temperature': 0, 'num_predict': 128, 'num_ctx': 4096},
    })
    if response.get('model') != MODEL or not response.get('done'):
        raise RuntimeError('Incomplete compact response')
    text = response['response']
    return {'model': MODEL, 'model_digest': digest,
            'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(),
            'text': text, 'answer': parse_answer(domain, text), 'finish': response['done_reason'],
            'input_tokens': int(response['prompt_eval_count']), 'output_tokens': int(response.get('eval_count', 0))}


def collect_split(args):
    models = {item['name']: item['digest'] for item in request('/api/tags')['models']}
    digest = models[MODEL]
    if digest != args.digest:
        raise ValueError('The installed model does not match the published compact solver')
    questions = read(args.questions)
    done = {row['id']: row for row in read(args.output)} if args.output.exists() else {}
    if not set(done).issubset({question['id'] for question in questions}):
        raise ValueError('Existing output contains a different question split')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a', encoding='utf-8') as stream:
        for index, row in enumerate(questions, 1):
            key, domain, question = row['id'], row['domain'], row['question']
            if key not in done:
                first = compact_answer(digest, domain, question)
                retry = collect(MODEL, digest, domain, question)
                stream.write(json.dumps({'id': key, 'compact': first, 'short': retry}) + '\n')
                stream.flush()
            if index % 20 == 0 or index == len(questions):
                print(f'{args.questions.stem} {index}/{len(questions)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='action', required=True)
    make = subcommands.add_parser('prepare')
    for name in ('math-train', 'math-test', 'science', 'output'):
        make.add_argument(f'--{name}', required=True, type=Path)
    make.add_argument('--train', type=int, default=180)
    make.add_argument('--test', type=int, default=180)
    run = subcommands.add_parser('collect')
    run.add_argument('--questions', required=True, type=Path)
    run.add_argument('--output', required=True, type=Path)
    run.add_argument('--digest', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        if any(size <= 0 or size % 6 for size in (args.train, args.test)):
            parser.error('Split sizes must be positive multiples of six')
        prepare(args)
    else:
        collect_split(args)


if __name__ == '__main__':
    main()
