"""Collect frozen compact calls with probability margins and a new balanced split."""

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys


PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))

from data import DATA, MODEL, MODEL_DIGEST, ROOT, SCIENCE, collect, compact_call, existing_questions, gsm8k, read, request, science, sha, stem
from compact import parse_answer, prompt
from evaluate import check_disjoint, validate_response
from features import archived, index_responses, new_pairs, question_rows, records
from probe import ANSWER, FINAL, confidence
from scoring import correct


LAST = ROOT / 'results/controller-hillclimb/retry-gate/paired-uplift/data'
SEED = 2026111801
TOP_LOGPROBS = 5


def prepare(args):
    history = existing_questions()
    history += [row for split in ('train', 'dev', 'test') for row in question_rows(LAST, split)]
    ids = {row['id'] for row in history}
    stems = {stem(row['question']) for row in history}
    previous = json.loads((DATA.parent / 'provenance/selection-manifest.json').read_bytes())
    ids.update(previous['historical_math_test_exclusions'])
    ids.update(f'gsm8k-train-{i:05d}' for i in previous['historical_math_training_exclusions'])
    ids.update(previous['historical_science_exclusions'])
    ids.update(entry['id'] for entry in previous['source_quality_exclusions'])

    def take(pool, split, n):
        picked = []
        for row in sorted(pool, key=lambda row: hashlib.sha256(f'{SEED}:{split}:{row["id"]}'.encode()).digest()):
            title = stem(row['question'])
            if row['id'] in ids or title in stems:
                continue
            ids.add(row['id'])
            stems.add(title)
            picked.append(row)
            if len(picked) == n:
                return picked
        raise ValueError(f'Found only {len(picked)} of {n} unused questions for {split}')

    science_groups = science(args.science)
    test = take(gsm8k(args.math_test, 'test'), 'test', args.test // 2)
    dev = take(gsm8k(args.math_train, 'train'), 'dev', args.dev // 2)
    for subject in SCIENCE:
        test.extend(take(science_groups[subject], 'test', args.test // 6))
        dev.extend(take(science_groups[subject], 'dev', args.dev // 6))
    check_disjoint(dev, test)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'seed': SEED, 'source_sha256': {key: sha(getattr(args, key)) for key in ('math_train', 'math_test', 'science')},
                'splits': {}}
    for split, rows in [('dev', dev), ('test', test)]:
        random.Random(f'{SEED}:{split}').shuffle(rows)
        path = args.output / f'{split}-questions.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        manifest['splits'][split] = {'n': len(rows), 'subjects': dict(Counter(row['subject'] for row in rows)), 'sha256': sha(path)}
    (args.output / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def margin_features(text, tokens, domain, answer):
    spans, gaps, masses, offset = [], [], [], 0
    for item in tokens:
        raw = bytes(item['bytes']) if 'bytes' in item else item['token'].encode()
        spans.append((offset, offset + len(raw), raw))
        offset += len(raw)
        top = sorted((token['logprob'] for token in item.get('top_logprobs', []) if math.isfinite(token['logprob'])), reverse=True)
        gaps.append(min(top[0] - top[1], 16) if len(top) > 1 else 16)
        masses.append(min(1.0, sum(math.exp(value) for value in top)))
    start = b''.join(part for _, _, part in spans).find(text.encode())
    match = FINAL.search(text)
    selected = []
    if answer and start >= 0 and match:
        parsed = ANSWER[domain].search(match[1])
        if parsed:
            lo = start + len(text[:match.start(1) + parsed.start()].encode())
            hi = start + len(text[:match.start(1) + parsed.end()].encode())
            selected = [i for i, (left, right, _) in enumerate(spans) if right > lo and left < hi]

    def summarize(a, b):
        if not a:
            return [0.0, 0.0, 0.0, 0.0]
        return [sum(a) / (4 * len(a)), min(a) / 4, sum(value < 1 for value in a) / len(a), sum(b) / len(b)]

    alternatives = {}
    if domain == 'science':
        for i in selected:
            for item in tokens[i].get('top_logprobs', []):
                key = item['token'].strip()
                if len(key) == 1 and key in 'ABCDEFGHIJ' and math.isfinite(item['logprob']):
                    alternatives[key] = max(alternatives.get(key, 0), math.exp(item['logprob']))
    return {'features': [int(bool(selected)), *summarize([gaps[i] for i in selected], [masses[i] for i in selected]),
                         *summarize(gaps, masses), *summarize(gaps[-16:], masses[-16:]), *summarize(gaps[:16], masses[:16])],
            'alternatives': alternatives}


def compact(row):
    domain, question = row['domain'], row['question']
    formatted = prompt(domain, question)
    reply = request('/api/generate', {
        'model': MODEL, 'prompt': formatted, 'raw': True, 'stream': False, 'logprobs': True, 'top_logprobs': TOP_LOGPROBS,
        'options': {'temperature': 0, 'num_predict': 128, 'num_ctx': 4096},
    })
    if reply.get('model') != MODEL or not reply.get('done'):
        raise RuntimeError('Incomplete compact response')
    text = reply['response']
    answer = parse_answer(domain, text)
    tokens = reply.get('logprobs', [])
    if text and (not tokens or not any(item.get('top_logprobs') for item in tokens)):
        raise RuntimeError('Compact response is missing its top token probabilities')
    return {'model': MODEL, 'model_digest': MODEL_DIGEST, 'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(),
            'text': text, 'answer': answer, 'finish': reply['done_reason'],
            'input_tokens': int(reply['prompt_eval_count']), 'output_tokens': int(reply.get('eval_count', 0)),
            'confidence': confidence(text, tokens, domain), 'margin': margin_features(text, tokens, domain, answer)}


def prior_pairs():
    return archived() + new_pairs(LAST, 'train') + new_pairs(LAST, 'dev')


def probe_stratum(row):
    a, b = (correct(row[key], row['answer']) for key in ('compact', 'short'))
    transition = 'fix' if b and not a else 'regression' if a and not b else 'other'
    return row['domain'], 'valid' if row['compact']['answer'] else 'invalid', transition


def probe_sample(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[probe_stratum(row)].append(row)
    sample = []
    for key, group in sorted(buckets.items()):
        cap = {'fix': 30, 'regression': 22, 'other': 42}[key[-1]] if key[1] == 'valid' else 30
        picked = sorted(group, key=lambda row: hashlib.sha256(f'{SEED}:probe:{row["id"]}'.encode()).digest())[:cap]
        weight = len(group) / len(picked)
        sample.extend((row, weight) for row in picked)
    return sorted(sample, key=lambda item: hashlib.sha256(f'{SEED}:order:{item[0]["id"]}'.encode()).digest())


def prior_test():
    questions = question_rows(LAST, 'test')
    first = index_responses(questions, records(LAST, 'test-first'))
    retries = {row['id']: row['short'] for row in records(LAST, 'test-retries')}
    return [{**row, 'compact': first[row['id']]['compact'], 'short': retries.get(row['id'])} for row in questions]


def run(args):
    samples = []
    if args.action == 'probe':
        samples = probe_sample(prior_pairs())
    elif args.action == 'complete-prior':
        samples = [(row, None) for row in prior_test() if row['short'] is None]
    else:
        rows = read(args.questions)
        if args.action == 'retries':
            plan = json.loads(args.plan.read_bytes())
            if plan['question_sha256'] != sha(args.questions):
                raise ValueError('Retry plan references a different test split')
            needed = set(plan['retry_union'])
            rows = [row for row in rows if row['id'] in needed]
            if len(rows) != len(needed):
                raise ValueError('Retry plan references an unknown ID')
        samples = [(row, None) for row in rows]
    tags = {item['name']: item['digest'] for item in request('/api/tags')['models']}
    if tags[MODEL] != MODEL_DIGEST:
        raise ValueError('The compact weights changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        existing = [json.loads(line) for line in stream]
        known = {item['id'] for item in existing}
        if len(known) != len(existing) or not known <= {row['id'] for row, _ in samples}:
            raise ValueError('Unexpected rows in existing collection')
        for i, (row, weight) in enumerate(samples, 1):
            if row['id'] not in known:
                observation = {'id': row['id']}
                if args.action == 'first':
                    observation['compact'] = compact_call(row)
                if args.action in ('pairs', 'probe'):
                    observation['compact'] = compact(row)
                if args.action in ('pairs', 'retries', 'complete-prior'):
                    observation['short'] = collect(MODEL, MODEL_DIGEST, row['domain'], row['question'])
                if args.action == 'probe':
                    first = observation['compact']
                    if any(first[key] != row['compact'][key] for key in ('text', 'answer', 'input_tokens', 'output_tokens', 'finish')):
                        raise RuntimeError(f'Top-logprob capture changed the archived response for {row["id"]}')
                    observation['weight'] = weight
                stream.write(json.dumps(observation) + '\n')
                stream.flush()
            if i % 30 == 0 or i == len(samples):
                print(f'{args.action} {args.output.stem} {i}/{len(samples)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    select = actions.add_parser('prepare')
    for key in ('math-train', 'math-test', 'science', 'output'):
        select.add_argument(f'--{key}', required=True, type=Path)
    select.add_argument('--dev', type=int, default=180)
    select.add_argument('--test', type=int, default=600)
    for name in ('first', 'pairs', 'probe', 'complete-prior', 'retries'):
        action = actions.add_parser(name)
        action.add_argument('--output', required=True, type=Path)
        if name in ('first', 'pairs', 'retries'):
            action.add_argument('--questions', required=True, type=Path)
        if name == 'retries':
            action.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if any(size <= 0 or size % 6 for size in (args.dev, args.test)):
            parser.error('Split sizes must be positive multiples of six')
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
