"""Capture answer confidence from the same compact decoding pass."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_pairs import MODEL, read, request
from compact import parse_answer, prompt
from evaluate import MODEL_DIGEST, fresh, published


FINAL = re.compile(r'\b(?:final(?:\s+answer)?|answer)\s*\**\s*:\s*([^\n]+)', re.IGNORECASE)
ANSWER = {
    'math': re.compile(r'[-+]?\d[\d.,/]*'),
    'science': re.compile(r'(?:(?i:option)\s+)?[\s\W]*[A-J]\b'),
}


def confidence(text, tokens, domain):
    probs, spans, offset = [], [], 0
    for entry in tokens:
        part = bytes(entry['bytes']) if 'bytes' in entry else entry['token'].encode()
        score = entry['logprob']
        probs.append(math.exp(score) if math.isfinite(score) else 0.0)
        spans.append((offset, offset + len(part), part))
        offset += len(part)

    blob = b''.join(item[2] for item in spans)
    response = text.encode()
    placement = blob.find(response)
    match = FINAL.search(text)
    answer = []
    if placement >= 0 and match:
        symbol = ANSWER[domain].search(match[1])
        if symbol:
            start = placement + len(text[:match.start(1) + symbol.start()].encode())
            end = placement + len(text[:match.start(1) + symbol.end()].encode())
            answer = [score for score, (left, right, _) in zip(probs, spans) if right > start and left < end]

    def stats(values):
        if not values:
            return [0.0, 0.0, 0.0]
        return [sum(values) / len(values), min(values), sum(value < .45 for value in values) / len(values)]

    return [bool(answer), *stats(answer), *stats(probs), *stats(probs[-12:]), *stats(probs[:12])]


def collect(row):
    domain, question = row['domain'], row['question']
    formatted = prompt(domain, question)
    response = request('/api/generate', {
        'model': MODEL, 'prompt': formatted, 'raw': True, 'stream': False, 'logprobs': True,
        'options': {'temperature': 0, 'num_predict': 128, 'num_ctx': 4096},
    })
    if response.get('model') != MODEL or not response.get('done'):
        raise RuntimeError('Incomplete compact response')
    text = response['response']
    logits = response.get('logprobs', [])
    if not logits:
        raise RuntimeError('The solver did not return completion token probabilities')
    record = {'model': MODEL, 'model_digest': MODEL_DIGEST,
              'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(), 'text': text,
              'answer': parse_answer(domain, text), 'finish': response['done_reason'],
              'input_tokens': int(response['prompt_eval_count']), 'output_tokens': int(response.get('eval_count', 0))}
    compared = row['compact']
    same = all(record[key] == compared[key] for key in ('text', 'input_tokens', 'output_tokens', 'finish'))
    return {'id': row['id'], 'confidence': confidence(text, logits, domain), 'compact': record,
            'matches_archive': same}


def sample(rows, per_domain):
    if per_domain is None:
        return rows
    result = []
    for domain in ('math', 'science'):
        group = [row for row in rows if row['domain'] == domain]
        order = sorted(group, key=lambda row: (bool(row['compact']['answer']),
                                              hashlib.sha256(('confidence:' + row['id']).encode()).digest()))
        result.extend(order[:per_domain])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('train', 'dev'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-per-domain', type=int, default=80)
    args = parser.parse_args()
    models = {entry['name']: entry['digest'] for entry in request('/api/tags')['models']}
    if models[MODEL] != MODEL_DIGEST:
        raise ValueError('Token confidence requires the published weights')
    sources = {'train': lambda: sample(published('train') + fresh(args.data, 'train'), args.train_per_domain),
               'dev': lambda: published('dev')}
    rows = sources[args.split]()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {row['id']: row for row in read(args.output)} if args.output.exists() else {}
    if not set(existing).issubset({row['id'] for row in rows}):
        raise ValueError('Token confidence output belongs to a different sample')
    with args.output.open('a', encoding='utf-8') as stream:
        for index, row in enumerate(rows, 1):
            if row['id'] not in existing:
                stream.write(json.dumps(collect(row)) + '\n')
                stream.flush()
            if index % 20 == 0 or index == len(rows):
                print(f'{args.split} confidence {index}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
