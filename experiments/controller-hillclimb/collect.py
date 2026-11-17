"""Collect short, independently generated answer signals from a local model."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results' / 'cached-tree-dataset' / 'data'
sys.path.insert(0, str(DATA.parent))
from scoring import canonical_answer

OLLAMA = 'http://127.0.0.1:11434'
MODEL_SIZES = {'qwen3.5:0.8b': 0.8, 'qwen3.5:2b': 2.0, 'qwen3.5:4b': 4.0}
SCIENCE_SUFFIX = '\n\nChoose the best answer. In the JSON answer field, return only one option letter.'


def trees(split):
    with gzip.open(DATA / f'{split}-trees.jsonl.gz', 'rt', encoding='utf-8') as source:
        return [json.loads(line) for line in source]


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {'Content-Type': 'application/json'} if data is not None else {}
    with urlopen(Request(OLLAMA + path, data=data, headers=headers), timeout=60) as response:
        return json.load(response)


def prompt(domain, question):
    if domain == 'science':
        question = question.removesuffix(SCIENCE_SUFFIX)
        user = 'Choose the best answer. Reply only with the option letter.\n\n' + question
        prefix = 'The correct option is **'
        limit = 1
    else:
        user = (
            'Solve by writing a single arithmetic expression for the answer. Use numbers and + - * / ( ). '
            'Return the expression only, without evaluating it.\n\n'
            + question
        )
        prefix = 'The expression is `'
        limit = 32
    return f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{prefix}', limit


def collect(model, digest, domain, question):
    formatted, limit = prompt(domain, question)
    payload = {
        'model': model, 'prompt': formatted, 'raw': True, 'stream': False,
        'logprobs': True, 'top_logprobs': 20 if domain == 'science' else 0,
        'options': {'temperature': 0, 'num_predict': limit, 'num_ctx': 4096},
    }
    result = request('/api/generate', payload)
    if not result.get('done') or result.get('model') != model:
        raise RuntimeError(f'Incomplete response from {model}')
    text = result['response'].strip()
    probabilities = result.get('logprobs', [])
    if domain == 'math':
        expression = text.split('`', 1)[0].split('=', 1)[0].strip().replace(',', '')
        answer = ''
        if result['done_reason'] != 'length' or '`' in text:
            if re.fullmatch(r'[0-9.()+\-*/\s]+', expression):
                simplified = canonical_answer(expression)
                if re.fullmatch(r'-?\d+(?:/\d+)?', simplified):
                    answer = simplified
        scores = [value['logprob'] for value in probabilities if math.isfinite(value['logprob'])]
        evidence = {'mean_logprob': sum(scores) / len(scores) if scores else None}
    else:
        labels = dict.fromkeys(re.findall(r'^([A-J])\. ', question, re.MULTILINE), 0.0)
        for token in probabilities[0].get('top_logprobs', []) if probabilities else []:
            label = token['token'].strip()
            if label in labels:
                labels[label] += math.exp(token['logprob'])
        total = sum(labels.values())
        evidence = {label: value / total for label, value in labels.items()} if total else {}
        answer = max(evidence, key=evidence.get) if evidence else ''
    return {
        'model': model, 'model_digest': digest,
        'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(),
        'text': text, 'answer': answer, 'finish': result['done_reason'],
        'input_tokens': int(result['prompt_eval_count']),
        'output_tokens': int(result.get('eval_count', 0)),
        'evidence': evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=['train', 'dev', 'test'])
    parser.add_argument('--model', choices=MODEL_SIZES, required=True)
    parser.add_argument('--domain', choices=['math', 'science', 'all'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    models = {item['name']: item['digest'] for item in request('/api/tags')['models']}
    digest = models[args.model]
    rows = [row for row in trees(args.split) if args.domain in ('all', row['domain'])][:args.limit]
    existing = {}
    if args.output.exists():
        with args.output.open(encoding='utf-8') as stream:
            existing = {row['id']: row for row in map(json.loads, stream)}
        expected = {row['id']: row for row in rows}
        for key, saved in existing.items():
            row = expected[key]
            formatted, _ = prompt(row['domain'], row['question'])
            if (saved['model_digest'] != digest or saved['model'] != args.model
                    or saved['prompt_sha256'] != hashlib.sha256(formatted.encode()).hexdigest()):
                raise ValueError(f'Collection settings changed for {key}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a', encoding='utf-8') as stream:
        for index, row in enumerate(rows, 1):
            if row['id'] not in existing:
                signal = collect(args.model, digest, row['domain'], row['question'])
                stream.write(json.dumps({'id': row['id'], **signal}) + '\n')
                stream.flush()
            if index % 10 == 0 or index == len(rows):
                print(f'{args.model} {args.split} {args.domain}: {index}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
