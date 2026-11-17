"""Collect a new low-token retry branch without modifying the solver cache."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from collect import SCIENCE_SUFFIX, canonical_answer, request, trees


MODELS = {'qwen3.5:2b': 2.0, 'qwen3.5:4b': 4.0, 'qwen3.5:9b': 9.0}


def prompt(domain, question):
    question = question.removesuffix(SCIENCE_SUFFIX)
    answer_format = 'one option letter' if domain == 'science' else 'a number'
    user = (
        f'{question}\n\nSolve with one short line of scratch work (such as an equation). '
        f'Immediately follow it with a line "Final: X" where X is {answer_format}. '
        'Keep the entire response under 45 words.'
    )
    return f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def parse_answer(domain, text):
    match = re.search(r'\b(?:final(?:\s+answer)?|answer)\s*\**\s*:\s*([^\n]+)', text, re.IGNORECASE)
    if not match:
        return ''
    candidate = match[1].strip().strip('*` ').strip()
    if domain == 'science':
        option = re.match(r'(?:option\s+)?\**\(?([A-J])\b', candidate, re.IGNORECASE)
        return option[1].upper() if option else ''
    answer = canonical_answer(candidate)
    return answer if re.fullmatch(r'-?\d+(?:/\d+)?', answer) else ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=['train', 'dev', 'test'])
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--domain', choices=['all', 'math', 'science'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tags = {item['name']: item['digest'] for item in request('/api/tags')['models']}
    digest = tags[args.model]
    rows = [r for r in trees(args.split) if args.domain in ('all', r['domain'])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.output.exists():
        with args.output.open(encoding='utf-8') as stream:
            existing = {row['id']: row for row in map(json.loads, stream)}
    with args.output.open('a', encoding='utf-8') as stream:
        for i, row in enumerate(rows, 1):
            formatted = prompt(row['domain'], row['question'])
            fingerprint = hashlib.sha256(formatted.encode()).hexdigest()
            if row['id'] in existing:
                saved = existing[row['id']]
                if (saved['model'], saved['model_digest'], saved['prompt_sha256']) != (args.model, digest, fingerprint):
                    raise ValueError(f'Retry settings changed for {row["id"]}')
            else:
                response = request('/api/generate', {
                    'model': args.model, 'prompt': formatted, 'raw': True, 'stream': False,
                    'logprobs': True, 'options': {'temperature': 0, 'num_predict': 128, 'num_ctx': 4096},
                })
                if response.get('model') != args.model or not response.get('done'):
                    raise ValueError(f'Incomplete response for {row["id"]}')
                text = response['response']
                result = {
                    'id': row['id'], 'model': args.model, 'model_digest': digest, 'prompt_sha256': fingerprint,
                    'text': text, 'answer': parse_answer(row['domain'], text), 'finish': response['done_reason'],
                    'input_tokens': int(response['prompt_eval_count']), 'output_tokens': int(response.get('eval_count', 0)),
                }
                stream.write(json.dumps(result) + '\n')
                stream.flush()
            if i % 10 == 0 or i == len(rows):
                print(f'compact {args.model} {args.split} {args.domain}: {i}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
