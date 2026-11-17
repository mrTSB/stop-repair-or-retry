"""Collect compact one-token science retries using the existing model and trace."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from collect_pairs import MODEL, request
from evaluate import MODEL_DIGEST, fresh, jsonl, published
from signals import choices


def clip(text, limit):
    if len(text) <= limit:
        return text
    first = limit * 2 // 3
    return text[:first] + ' … ' + text[-(limit - first):]


def prompt(row, kind):
    options = choices(row['question'])
    compact_choices = '\n'.join(f'{key}. {clip(option, 45)}' for key, option in options.items())
    stem = row['question'].removeprefix('Question: ').split('\n\nOptions:\n', 1)[0]
    if kind == 'trace':
        work = row['compact']['text'].strip()
        body = f'{clip(stem, 175)}\nWork so far: {work[-250:]}\n{compact_choices}'
    else:
        body = f'{stem}\n{compact_choices}'
    formatted = ('<|im_start|>user\nPick the best choice. Output its letter only.\n'
                 f'{body}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe correct option is **')
    return formatted, options


def one_token(row, kind):
    formatted, options = prompt(row, kind)
    result = request('/api/generate', {
        'model': MODEL, 'prompt': formatted, 'raw': True, 'stream': False,
        'logprobs': True, 'top_logprobs': 20,
        'options': {'temperature': 0, 'num_predict': 1, 'num_ctx': 4096},
    })
    if not result.get('done') or result.get('model') != MODEL:
        raise RuntimeError('Incomplete choice retry')
    logits = result.get('logprobs', [])
    weights = dict.fromkeys(options, 0.0)
    for token in logits[0].get('top_logprobs', []) if logits else []:
        key = token['token'].strip()
        if key in weights:
            weights[key] += math.exp(token['logprob'])
    total = sum(weights.values())
    evidence = {key: value / total for key, value in weights.items()} if total else {}
    return {'answer': max(evidence, key=evidence.get) if evidence else '', 'evidence': evidence,
            'input_tokens': int(result['prompt_eval_count']), 'output_tokens': int(result.get('eval_count', 0)),
            'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(), 'model_digest': MODEL_DIGEST}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('train', 'dev', 'published', 'fresh'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--kind', choices=('trace', 'stem'), required=True)
    args = parser.parse_args()
    models = {entry['name']: entry['digest'] for entry in request('/api/tags')['models']}
    if models[MODEL] != MODEL_DIGEST:
        raise ValueError('Choice retry requires the published weights')
    sources = {'train': lambda: published('train') + fresh(args.data, 'train'), 'dev': lambda: published('dev'),
               'published': lambda: published('test'), 'fresh': lambda: fresh(args.data, 'test')}
    group = [row for row in sources[args.split]() if row['domain'] == 'science']
    output = args.data / f'{args.kind}-{args.split}.jsonl'
    existing = {row['id']: row for row in jsonl(output)} if output.exists() else {}
    with output.open('a', encoding='utf-8') as stream:
        for index, row in enumerate(group, 1):
            key = row['id']
            if key not in existing:
                stream.write(json.dumps({'id': key, **one_token(row, args.kind)}) + '\n')
                stream.flush()
            elif existing[key]['prompt_sha256'] != hashlib.sha256(prompt(row, args.kind)[0].encode()).hexdigest():
                raise ValueError(f'Changed choice retry prompt for {key}')
            if index % 25 == 0 or index == len(group):
                print(f'{args.kind} {args.split} {index}/{len(group)}', flush=True)


if __name__ == '__main__':
    main()
