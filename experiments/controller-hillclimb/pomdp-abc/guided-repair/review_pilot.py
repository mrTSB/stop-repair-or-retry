"""Training-only probes of independent review and a shared longer solver prompt."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from guided_solver import CAP, chat, confidence, parse_answer, read, repair, request
from acquire import JUDGE, JUDGE_DIGEST, MODEL, MODEL_DIGEST, SCIENCE_SUFFIX, SEED


def first(row):
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    formatted = chat(
        row['question'].removesuffix(SCIENCE_SUFFIX) + '\n\n'
        'Work through the problem carefully. Check the assumptions, units, and calculations. '
        f'Give at most 90 words of scratch work, then a line "Final: X" where X is {unit}.')
    response = request('/api/generate', {
        'model': MODEL, 'prompt': formatted, 'raw': True, 'stream': False,
        'logprobs': True, 'keep_alive': '30m',
        'options': {'temperature': 0, 'seed': SEED, 'num_predict': CAP, 'num_ctx': 4096}})
    if not response.get('done') or response.get('model') != MODEL:
        raise ValueError('Incomplete compact response')
    text = response['response']
    return {'model': MODEL, 'model_digest': MODEL_DIGEST, 'text': text,
            'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(),
            'answer': parse_answer(row['domain'], text), 'finish': response['done_reason'],
            'input_tokens': int(response['prompt_eval_count']),
            'output_tokens': int(response.get('eval_count', 0)),
            'confidence': confidence(text, response.get('logprobs', []), row['domain'])}


def review(row, current, judge_model=JUDGE, judge_digest=JUDGE_DIGEST):
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    formatted = chat(
        'Control a compact solver by independently checking the original problem. '
        'First solve it yourself, explicitly checking each calculation or scientific principle. '
        'Then compare the CURRENT solution with your independent solution. '
        'Evaluate whether the CURRENT final answer is correct; different reasoning can still give a correct answer. '
        'STOP submits the CURRENT answer, REPAIR asks the compact solver to revise it using your review, '
        'and RETRY starts a fresh compact solution. You have two further solver calls. '
        'Choose the action that maximizes final answer accuracy. '
        'Return ONLY JSON with fields in this order: '
        '"solution" (your independent solution, at most 90 words), '
        f'"answer" (your final answer, {unit}), '
        '"check" (the specific error or evidence that the current solution is correct), '
        '"p_correct" (probability the CURRENT final answer is correct, from 0 to 1), '
        '"action" ("stop", "repair", or "retry").\n\n'
        + row['question'].removesuffix(SCIENCE_SUFFIX)
        + '\n\nCURRENT solution:\n' + current['text'])
    start = time.monotonic()
    response = request('/api/generate', {
        'model': judge_model, 'prompt': formatted, 'raw': True, 'stream': False, 'keep_alive': '30m',
        'options': {'temperature': 0, 'seed': SEED, 'num_predict': 512, 'num_ctx': 4096}})
    if not response.get('done') or response.get('model') != judge_model:
        raise ValueError('Incomplete reviewer response')
    text = response['response']
    from extended_acquire import parse_verdict
    obj, valid = parse_verdict(text)
    return {'model': judge_model, 'model_digest': judge_digest, 'text': text, 'verdict': obj, 'parse_valid': valid,
            'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(),
            'finish': response['done_reason'], 'input_tokens': int(response['prompt_eval_count']),
            'output_tokens': int(response.get('eval_count', 0)),
            'latency_seconds': time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=24)
    parser.add_argument('--judge', choices=(JUDGE, 'qwen3.5:35b'), default=JUDGE)
    args = parser.parse_args()
    questions = {r['id']: r for r in read(args.source / 'splits/train-questions.jsonl')}
    trees = {r['id']: r['tree'] for r in read(args.source / 'train-trees.jsonl')}
    pilot_ids = {r['id'] for r in read(args.source / 'judge-deliberate-pilot.jsonl')}
    rows = []
    for domain in ('math', 'science'):
        candidates = [r for r in questions.values() if r['id'] in pilot_ids and r['domain'] == domain]
        rows += sorted(candidates, key=lambda r: hashlib.sha256(f'review-pilot:{r["id"]}'.encode()).digest())[:args.n // 2]
    if len(rows) != args.n:
        raise ValueError('Insufficient training pilot rows')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    if tags.get(MODEL) != MODEL_DIGEST or tags.get(JUDGE) != JUDGE_DIGEST:
        raise ValueError('Model weights changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a+', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        known = {(r['id'], r['kind']): r['response'] for r in map(json.loads, stream)}
        for index, row in enumerate(rows, 1):
            for kind in ('initial', 'review', 'guided'):
                key = row['id'], kind
                if key in known:
                    continue
                if kind == 'initial':
                    response = first(row)
                elif kind == 'review':
                    response = review(row, trees[row['id']][''], args.judge, tags[args.judge])
                else:
                    verdict = known[row['id'], 'review']['verdict']
                    hint = verdict['solution'] + '\nIndependent answer: ' + str(verdict['answer']) + '\n' + verdict['check']
                    response = repair(row, trees[row['id']][''], hint, 'full-review')
                stream.write(json.dumps({'id': row['id'], 'kind': kind, 'response': response}) + '\n')
                stream.flush()
                known[key] = response
            print(f'full-review training pilot {index}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
