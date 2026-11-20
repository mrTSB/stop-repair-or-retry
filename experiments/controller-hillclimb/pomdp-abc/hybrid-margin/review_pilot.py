"""Training-only paired review and repair variants with full inference receipts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

from margin_data import PREVIOUS, ROOT, SEED, choose, read
import extended_acquire as old
from extended_study import records
from scoring import correct
from review_fields import parse as parse_fields
from transport import resilient

VARIANTS = ('independent', 'thorough', 'challenge', 'brief', 'answer-first')


def prompt(row, current, previous, variant):
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    science = ('Check the precise scientific assumptions, signs, units, and whether every selected option '
               'actually answers the question. Derive the result before matching an option; do not infer an '
               'answer from superficial wording. ' if row['domain'] == 'science' else
               'Translate the quantities carefully, calculate each step, and independently check the final arithmetic. ')
    instructions = ('Solve the problem independently and verify your own result. ' + science +
                    'Return ONLY JSON with fields "solution" (at most 160 words of checked reasoning), '
                    f'"answer" ({unit}), "check" (at most 24 words), '
                    '"p_correct" (probability the CURRENT answer is correct, from 0 to 1), '
                    '"action" ("stop", "repair", or "retry"). ')
    task = row['question'].removesuffix(old.SCIENCE_SUFFIX)
    if variant == 'brief':
        instructions = ('Independently check the problem and the CURRENT solution. ' + science +
                        'Return ONLY JSON with fields in this order: "solution" (at most 70 words), '
                        f'"answer" ({unit}), "p_correct" (probability CURRENT answer is correct), '
                        '"action" ("stop", "repair", or "retry"), "check" (at most 12 words). '
                        'Keep the solution short so the final answer and decision fields are completed.')
    if variant == 'answer-first':
        instructions = ('Check the CURRENT solution against the original question. ' + science +
                        'Return ONLY JSON with fields in this order: '
                        f'"answer" (the correct answer, {unit}), '
                        '"p_correct" (probability the CURRENT answer is correct), '
                        '"action" ("stop", "repair", or "retry"), '
                        '"solution" (at most 50 words supporting your answer), "check" (at most 12 words).')
    if variant == 'independent':
        instructions = ('You are an independent subject expert. ' + science +
                        'Do not use any proposed answer as evidence. Derive the answer from the question. '
                        'Return ONLY JSON with fields "solution" (at most 160 words of checked reasoning), '
                        f'"answer" ({unit}), "check" (a short independent verification), '
                        '"p_correct" (confidence in your independent result, from 0 to 1), '
                        '"action" (use "repair").')
        return old.chat(instructions + '\n\n' + task)
    if variant == 'challenge':
        instructions += ('Two proposed solutions follow. Either or both can be wrong. Explicitly check the '
                         'most consequential assumption or calculation in the expert proposal. Resolve '
                         'disagreement using an independent derivation rather than voting. ')
        proposal = '\n\nExpert proposal:\n' + old.feedback(previous)
    else:
        proposal = ''
    return old.chat(instructions + '\n\n' + task + '\n\nCURRENT solution:\n' + current['text'] + proposal)


def inference(request, row, formatted, variant, reviewer, logical_path=None):
    model, fingerprint = (old.JUDGE, old.JUDGE_DIGEST) if reviewer else (old.MODEL, old.MODEL_DIGEST)
    cap = (512 if variant=='brief' else 256 if variant=='answer-first' else 768) if reviewer else 256
    logical = ('' if reviewer else 'repair') if logical_path is None else logical_path
    seeded = (f'{old.SEED}:{row["id"]}:{logical}' if variant in ('brief','answer-first')
              else f'{SEED}:{row["id"]}:{variant}')
    seed = int(hashlib.sha256(seeded.encode()).hexdigest()[:8], 16) % 2**31
    started = time.monotonic()
    result = request('/api/generate', {'model': model, 'prompt': formatted, 'raw': True,
        'stream': False, 'logprobs': not reviewer, 'keep_alive': '30m',
        'options': {'temperature': 0 if reviewer else .65, 'seed': seed, 'num_predict': cap, 'num_ctx': 4096}})
    if not result.get('done') or result.get('model') != model or not result.get('prompt_eval_count'):
        raise RuntimeError('Incomplete inference receipt')
    text = result['response']
    node = {'model': model, 'model_digest': fingerprint, 'text': text,
            'prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(), 'seed': seed,
            'temperature': 0 if reviewer else .65, 'output_cap': cap, 'finish': result['done_reason'],
            'input_tokens': int(result['prompt_eval_count']), 'output_tokens': int(result.get('eval_count', 0)),
            'latency_seconds': time.monotonic()-started, 'backend': result.get('srr_backend')}
    if reviewer:
        if variant in ('brief','answer-first'):
            verdict,valid,complete=parse_fields(text)
            node['json_complete']=complete
            node['response_format']='json-completed-fields'
        else:
            verdict, valid = old.parse_verdict(text)
        node.update(verdict=verdict, parse_valid=valid,
                    p_correct=verdict.get('p_correct',.5), action=verdict.get('action','repair'))
        node['answer'] = old.parse_answer(row['domain'], 'Final: ' + str(verdict.get('answer', '')))
    else:
        if text and not result.get('logprobs'):
            raise RuntimeError('Missing compact token probabilities')
        node.update(answer=old.parse_answer(row['domain'], text),
                    confidence=old.confidence(text, result.get('logprobs', []), row['domain']))
    return node


def run(args):
    import modal
    worker = modal.Cls.from_name(args.app, 'Ollama')()
    request = resilient(worker,args.data/'transport-errors.jsonl')
    questions = {r['id']: r for r in read(args.data / 'splits/train-questions.jsonl')}
    selected = choose(list(questions.values()), {'math': args.n//2, 'chemistry': args.n//4,
                                               'physics': args.n//4}, 'review-pilot')
    trees, judges = {}, {}
    for split in ('train', 'dev', 'test'):
        _, group, reviews = records(PREVIOUS, split, split != 'test')
        trees.update(group)
        judges.update(reviews)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        known = {(r['id'],r['variant'],r['kind']):r['response'] for r in map(json.loads, stream)}
        lock = threading.Lock()

        def add(row, variant, kind, formatted):
            key = row['id'], variant, kind
            if key not in known:
                node = inference(request, row, formatted, variant, kind == 'review')
                with lock:
                    stream.write(json.dumps({'id':row['id'],'variant':variant,'kind':kind,'response':node})+'\n')
                    stream.flush()
                    known[key] = node
            return known[key]

        def one(row):
            for variant in args.variants:
                review = add(row, variant, 'review', prompt(row, trees[row['id']][''], judges[row['id'],''], variant))
                repair = old.repair_prompt(row, trees[row['id']][''], old.feedback(review))
                add(row, variant, 'repair', repair)
            with lock:
                finished = sum(all((r['id'],v,k) in known for v in args.variants for k in ('review','repair')) for r in selected)
                if finished % 8 == 0 or finished == len(selected):
                    print(f'Training paired pilot {finished}/{len(selected)}', flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(one, selected))
    results = {}
    for variant in args.variants:
        results[variant] = {'n':len(selected)}
        for kind in ('review','repair'):
            nodes = [known[r['id'],variant,kind] for r in selected]
            results[variant][kind] = {'correct':sum(correct(known[r['id'],variant,kind],r['answer']) for r in selected),
                                     'mean_tokens':sum(n['input_tokens']+n['output_tokens'] for n in nodes)/len(nodes),
                                     'truncated':sum(n['finish']=='length' for n in nodes)}
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n',type=int,default=96)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=['independent','thorough'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--app',default='srr-hybrid-margin-20261119')
    args = parser.parse_args()
    if args.n % 4 or not 1 <= args.workers <= 4:
        parser.error('Use a multiple of four questions and between one and four workers')
    run(args)
