"""Training-only paired reasoning reviews and short compact-solver repairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

from semantic_data import choose, read, records, write
import extended_acquire as old
from review_fields import parse as fields
from scoring import correct
from transport import resilient


def reasoning_prompt(row, current):
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    request = ('Independently solve the original problem, checking the pivotal calculation or scientific assumption. '
               'Keep your reasoning under 220 words. Then assess whether the CURRENT final answer is correct. '
               'After reasoning return ONLY JSON with fields in this order: '
               f'"answer" ({unit}), "p_correct" (probability CURRENT answer is correct), '
               '"action" ("stop", "repair", or "retry"), '
               '"check" (at most 55 words with the decisive justification). '
               'STOP submits CURRENT, REPAIR revises CURRENT using your review, RETRY solves afresh.\n\n'
               + row['question'].removesuffix(old.SCIENCE_SUFFIX) + '\n\nCURRENT solution:\n' + current['text'])
    return '<|im_start|>user\n' + request + '<|im_end|>\n<|im_start|>assistant\n<think>\n'


def short_repair(row, current, review):
    unit = 'one option letter' if row['domain'] == 'science' else 'a number'
    packet = 'Independent answer: ' + review['answer'] + '\nCheck: ' + str(review['verdict'].get('check', ''))
    return old.chat('Correct the CURRENT solution using the independent review. Check that the proposed answer '
                    'answers the original question. Write a line "Final: X" FIRST, where X is ' + unit +
                    ', then at most 35 words justifying it.\n\n' + row['question'].removesuffix(old.SCIENCE_SUFFIX) +
                    '\n\nCURRENT solution:\n' + current['text'] + '\n\nREVIEW:\n' + packet)


def inference(request, row, prompt, reviewer, cap):
    model, digest = (old.JUDGE, old.JUDGE_DIGEST) if reviewer else (old.MODEL, old.MODEL_DIGEST)
    logical = 'reasoning-review-v1' if reviewer else 'repair'
    seed = int(hashlib.sha256(f'{old.SEED}:{row["id"]}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31
    started = time.monotonic()
    result = request('/api/generate', {'model':model, 'prompt':prompt, 'raw':True, 'stream':False,
                     'logprobs':not reviewer, 'keep_alive':'30m', 'options':{
                         'temperature':0 if reviewer else .65, 'seed':seed, 'num_predict':cap, 'num_ctx':4096}})
    if not result.get('done') or result.get('model') != model or not result.get('prompt_eval_count'):
        raise ValueError('Incomplete inference receipt')
    text = result['response']
    node = {'model':model, 'model_digest':digest, 'text':text,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(), 'seed':seed,
            'temperature':0 if reviewer else .65, 'output_cap':cap, 'finish':result['done_reason'],
            'input_tokens':int(result['prompt_eval_count']), 'output_tokens':int(result.get('eval_count',0)),
            'latency_seconds':time.monotonic()-started, 'backend':result.get('srr_backend')}
    if reviewer:
        thought, end, body = text.rpartition('</think>')
        if not end:
            thought, body = '', text if text.lstrip().startswith('{') else ''
        obj, valid, complete = fields(body)
        answer = old.parse_answer(row['domain'], 'Final: ' + str(obj.get('answer','')))
        p = obj.get('p_correct', .5); action = obj.get('action','repair')
        if type(p) not in (int,float) or not 0 <= p <= 1:
            p = .5
        if action not in ('stop','repair','retry'):
            action = 'repair'
        node.update(raw_verdict=obj, verdict={**obj, 'solution':thought},
                    reasoning_source='observed text before </think>', parse_valid=valid,
                    json_complete=complete, answer=answer, p_correct=p, action=action)
    else:
        if text and not result.get('logprobs'):
            raise ValueError('Missing compact token probabilities')
        node.update(answer=old.parse_answer(row['domain'],text),
                    confidence=old.confidence(text,result.get('logprobs',[]),row['domain']))
    return node


def run(args):
    import modal
    rows, trees, judges = records(args.data, 'train', False)
    selected = choose(rows, {'math':args.n//2, 'chemistry':args.n//4, 'physics':args.n//4}, 'reasoning-pilot')
    worker = modal.Cls.from_name('srr-semantic-solver-20261120','Ollama')()
    request = resilient(worker,args.output.parent/'transport-errors.jsonl')
    tags = {r['name']:r['digest'] for r in request('/api/tags')['models']}
    assert tags[old.MODEL] == old.MODEL_DIGEST and tags[old.JUDGE] == old.JUDGE_DIGEST
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stream.seek(0); saved = [json.loads(s) for s in stream]
        known = {(r['id'],r['kind']):r['response'] for r in saved}
        assert len(known) == len(saved)
        lock = threading.Lock(); done = 0
        def add(row, kind, prompt, reviewer, cap):
            key = row['id'],kind
            if key not in known:
                node = inference(request,row,prompt,reviewer,cap)
                with lock:
                    stream.write(json.dumps({'id':row['id'],'kind':kind,'response':node})+'\n')
                    stream.flush(); known[key] = node
            assert known[key]['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
            return known[key]
        def one(row):
            nonlocal done
            current = trees[row['id']]['']
            review = add(row,'reasoning-review',reasoning_prompt(row,current),True,args.cap)
            add(row,'reasoning-short-repair',short_repair(row,current,review),False,128)
            if (row['id'],'') in judges:
                add(row,'standard-short-repair',short_repair(row,current,judges[row['id'],'']),False,128)
            with lock:
                done += 1
                if done % 8 == 0 or done == len(selected):
                    print(f'Training reasoning pilot {done}/{len(selected)}',flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one,selected))
    summary = {'n':len(selected),'split':'training only','fresh_data_used':False,'reasoning_output_cap':args.cap,
               'compact_repair_output_cap':128,'methods':{}}
    for kind in ('standard-review','reasoning-review','standard-guided-repair','standard-short-repair','reasoning-short-repair'):
        values=[]
        for row in selected:
            qid=row['id']
            if kind=='standard-review':node=judges.get((qid,''))
            elif kind=='standard-guided-repair':node=trees[qid].get('guided')
            else:node=known.get((qid,kind))
            if node is not None:values.append((row,node))
        summary['methods'][kind]={'n':len(values),'correct':sum(correct(n,r['answer']) for r,n in values),
            'mean_tokens':sum(n['input_tokens']+n['output_tokens'] for _,n in values)/len(values),
            'output_cap_reached':sum(n['output_tokens']>=n.get('output_cap',512 if 'review' in kind else 256) for _,n in values)}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n',type=int,default=96)
    parser.add_argument('--cap',type=int,default=1536)
    args=parser.parse_args()
    if args.n%4:parser.error('Use a multiple of four for the fixed subject mixture')
    run(args)
