"""Bounded reasoning followed by an explicit answer phase; both API calls are charged."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

from semantic_data import choose, read, records
import extended_acquire as old
from reasoning_pilot import reasoning_prompt, short_repair, inference
from review_fields import parse as fields
from scoring import correct
from transport import resilient


def phase(request, row, prompt, name, cap):
    seed=int(hashlib.sha256(f'{old.SEED}:{row["id"]}:{name}'.encode()).hexdigest()[:8],16)%2**31
    started=time.monotonic()
    result=request('/api/generate',{'model':old.JUDGE,'prompt':prompt,'raw':True,'stream':False,
        'keep_alive':'30m','options':{'temperature':0,'seed':seed,'num_predict':cap,'num_ctx':4096}})
    if not result.get('done') or result.get('model')!=old.JUDGE or not result.get('prompt_eval_count'):
        raise ValueError('Incomplete bounded-review receipt')
    return {'phase':name,'text':result['response'],'thinking':result.get('thinking',''),
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'seed':seed,
        'input_tokens':int(result['prompt_eval_count']),'output_tokens':int(result.get('eval_count',0)),
        'output_cap':cap,'temperature':0,'finish':result['done_reason'],
        'latency_seconds':time.monotonic()-started,'backend':result.get('srr_backend')}


def review(request,row,current,think_cap=256,answer_cap=256):
    original=reasoning_prompt(row,current)
    first=phase(request,row,original,'bounded-thought-v1',think_cap)
    thought,end,body=first['text'].rpartition('</think>')
    if not end:
        thought=first['thinking'] or first['text']
        body=first['text'] if first['text'].lstrip().startswith('{') else ''
    obj,valid,complete=fields(body)
    calls=[first]
    if not (valid and complete and obj.get('answer')):
        # Phase one is a deliberately bounded scratchpad, not a completed
        # answer retried according to correctness. Phase two is scheduled
        # whenever no complete decision exists, without accessing references.
        if end:
            thought=first['text'].rsplit('</think>',1)[0]
        prefix='{"answer": "'
        final_prompt=original+thought+'\n</think>\n\n'+prefix
        final=phase(request,row,final_prompt,'bounded-answer-v1',answer_cap)
        calls.append(final);body=prefix+final['text']
        obj,valid,complete=fields(body)
    p=obj.get('p_correct',.5);action=obj.get('action','repair')
    if type(p) not in (int,float) or not 0<=p<=1:p=.5
    if action not in ('stop','repair','retry'):action='repair'
    answer=old.parse_answer(row['domain'],'Final: '+str(obj.get('answer','')))
    return {'model':old.JUDGE,'model_digest':old.JUDGE_DIGEST,
        'protocol':'bounded-reasoning-review-v1','text':thought+'\n</think>\n'+body,
        'prompt_sha256':hashlib.sha256(json.dumps([r['prompt_sha256'] for r in calls]).encode()).hexdigest(),
        'input_tokens':sum(r['input_tokens'] for r in calls),'output_tokens':sum(r['output_tokens'] for r in calls),
        'output_cap':sum(r['output_cap'] for r in calls),'physical_api_calls':len(calls),'calls':calls,
        'latency_seconds':sum(r['latency_seconds'] for r in calls),'raw_verdict':obj,
        'verdict':{**obj,'solution':thought},'reasoning_source':'observed bounded scratchpad',
        'answer':answer,'p_correct':p,'action':action,'parse_valid':valid,'json_complete':complete}


def run(args):
    import modal
    rows,trees,judges=records(args.data,'train',False)
    selected=choose(rows,{'math':args.n//2,'chemistry':args.n//4,'physics':args.n//4},'reasoning-pilot')
    request=resilient(modal.Cls.from_name('srr-semantic-solver-20261120','Ollama')(),
                      args.output.parent/'transport-errors.jsonl')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stream.seek(0);saved=[json.loads(s) for s in stream]
        known={(r['id'],r['kind']):r['response'] for r in saved}
        assert len(known)==len(saved)
        # Exact already-completed control responses may be reused. No poor
        # completed answer is regenerated or selected by its correctness.
        previous=args.output.parent/'reasoning-pilot.jsonl'
        reusable={}
        if previous.exists():
            reusable={(r['id'],r['kind']):r['response'] for r in read(previous) if r['kind']=='standard-short-repair'}
        lock=threading.Lock();done=0
        def store(row,kind,node):
            with lock:
                stream.write(json.dumps({'id':row['id'],'kind':kind,'response':node})+'\n')
                stream.flush();known[row['id'],kind]=node
            return node
        def one(row):
            nonlocal done
            qid=row['id'];current=trees[qid]['']
            node=known.get((qid,'bounded-review'))
            if node is None:node=store(row,'bounded-review',review(request,row,current,args.think_cap,args.answer_cap))
            for kind,mentor in (('bounded-short-repair',node),('standard-short-repair',judges[qid,''])):
                prompt=short_repair(row,current,mentor)
                if (qid,kind) not in known:
                    cached=reusable.get((qid,kind))
                    if cached is not None:
                        assert cached['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
                    else:cached=inference(request,row,prompt,False,128)
                    store(row,kind,cached)
            with lock:
                done+=1
                if done%8==0 or done==len(selected):print(f'Bounded training pilot {done}/{len(selected)}',flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,selected))
    summary={'split':'training only','n':len(selected),'fresh_data_used':False,'think_cap':args.think_cap,
             'answer_cap':args.answer_cap,'compact_repair_cap':128,'methods':{}}
    for name in ('standard','bounded'):
        review_nodes=[judges[r['id'],''] if name=='standard' else known[r['id'],'bounded-review'] for r in selected]
        repairs=[known[r['id'],name+'-short-repair'] for r in selected]
        summary['methods'][name]={'review_correct':sum(correct(n,r['answer']) for r,n in zip(selected,review_nodes)),
            'repair_correct':sum(correct(n,r['answer']) for r,n in zip(selected,repairs)),
            'mean_review_tokens':sum(n['input_tokens']+n['output_tokens'] for n in review_nodes)/len(selected),
            'mean_repair_tokens':sum(n['input_tokens']+n['output_tokens'] for n in repairs)/len(selected),
            'mean_physical_review_calls':sum(n.get('physical_api_calls',1) for n in review_nodes)/len(selected),
            'invalid_review':sum(not n['parse_valid'] for n in review_nodes)}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n',type=int,default=96);parser.add_argument('--think-cap',type=int,default=256)
    parser.add_argument('--answer-cap',type=int,default=256)
    args=parser.parse_args()
    if args.n%4:parser.error('Use a multiple of four')
    run(args)
