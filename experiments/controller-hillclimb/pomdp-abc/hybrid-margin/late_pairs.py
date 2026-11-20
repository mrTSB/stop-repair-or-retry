"""Paired reviews and reviewed repairs after the first continuation, with full receipts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import threading

from margin_data import read,wait_complete,choose
import extended_acquire as old
from review_pilot import inference
from review_fields import parse as parse_fields
from extended_study import records
from difficulty_pomdp import MissingObservation
from context_pomdp import equal,mentor_comparison,outcome
from transport import resilient

CHECKPOINTS=('repair','retry','guided')


def judge_prompt(row,path,tree):
    return old.judge_prompt(row,path,tree).replace(
        'REPAIR asks the compact solver to check its solution again',
        'REPAIR asks the compact solver to revise it using your review')


def sanitized(node):
    result=dict(node)
    p=node.get('p_correct');action=node.get('action')
    if type(p) not in (int,float) or not 0<=p<=1:result['p_correct']=.5
    if action not in ('stop','repair','retry'):result['action']='repair'
    return result


def collect(args):
    import modal
    worker=modal.Cls.from_name(args.app,'Ollama')()
    request=resilient(worker,args.data/'transport-errors.jsonl')
    groups={'train':wait_complete(args.data,'train',args.wait)}
    if args.overlap_with_dev:
        # Once all pilot requests finish, its two slots are available while the
        # older collector finishes development. Keep the combined limit at four.
        selected=choose(groups['train'][0],{'math':48,'chemistry':24,'physics':24},'review-pilot')
        with (args.data.parent/'review-pilot.jsonl').open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            stream.seek(0);keys={(r['id'],r['variant'],r['kind']) for r in map(json.loads,stream)}
            if not all((r['id'],v,k) in keys for r in selected
                       for v in ('independent','thorough','brief','answer-first','challenge')
                       for k in ('review','repair')):
                raise ValueError('Finish all pilot variants before overlapping collection')
    else:
        groups['dev']=wait_complete(args.data,'dev',args.wait)
    requested=('train','dev') if args.split=='both' else (args.split,)
    for split in requested:
        if split not in groups:groups[split]=wait_complete(args.data,split,args.wait)
        rows,trees,judges=groups[split]
        target=args.data/f'{split}-late-observations.jsonl'
        with target.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            stream.seek(0);saved=[json.loads(line) for line in stream]
            known={(r['id'],r['kind'],r['path']):r['response'] for r in saved}
            if len(known)!=len(saved):raise ValueError('Duplicate late observation')
            lock=threading.Lock();completed=0

            def add(row,kind,path,formatted,logical):
                key=row['id'],kind,path
                if key not in known:
                    node=inference(request,row,formatted,'brief',kind=='judge',logical)
                    if kind=='judge':node=sanitized(node)
                    node['protocol']='late-reviewed-repair-v1'
                    with lock:
                        stream.write(json.dumps({'id':row['id'],'kind':kind,'path':path,'response':node})+'\n')
                        stream.flush();known[key]=node
                return known[key]

            def one(row):
                nonlocal completed
                tree=trees[row['id']]
                for path in CHECKPOINTS:
                    judge=add(row,'judge',path,judge_prompt(row,path,tree),path)
                    formatted=old.repair_prompt(row,tree[path],old.feedback(judge))
                    add(row,'solver',path+'/reviewed',formatted,path+'/repair')
                with lock:
                    completed+=1
                    if completed%8==0 or completed==len(rows):
                        print(f'{split} late paired {completed}/{len(rows)}',flush=True)

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(one,rows))


def merged_records(source,split,complete=True):
    rows,trees,judges=records(source,split,complete)
    # Stage-one reviews use a different prompt. An old review cannot stand in for
    # an unacquired new review merely because both share a checkpoint label.
    judges={key:node for key,node in judges.items() if key[1]==''}
    path=source/f'{split}-late-observations.jsonl'
    if not path.exists():path=path.with_suffix('.jsonl.gz')
    extra=read(path)
    known={(r['id'],r['kind'],r['path']):r['response'] for r in extra}
    ids={r['id'] for r in rows}
    allowed={(i,k,p if k=='judge' else p+'/reviewed') for i in ids
             for k in ('judge','solver') for p in CHECKPOINTS}
    if len(known)!=len(extra) or not set(known)<=allowed or (complete and set(known)!=allowed):
        raise ValueError('Unexpected or incomplete late observations')
    for row in rows:
        tree=trees[row['id']]
        for p in CHECKPOINTS:
            review=known.get((row['id'],'judge',p))
            node=known.get((row['id'],'solver',p+'/reviewed'))
            for current,reviewer in ((review,True),(node,False)):
                if current is None:continue
                if current['protocol']!='late-reviewed-repair-v1':raise ValueError('Changed late protocol')
                expected=judge_prompt(row,p,tree) if reviewer else old.repair_prompt(row,tree[p],old.feedback(review))
                logical=p if reviewer else p+'/repair'
                seed=int(hashlib.sha256(f'{old.SEED}:{row["id"]}:{logical}'.encode()).hexdigest()[:8],16)%2**31
                model,digest=(old.JUDGE,old.JUDGE_DIGEST) if reviewer else (old.MODEL,old.MODEL_DIGEST)
                cap=512 if reviewer else 256
                if (current['model'],current['model_digest'],current['seed'],current['temperature'],current['output_cap'])!=(model,digest,seed,0 if reviewer else .65,cap):
                    raise ValueError('Changed late inference settings')
                if current['prompt_sha256']!=hashlib.sha256(expected.encode()).hexdigest():raise ValueError('Changed late prompt')
                if current['input_tokens']<=0 or not 0<=current['output_tokens']<=cap:raise ValueError('Invalid late token charge')
                if reviewer:
                    obj,valid,complete_json=parse_fields(current['text'])
                    answer=old.parse_answer(row['domain'],'Final: '+str(obj.get('answer','')))
                    if (obj,valid,complete_json,answer)!=(current['verdict'],current['parse_valid'],current['json_complete'],current['answer']):
                        raise ValueError('Changed late review parsing')
                    expected_fields=sanitized({'p_correct':obj.get('p_correct',.5),'action':obj.get('action','repair')})
                    if (current['p_correct'],current['action'])!=(expected_fields['p_correct'],expected_fields['action']):
                        raise ValueError('Changed late decision fallback')
                elif current['answer']!=old.parse_answer(row['domain'],current['text']):raise ValueError('Changed late solver parsing')
            if review is not None:judges[row['id'],p]=review
            if node is not None:tree[p+'/reviewed']=node
    return rows,trees,judges


def llm_policy(row,tree,judges,settings):
    path='';visited=[''];acquired=[];decisions=[];previous=None
    for remaining in (2,1):
        if path and settings.get('reuse_review',False) and mentor_comparison(previous,tree[path])==2:
            decisions.append({'path':path,'action':'stop','queried':False});break
        if (row['id'],path) not in judges:raise MissingObservation('judge',path)
        judge=judges[row['id'],path];previous=judge;acquired.append(path)
        threshold=settings.get('judge_stop','action');action=judge['action']
        if threshold=='agreement':action='stop' if equal(judge['answer'],tree[path]['answer']) else 'repair'
        elif threshold!='action':action='stop' if judge['p_correct']>=threshold else (action if action!='stop' else 'repair')
        decisions.append({'path':path,'action':action,'queried':True})
        if action=='stop':break
        edge=('guided' if not path else 'reviewed') if action=='repair' else action
        path=f'{path}/{edge}' if path else edge
        if path not in tree:raise MissingObservation('solver',path)
        visited.append(path)
    return outcome(row,tree,judges,path,visited,acquired,decisions)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--app',default='srr-hybrid-margin-20261119')
    parser.add_argument('--wait',type=int,default=21600)
    parser.add_argument('--overlap-with-dev',action='store_true')
    parser.add_argument('--split',choices=('train','dev','both'),default='both')
    args=parser.parse_args()
    if not 1<=args.workers<=2:parser.error('Use at most two workers')
    collect(args)
