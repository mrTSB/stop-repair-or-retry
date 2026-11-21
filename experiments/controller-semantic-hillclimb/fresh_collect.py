"""Collect only the frozen methods' union, keeping all test references hidden."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import threading

from semantic_data import read
import extended_acquire as old
from difficulty_pomdp import MissingObservation
from late_pairs import CHECKPOINTS,judge_prompt,sanitized
from review_pilot import inference
from semantic_study_common import evidence,models,run_policy,sha
from transport import resilient


def collect(args):
    import modal
    output=args.frozen.parent
    frozen=json.loads(args.frozen.read_bytes())
    if frozen['evidence']!=evidence(args.data):raise ValueError('Source or development evidence changed after freeze')
    verification=json.loads((output/'development-reproduction.json').read_bytes())
    if not verification['passed'] or verification['freeze_sha256']!=sha(args.frozen):
        raise ValueError('Refit and independently replay the selected development policies first')
    planners=models(frozen,output)
    rows=read(args.data/'splits/test-questions.jsonl')
    if len(rows)!=192 or len({r['id'] for r in rows})!=192:raise ValueError('Changed fresh test panel')
    identity={'freeze_sha256':sha(args.frozen),'questions':192,'rule':'All references hidden; no test scoring until complete.'}
    metadata=args.data/'test-collection.json'
    if metadata.exists():
        previous=json.loads(metadata.read_bytes())
        if any(previous.get(k)!=v for k,v in identity.items()):raise ValueError('Holdout already belongs to a different freeze')
        completed=set(previous['completed_ids'])
    else:completed=set()
    worker=modal.Cls.from_name(args.app,'Ollama')()
    request=resilient(worker,args.data/'transport-errors.jsonl')
    tags={r['name']:r['digest'] for r in request('/api/tags')['models']}
    if tags.get(old.MODEL)!=old.MODEL_DIGEST or tags.get(old.JUDGE)!=old.JUDGE_DIGEST:
        raise ValueError('Frozen solver or reviewer weights changed')
    old.request=request
    base_path=args.data/'test-observations.jsonl';late_path=args.data/'test-late-observations.jsonl'
    with base_path.open('a+') as base_stream,late_path.open('a+') as late_stream:
        fcntl.flock(base_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(late_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        streams={'base':base_stream,'late':late_stream};known={}
        for protocol,stream in streams.items():
            stream.seek(0);items=[json.loads(s) for s in stream]
            known[protocol]={(r['id'],r['kind'],r['path']):r['response'] for r in items}
            if len(known[protocol])!=len(items):raise ValueError('Duplicate fresh observations')
        ids={r['id'] for r in rows}
        if not completed<=ids or any(qid not in ids for k in known.values() for qid,kind,path in k):
            raise ValueError('Unexpected test IDs')
        cache_path=output/'test-neural-cache.jsonl'
        recorded=set()
        if cache_path.exists():
            for item in read(cache_path):recorded.add((item['family'],tuple(item['key'])))
        lock=threading.Lock()

        def append(row,protocol,kind,path,node):
            with lock:
                key=row['id'],kind,path
                if key in known[protocol]:raise ValueError('Duplicate concurrent observation')
                streams[protocol].write(json.dumps({'id':row['id'],'kind':kind,'path':path,'response':node})+'\n')
                streams[protocol].flush();known[protocol][key]=node

        def one(full_row):
            # Only identifiers and observable task text cross the decision boundary.
            row={k:full_row[k] for k in ('id','domain','subject','question')};row['answer']=''
            with lock:
                if row['id'] in completed:return
                local={protocol:{(kind,p):node for (qid,kind,p),node in values.items() if qid==row['id']}
                       for protocol,values in known.items()}
            tree={p:n for (kind,p),n in local['base'].items() if kind=='solver'}
            tree.update({p:n for (kind,p),n in local['late'].items() if kind=='solver'})
            base_judges={(row['id'],p):n for (kind,p),n in local['base'].items() if kind=='judge'}
            late_judges={k:v for k,v in base_judges.items() if k[1]==''}
            late_judges.update({(row['id'],p):n for (kind,p),n in local['late'].items() if kind=='judge'})

            def acquire(kind,path,protocol):
                if kind=='judge' and path not in tree:
                    acquire('solver',path,protocol)
                if kind=='solver' and path:
                    parent=path.rsplit('/',1)[0] if '/' in path else ''
                    if parent not in tree:acquire('solver',parent,protocol)
                    if path=='guided' and (row['id'],'') not in base_judges:
                        acquire('judge','','original')
                    if path.endswith('/reviewed') and (row['id'],parent) not in late_judges:
                        acquire('judge',parent,'late')
                if kind=='judge' and path and protocol=='late':
                    if path not in CHECKPOINTS:raise ValueError('Unexpected late checkpoint')
                    node=sanitized(inference(request,row,judge_prompt(row,path,tree),'brief',True,path))
                    node['protocol']='late-reviewed-repair-v1';storage='late'
                    late_judges[row['id'],path]=node
                elif kind=='solver' and path.endswith('/reviewed'):
                    parent=path.rsplit('/',1)[0]
                    review=late_judges[row['id'],parent]
                    formatted=old.repair_prompt(row,tree[parent],old.feedback(review))
                    node=inference(request,row,formatted,'brief',False,parent+'/repair')
                    node['protocol']='late-reviewed-repair-v1';storage='late';tree[path]=node
                else:
                    if (kind=='solver' and path not in old.PATHS) or (kind=='judge' and path not in old.CHECKPOINTS):
                        raise ValueError('Unexpected frozen observation request')
                    node=old.call(row,path,tree,base_judges,kind=='judge');storage='base'
                    if kind=='solver':tree[path]=node
                    else:
                        base_judges[row['id'],path]=node
                        if path=='':late_judges[row['id'],path]=node
                append(row,storage,kind,path,node)

            # Every fixed variant shares these exact calls and all adaptive methods replay them.
            for path in ('','repair','repair/repair','retry','retry/retry'):
                if path not in tree:acquire('solver',path,'original')
            for family in ('A','B','C'):
                configuration=frozen['selected'][family]
                while True:
                    try:
                        run_policy(family,row,tree,base_judges,late_judges,
                                   planners.get(family),configuration)
                        break
                    except MissingObservation as missing:
                        if family=='A' and missing.kind=='judge':raise ValueError('A attempted an LLM query')
                        acquire(missing.kind,missing.path,configuration['protocol'])
                    except KeyError as error:
                        key=error.args[0]
                        if isinstance(key,tuple) and len(key)==2 and key[0]==row['id']:
                            acquire('judge',key[1],configuration['protocol'])
                        elif isinstance(key,str) and key in old.PATHS:
                            acquire('solver',key,configuration['protocol'])
                        else:raise
            with lock:
                # Preserve the exact reference-free forward passes used for decisions.
                with cache_path.open('a') as neural_stream:
                    for family,model in planners.items():
                        descriptor=frozen['models'][family]
                        recipe_sha=sha(output/descriptor['observer_directory']/'recipe.json')
                        for key,values in list(model.classifier._cache.items()):
                            if key[1]!=row['id'] or (family,key) in recorded:continue
                            neural_stream.write(json.dumps({'family':family,'key':key,'recipe_sha256':recipe_sha,
                                'state':values[0].tolist(),'plain':values[1].tolist(),'guided':values[2].tolist(),
                                'tokens':values[3]})+'\n')
                            recorded.add((family,key))
                    neural_stream.flush()
                completed.add(row['id'])
                receipt={**identity,'completed':len(completed),'completed_ids':sorted(completed),
                         'base_observations':len(known['base']),'late_observations':len(known['late']),
                         'neural_cache_sha256':sha(cache_path)}
                temporary=metadata.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(metadata)
                if len(completed)%10==0 or len(completed)==192:
                    print(f'Frozen fresh union {len(completed)}/192; no accuracy inspected',flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(one,rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--app',default='srr-grounded-solver-20261120')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=4:parser.error('Use between one and four workers')
    collect(args)
