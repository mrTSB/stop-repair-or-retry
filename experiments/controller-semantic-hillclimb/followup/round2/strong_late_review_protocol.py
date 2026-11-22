"""Exact paid protocol for one strong review after ordinary repair or retry.

This new timing study leaves the frozen root-review implementation intact.
The reference-free current prefix is the only solution supplied to the teacher;
its review guides one final128-token compact repair. No inference entrypoint.
"""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

import drop_review_repair_pilot as previous
import science_complete_actions as completion
import expanded_mentor_io as io
import long_review_pilot as reviewer
from reasoning_pilot import short_repair, reasoning_prompt

protocol=previous.protocol
SCHEMA='strong-late-review-timing-TRAIN-v1'
PREFIXES=('repair','retry')
COUNTS={'chemistry':32,'physics':32}
SCREEN={'novel_questions_beyond_plain_Complete_rootstrong_at_least':6,
        'mean_all128_full_trajectory_compute_over_rootstrong_retained_at_most':1.25,
        'scope':'TRAIN action headroom only; no controller, fresh-data or learnability claim'}


class MissingPhysical(Exception):
    def __init__(self,payload): self.payload=payload;super().__init__('Missing exact physical request')


def seeded_row(row,parent):
    if row.get('answer')!='' or row['subject'] not in COUNTS or parent not in PREFIXES:
        raise ValueError('Only a blank-reference science prefix may reach the late reviewer')
    return {**row,'id':row['id']+'::strong-late-'+parent+'-v1'}


def physical(raw,payload):
    previous.RequestJournal.validate(raw)
    if raw['payload']!=payload or raw['endpoint']!='/api/generate': raise ValueError('Physical prompt/seed changed')
    result=raw['response'];options=payload['options']
    if (result.get('model')!=payload['model'] or result.get('done') is not True
            or type(result.get('prompt_eval_count')) is not int or not 0<result['prompt_eval_count']<=options['num_ctx']
            or type(result.get('eval_count')) is not int or not 0<=result['eval_count']<=options['num_predict']
            or not math.isfinite(raw['returned_unix']-raw['started_unix']) or raw['returned_unix']<raw['started_unix']):
        raise ValueError('Invalid completed physical generation; preserve it without regeneration')


def saved_request(lookup,used,captured):
    def request(endpoint,payload):
        if endpoint!='/api/generate': raise ValueError('Unexpected replay endpoint')
        key=previous.digest({'endpoint':endpoint,'payload':payload})
        if key not in lookup: raise MissingPhysical(payload)
        raw=lookup[key];used.add(key);physical(raw,payload);captured.append(raw)
        return copy.deepcopy(raw['response'])
    return request


def review(row,tree,parent,lookup,used=None):
    used=set() if used is None else used;captured=[]
    teacher_row=seeded_row(row,parent)
    node=reviewer.review(saved_request(lookup,used,captured),teacher_row,tree[parent],1024,512)
    if len(node['calls']) not in (1,2) or len(node['calls'])!=len(captured):
        raise ValueError('Strong late review must retain its exact one/two physical phases')
    for call,raw in zip(node['calls'],captured):
        call['latency_seconds']=raw['returned_unix']-raw['started_unix']
    node['latency_seconds']=sum(c['latency_seconds'] for c in node['calls'])
    node['finish']=node['calls'][-1]['finish']
    previous.validate_review_charge(node)
    return node,captured


def solver_payload(row,tree,parent,teacher):
    seeded_row(row,parent)
    return {'model':protocol.old.MODEL,'prompt':short_repair(row,tree[parent],teacher),'raw':True,
        'stream':False,'logprobs':True,'keep_alive':'30m','options':{'temperature':.65,
        'seed':protocol.seed(row['id'],parent+'/repair'),'num_predict':128,'num_ctx':4096}}


def solver(row,tree,parent,teacher,lookup,used=None):
    used=set() if used is None else used;captured=[]
    payload=solver_payload(row,tree,parent,teacher)
    node=protocol.actor(saved_request(lookup,used,captured),row,payload['prompt'],payload['options']['seed'],128)
    if len(captured)!=1: raise ValueError('One compact final repair only')
    node['latency_seconds']=captured[0]['returned_unix']-captured[0]['started_unix']
    return node,captured


def logical(row,tree,parent,kind,node,raw,teacher=None):
    return {'id':row['id'],'subject':row['subject'],'kind':kind,
        'path':parent if kind=='judge' else parent+'/late-guided','protocol':SCHEMA,
        'seed_identity':seeded_row(row,parent)['id'] if kind=='judge' else row['id'],
        'paid_prefix':{'root_sha256':previous.digest(tree['']),'current_sha256':previous.digest(tree[parent]),
            'late_review_sha256':previous.digest(teacher) if teacher is not None else None},
        'physical_requests':[r['request_sha256'] for r in raw],
        'physical_responses':[r['response_sha256'] for r in raw],'response':node}


def restore(rows,trees,records,raw_records,complete=False):
    if len(rows)!=64 or len({r['id'] for r in rows})!=64 or Counter(r['subject'] for r in rows)!=COUNTS:
        raise ValueError('All64 fixed science questions must remain in the pilot')
    lookup={r['request_sha256']:r for r in raw_records}
    if len(lookup)!=len(raw_records): raise ValueError('Duplicate completed raw request')
    expected,used={},set()
    for row in rows:
        qid=row['id'];tree=trees[qid]
        for parent in PREFIXES:
            try: teacher,raw=review(row,tree,parent,lookup,used)
            except MissingPhysical: continue
            expected[qid,'judge',parent]=logical(row,tree,parent,'judge',teacher,raw)
            try: answer,raw=solver(row,tree,parent,teacher,lookup,used)
            except MissingPhysical: continue
            expected[qid,'solver',parent+'/late-guided']=logical(row,tree,parent,'solver',answer,raw,teacher)
    if set(lookup)!=used: raise ValueError('Unrecognized physical request or unpaid-prefix generation')
    known={}
    for record in records:
        key=record['id'],record['kind'],record['path']
        if key in known or key not in expected or record!=expected[key]:
            raise ValueError('Logical node does not match its exact paid physical replay')
        if key[1]=='solver' and (key[0],'judge',key[2].split('/')[0]) not in known:
            raise ValueError('A logical final solver must follow its durable review receipt')
        known[key]=record
    judges=[r['response'] for r in known.values() if r['kind']=='judge']
    coverage={'questions':64,'logical_reviews':len(judges),'compact_repairs':len(known)-len(judges),
              'physical_review_phases':sum(r['physical_api_calls'] for r in judges),'physical_calls':len(lookup)}
    if complete and (len(known)!=256 or len(expected)!=256 or coverage['logical_reviews']!=128
            or coverage['compact_repairs']!=128 or len(lookup)!=128+coverage['physical_review_phases']):
        raise ValueError('All64×2 review/final-answer cells must be complete before grading')
    return known,coverage


def ordinary_controls(data,rows,plain):
    sealed=io.read(data/'train-ordinary-complete.json');ids={r['id'] for r in rows}
    trees={qid:{p:copy.deepcopy(plain[qid][p]) for p in protocol.PLAIN} for qid in ids};judges={};selected=[]
    files={str((data/'train-ordinary-complete.json').resolve()):io.sha(data/'train-ordinary-complete.json')}
    for role,name in (('base','train-observations.jsonl'),('late','train-late-observations.jsonl')):
        path=data/name
        if io.sha(path)!=sealed[role+'_bank_sha256']: raise ValueError('Original ordinary controls changed')
        files[str(path.resolve())]=io.sha(path)
        for item in protocol.strict_records(path):
            if item['id'] not in ids: continue
            selected.append(item)
            if item['kind']=='judge':
                key=item['id'],item['path']
                if key in judges: raise ValueError('Duplicate ordinary teacher control')
                judges[key]=item['response']
            elif item['kind']=='solver': trees[item['id']][item['path']]=item['response']
            else: raise ValueError('Unknown ordinary control kind')
    by_id={r['id']:r for r in rows}
    for item in selected:
        qid,kind,path=item['id'],item['kind'],item['path']
        bank='plain' if kind=='solver' and path in protocol.PLAIN else 'ordinary'
        protocol.validate_node(by_id[qid],bank,kind,path,item['response'],trees[qid],judges)
    for row in rows:
        qid=row['id']
        if any((qid,p) not in judges or p+'/reviewed' not in trees[qid] for p in PREFIXES):
            raise ValueError('Require both ordinary late-review controls for every question')
    return trees,judges,files


def public_inputs(args):
    rows,trees,root_reviews,chosen,prefix_sha,evidence=previous.public_inputs(args)
    # Reconstruct the full already-sealed768 TRAIN Complete grid directly;
    # no historical DEV prerequisite is reopened and no label is decoded.
    registration=io.read(args.complete/'registration.json');done=io.read(args.complete/'complete.json')
    if (done['schema']!=completion.SCHEMA or done['questions']!=256 or done['training_only'] is not True
            or done['test_used'] is not False or done['scored'] is not False
            or done['registration_sha256']!=io.sha(args.complete/'registration.json')
            or registration['paths']['data']!=str(args.data.resolve())
            or registration['paths']['output']!=str(args.complete.resolve())):
        raise ValueError('Require the complete sealed TRAIN-only Complete grid')
    io.verify(registration['source_files'])
    io.verify({str(args.complete/name):value for name,value in done['digests'].items()})
    all_rows=[r for r in previous.gates.public_train(args.data) if r['subject'] in COUNTS]
    plain,_=previous.original.plain_controls(args.data,{r['id'] for r in all_rows})
    if registration['selection']!=[{'id':r['id'],'subject':r['subject']} for r in all_rows]:
        raise ValueError('The original256 science Complete cohort changed')
    pilot=Path(registration['paths']['pilot']);reused=protocol.strict_records(pilot/'raw-generations.jsonl')
    if len(reused)!=192 or io.sha(pilot/'raw-generations.jsonl')!=registration['raw_reuse']['sha256']:
        raise ValueError('Changed exact inherited Complete physical generations')
    known,coverage=completion.restore(all_rows,plain,{r['request_sha256'] for r in reused},
        protocol.strict_records(args.complete/'train.jsonl'),protocol.strict_records(args.complete/'raw-generations.jsonl'),True)
    if coverage!=done['coverage']: raise ValueError('Full768 Complete physical grid changed')
    for row in rows:
        qid=row['id']
        if any(trees[qid][p]!=plain[qid][p] for p in protocol.PLAIN): raise ValueError('Original compact control changed')
        for p in completion.PREFIXES: trees[qid][completion.action_path(p)]=known[qid,p]['response']
    ordinary,ordinary_reviews,files=ordinary_controls(args.data,rows,trees)
    evidence.update(files);evidence.update({str((args.complete/name).resolve()):value for name,value in done['digests'].items()})
    evidence[str((args.complete/'complete.json').resolve())]=io.sha(args.complete/'complete.json')
    evidence.update(registration['source_files'])
    return rows,trees,root_reviews,ordinary,ordinary_reviews,chosen,prefix_sha,evidence
