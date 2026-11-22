"""Explicit question-only thought, then original-current review; TRAIN pilot.

The two teacher phases always execute, with the original paired phase seeds.
The first phase cannot see any compact solution. The second sees the complete
paid first response and the original root only. No acquisition entrypoint.
"""
from collections import Counter
import hashlib
import json
from types import SimpleNamespace

import strong_late_review_pilot as previous
from long_review_pilot import phase
from review_fields import parse as fields

old,io,protocol=previous.old,previous.io,previous.protocol
SCHEMA='blind-first-root-review-TRAIN-v1'
COUNTS={'chemistry':32,'physics':32}
PATHS=('blind-guided','blind-guided/reuse')
PHASES=('bounded-thought-v1','bounded-answer-v1')
JSON_PREFIX='{"answer": "'
SCREEN={'novel_questions_beyond_registered_full_known_union_at_least':6,
        'mean_second_trajectory_compute_over_rootstrong_retained_at_most':1.25,
        'registered_known_union_questions':48,
        'scope':'TRAIN action headroom only; no controller, causal anchoring-only or fresh-data claim'}
MissingPhysical=previous.spec.MissingPhysical


def public(row):
    if row.get('answer')!='' or row['subject'] not in COUNTS or row['domain']!='science':
        raise ValueError('Only blank-reference science questions may reach the reviewer')
    return row['question'].removesuffix(protocol.old.SCIENCE_SUFFIX)


def reasoning_prompt(row):
    request=('Solve this scientific multiple-choice problem independently from first principles. '
        'Identify the pivotal physical or chemical assumption, check calculations and units when relevant, '
        'and compare plausible options. Explain the decisive distinction and determine the best option. '
        'Use the available reasoning budget to check your conclusion.\n\n'+public(row))
    return '<|im_start|>user\n'+request+'<|im_end|>\n<|im_start|>assistant\n<think>\n'


def paid_reasoning(first):
    return '\n'.join(v for v in (first.get('thinking',''),first['text']) if v)


def decision_prompt(row,current,first):
    request=('Use your paid independent reasoning to give the best answer to the ORIGINAL QUESTION. '
        'Then assess CURRENT SOLUTION. Return ONLY JSON with fields in this order: '
        '"answer" (one offered option letter), "p_correct" (probability CURRENT SOLUTION is correct), '
        '"action" ("stop", "repair", or "retry"), and "check" (at most55 words with the decisive justification). '
        'STOP submits the current answer, REPAIR revises it using your review, and RETRY solves the original question afresh.'
        '\n\nORIGINAL QUESTION:\n'+public(row)+'\n\nYOUR PAID INDEPENDENT REASONING:\n'+paid_reasoning(first)
        +'\n\nCURRENT SOLUTION:\n'+current['text'])
    return '<|im_start|>user\n'+request+'<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n'+JSON_PREFIX


def review(row,tree,lookup,used=None):
    public(row);used=set() if used is None else used;captured=[]
    request=previous.spec.saved_request(lookup,used,captured)
    first=phase(request,row,reasoning_prompt(row),PHASES[0],1024)
    final=phase(request,row,decision_prompt(row,tree[''],first),PHASES[1],512)
    if len(captured)!=2:raise ValueError('Both paid teacher phases are mandatory')
    for call,raw in zip((first,final),captured):
        call['latency_seconds']=raw['returned_unix']-raw['started_unix']
    body=JSON_PREFIX+final['text'];obj,valid,complete=fields(body)
    p,action=obj.get('p_correct',.5),obj.get('action','repair')
    if type(p) not in (int,float) or not 0<=p<=1:p=.5
    if action not in ('stop','repair','retry'):action='repair'
    thought=paid_reasoning(first);calls=[first,final]
    result={'model':protocol.old.JUDGE,'model_digest':protocol.old.JUDGE_DIGEST,'protocol':SCHEMA,
        'text':thought+'\n</think>\n'+body,'prompt_sha256':hashlib.sha256(json.dumps(
            [n['prompt_sha256'] for n in calls]).encode()).hexdigest(),
        'input_tokens':sum(n['input_tokens'] for n in calls),'output_tokens':sum(n['output_tokens'] for n in calls),
        'output_cap':1536,'physical_api_calls':2,'calls':calls,
        'latency_seconds':sum(n['latency_seconds'] for n in calls),'raw_verdict':obj,
        'verdict':{**obj,'solution':thought},'reasoning_source':'Question-only1024; explicit current review512',
        'answer':protocol.old.parse_answer(row['domain'],'Final: '+str(obj.get('answer',''))),
        'p_correct':p,'action':action,'parse_valid':valid,'json_complete':complete,'finish':final['finish']}
    old.validate_review_charge(result)
    return result,captured


def solver_payload(row,current,teacher,path):
    public(row)
    if path not in PATHS:raise ValueError('Only first guided and retained second repair')
    logical='repair' if path==PATHS[0] else 'repair/repair'
    return {'model':protocol.old.MODEL,'prompt':previous.spec.short_repair(row,current,teacher),
        'raw':True,'stream':False,'logprobs':True,'keep_alive':'30m',
        'options':{'temperature':.65,'seed':protocol.seed(row['id'],logical),'num_predict':128,'num_ctx':4096}}


def solver(row,current,teacher,path,lookup,used=None):
    used=set() if used is None else used;captured=[];payload=solver_payload(row,current,teacher,path)
    node=protocol.actor(previous.spec.saved_request(lookup,used,captured),row,payload['prompt'],payload['options']['seed'],128)
    if len(captured)!=1:raise ValueError('One physical compact generation per continuation')
    node['latency_seconds']=captured[0]['returned_unix']-captured[0]['started_unix']
    return node,captured


def logical(row,root,kind,path,node,raw,current=None,teacher=None):
    return {'id':row['id'],'subject':row['subject'],'kind':kind,'path':path,'protocol':SCHEMA,
        'paid_prefix':{'root_sha256':old.digest(root),
            'current_sha256':old.digest(current) if current is not None else None,
            'new_review_sha256':old.digest(teacher) if teacher is not None else None},
        'physical_requests':[r['request_sha256'] for r in raw],
        'physical_responses':[r['response_sha256'] for r in raw],'response':node}


def restore(rows,trees,records,raw_records,complete=False):
    if len(rows)!=64 or len({r['id'] for r in rows})!=64 or Counter(r['subject'] for r in rows)!=COUNTS:
        raise ValueError('Use the exact entire64-question science cohort')
    lookup={r['request_sha256']:r for r in raw_records}
    if len(lookup)!=len(raw_records):raise ValueError('Duplicate physical request')
    expected,used={},set()
    for row in rows:
        qid=row['id'];tree=trees[qid];current=tree['']
        try:teacher,raw=review(row,tree,lookup,used)
        except MissingPhysical:continue
        expected[qid,'judge','']=logical(row,tree[''],'judge','',teacher,raw)
        for path in PATHS:
            try:node,raw=solver(row,current,teacher,path,lookup,used)
            except MissingPhysical:break
            expected[qid,'solver',path]=logical(row,tree[''],'solver',path,node,raw,current,teacher)
            current=node
    if set(lookup)!=used:raise ValueError('Foreign physical generation or unpaid-prefix request')
    known={}
    for record in records:
        key=record['id'],record['kind'],record['path'];qid=key[0]
        if key in known or expected.get(key)!=record:raise ValueError('Logical node mismatches exact physical replay')
        if key[1]=='solver':
            required=(qid,'judge','') if key[2]==PATHS[0] else (qid,'solver',PATHS[0])
            if required not in known:raise ValueError('Continuation lacks durable paid predecessor')
        known[key]=record
    coverage={'questions':64,'logical_reviews':sum(k[1]=='judge' for k in known),
        'first_guided':sum(k[2]==PATHS[0] for k in known),'second_repairs':sum(k[2]==PATHS[1] for k in known),
        'physical_teacher_phases':sum(r['payload']['model']==protocol.old.JUDGE for r in raw_records),
        'physical_compact_calls':sum(r['payload']['model']==protocol.old.MODEL for r in raw_records),
        'physical_calls':len(raw_records)}
    if complete and (len(known)!=192 or len(expected)!=192 or coverage!={
        'questions':64,'logical_reviews':64,'first_guided':64,'second_repairs':64,
        'physical_teacher_phases':128,'physical_compact_calls':128,'physical_calls':256}):
        raise ValueError('All64 review/first/second trajectories and256 physical calls must seal before grading')
    return known,coverage


def public_inputs(args):
    parent=SimpleNamespace(data=args.data,mentor=args.mentor,complete=args.complete,output=args.late)
    rows,trees,reviews,ordinary,ordinary_reviews,late_known,done=previous.validated_complete(parent)
    contract=previous.registration(parent)
    evidence={**contract['input_evidence'],str((args.late/'complete.json').resolve()):io.sha(args.late/'complete.json'),
        **{str((args.late/name).resolve()):value for name,value in done['digests'].items()}}
    return rows,trees,reviews,ordinary,ordinary_reviews,late_known,contract,evidence
