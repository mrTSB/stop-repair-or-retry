"""Same root review with4096 thinking tokens; unchanged compact solver/prompt.

This isolates the bounded reasoning dose against the retained1024 protocol.
All physical phases are paid, and no correct/incorrect completion is resampled.
"""
from collections import Counter
import hashlib
import json
from types import SimpleNamespace

import strong_late_review_pilot as previous
from long_review_pilot import review as bounded_review

old,io,protocol=previous.old,previous.io,previous.protocol
SCHEMA='long4096-root-review-TRAIN-v1'
COUNTS={'chemistry':32,'physics':32}
PATHS=('long4096-guided','long4096-guided/reuse')
PHASES=('bounded-thought-v1','bounded-answer-v1')
SCREEN={'novel_questions_beyond_registered_known_union_at_least':6,
        'mean_second_trajectory_compute_over_rootstrong_retained_at_most':2.5,
        'scope':'TRAIN headroom of longer reasoning; requires selective cost control, matched B, and fresh validation'}
EXTRA_BANKS=('blind-first-root-review-pilot-v1','independent-review-pilot-v1',
             'second-review-pilot-v1','option-proposal-pilot-v1',
             'drop-review-standard-pilot-v1','finish-once-pilot-v1')
MissingPhysical=previous.spec.MissingPhysical


def public(row):
    if row.get('answer')!='' or row['subject'] not in COUNTS or row['domain']!='science':
        raise ValueError('Only blank-reference science questions may reach the reviewer')
    return row['question'].removesuffix(protocol.old.SCIENCE_SUFFIX)


def review(row,tree,lookup,used=None):
    public(row);used=set() if used is None else used;captured=[]
    request=previous.spec.saved_request(lookup,used,captured)
    result=bounded_review(request,row,tree[''],4096,512)
    if len(captured)!=len(result['calls']) or len(captured) not in (1,2):
        raise ValueError('One or two bounded teacher phases required')
    for call,raw in zip(result['calls'],captured):
        call['latency_seconds']=raw['returned_unix']-raw['started_unix']
    result.update(protocol=SCHEMA,latency_seconds=sum(c['latency_seconds'] for c in result['calls']))
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
    expected_phases=sum(len(v['response']['calls']) for k,v in known.items() if k[1]=='judge')
    if complete and (len(known)!=192 or len(expected)!=192 or coverage['questions']!=64
        or coverage['logical_reviews']!=64 or coverage['first_guided']!=64 or coverage['second_repairs']!=64
        or coverage['physical_compact_calls']!=128 or not 64<=expected_phases<=128
        or coverage['physical_teacher_phases']!=expected_phases or coverage['physical_calls']!=128+expected_phases):
        raise ValueError('All64 complete trajectories and every physical phase required')
    return known,coverage


def public_inputs(args):
    parent=SimpleNamespace(data=args.data,mentor=args.mentor,complete=args.complete,output=args.late)
    rows,trees,reviews,ordinary,ordinary_reviews,late_known,done=previous.validated_complete(parent)
    contract=previous.registration(parent)
    evidence={**contract['input_evidence'],str((args.late/'complete.json').resolve()):io.sha(args.late/'complete.json'),
        **{str((args.late/name).resolve()):value for name,value in done['digests'].items()}}
    for name in EXTRA_BANKS:
        folder=args.data.parent/name;bank=folder/'train.jsonl'
        if not bank.is_file():raise ValueError('Required prior availability bank is absent')
        evidence[str(bank.resolve())]=io.sha(bank)
        for filename in ('complete.json','registration.json','contract.json','summary.json'):
            path=folder/filename
            if path.exists():evidence[str(path.resolve())]=io.sha(path)
    return rows,trees,reviews,ordinary,ordinary_reviews,late_known,contract,evidence


def extra_solver_answers(args,rows):
    result={r['id']:[] for r in rows}
    for name in EXTRA_BANKS:
        for record in protocol.strict_records(args.data.parent/name/'train.jsonl'):
            if record['id'] not in result or record.get('kind','solver')!='solver':continue
            node=record['response']
            if node.get('model_digest')!=protocol.old.MODEL_DIGEST:raise ValueError('Prior availability used a different solver')
            result[record['id']].append(node['answer'])
    return result
