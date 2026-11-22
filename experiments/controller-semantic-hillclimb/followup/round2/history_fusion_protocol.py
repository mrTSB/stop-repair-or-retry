"""Explicit new terminal-repair context using only the paid trajectory history.

After guided repair or independent retry, the same frozen compact solver sees
the root, current answer, original reviewer check, and a fixed160-word suffix
of its paid reasoning. This is a new repair protocol, not a change to any of
the five ordinary baselines. Matched B must get the same capability.
"""
import copy
import hashlib
import json
import math

import contrastive_review_protocol as source

old,io,protocol,previous = source.old,source.io,source.protocol,source.previous
SCHEMA = 'paid-history-fusion-terminal-repair-TRAIN-v1'
PREFIXES = ('guided','retry')
PATHS = {p:p+'/history-fusion' for p in PREFIXES}
COUNTS = source.COUNTS
EXTRA_BANKS = (*source.EXTRA_BANKS,'contrastive-three-phase-root-review-pilot-v1','dense-grounded1024-root-review-pilot-v1')
SCREEN = {'novel_correct_questions_beyond_complete_prior_union_at_least':6,
          'maximum_endpoint_mean_compute_over_rootstrong_retained_at_most':1.25,
          'scope':'TRAIN action availability only; new repair protocol requires matched B, learned POMDP integration and fresh validation'}


def public_inputs(args):
    *bundle,evidence = source.public_inputs(args)
    for name in EXTRA_BANKS:
        folder = args.data.parent/name
        for filename in ('train.jsonl','complete.json','graded/summary.json'):
            path = folder/filename
            if path.exists(): evidence[str(path.resolve())] = io.sha(path)
    latest = args.data.parent/'dense-grounded1024-root-review-pilot-v1'
    done = io.read(latest/'complete.json')
    io.verify({str(latest/k):v for k,v in done['digests'].items()})
    if io.read(latest/'graded/summary.json')['reports']['all_science']['combined_hindsight_union'] != 50:
        raise ValueError('Explicitly update novelty contract if the preceding path union changed')
    return *bundle,evidence


def prompt(row,tree,review,parent):
    question = source.public(row)
    if parent not in PREFIXES: raise ValueError('Exactly the two already-paid first continuations are allowed')
    old.validate_review_charge(review)
    first = review['calls'][0]
    reasoning = '\n'.join(str(first.get(k,'')) for k in ('thinking','text'))
    excerpt = ' '.join(reasoning.split()[-160:])
    packet = {'proposed_answer':review['answer'],'check':str(review['verdict'].get('check','')),
              'paid_reasoning_last160_words':excerpt}
    request = ('Make a final repair using the ORIGINAL QUESTION and the paid history below. '
        'The initial solution, current solution and reviewer can each be wrong; agreement is not proof. '
        'Compare their scientific arguments against the actual question, governing constraints and units. '
        'Keep a supported answer or choose another offered option when the arguments warrant it. '
        'Write "Final: X" FIRST, where X is one original option letter, then at most35 words of decisive justification.'
        '\n\nORIGINAL QUESTION:\n'+question+'\n\nPAID INITIAL SOLUTION:\n'+tree['']['text']
        +'\n\nPAID CURRENT SOLUTION:\n'+tree[parent]['text']
        +'\n\nPAID REVIEW (fallible; excerpt truncated by a fixed word rule):\n'+json.dumps(packet,sort_keys=True))
    return protocol.old.chat(request)


def payload(row,tree,review,parent):
    seed_path = 'repair/repair' if parent == 'guided' else 'retry/repair'
    return {'model':protocol.old.MODEL,'prompt':prompt(row,tree,review,parent),'raw':True,
        'stream':False,'logprobs':True,'keep_alive':'30m','options':{'temperature':.65,
        'seed':protocol.seed(row['id'],seed_path),'num_predict':128,'num_ctx':4096}}


def paid_prefix(tree,review,parent):
    return {'root_sha256':old.digest(tree['']),'review_sha256':old.digest(review),
            'first_continuation':parent,'first_continuation_sha256':old.digest(tree[parent])}


def replay_node(row,tree,review,parent,raw):
    old.RequestJournal.validate(raw)
    expected = payload(row,tree,review,parent)
    if raw['endpoint'] != '/api/generate' or raw['payload'] != expected:
        raise ValueError('Different model, prompt, seed or paid history')
    response = raw['response']
    if (response.get('model') != protocol.old.MODEL or response.get('done') is not True
            or type(response.get('prompt_eval_count')) is not int or not 0 < response['prompt_eval_count'] <= 4096
            or type(response.get('eval_count')) is not int or not 0 <= response['eval_count'] <= 128):
        raise ValueError('Invalid physical compact token receipt; preserve the response')
    elapsed = raw['returned_unix']-raw['started_unix']
    if not math.isfinite(elapsed) or elapsed < 0: raise ValueError('Invalid physical elapsed time')
    def request(endpoint,actual):
        if endpoint != '/api/generate' or actual != expected: raise ValueError('Actor requested another protocol')
        return copy.deepcopy(response)
    node = protocol.actor(request,row,expected['prompt'],expected['options']['seed'],128)
    node['latency_seconds'] = elapsed
    return node


def logical(row,tree,review,parent,raw):
    return {'id':row['id'],'subject':row['subject'],'kind':'solver','path':PATHS[parent],'parent':parent,
        'protocol':SCHEMA,'paid_prefix':paid_prefix(tree,review,parent),
        'request_sha256':raw['request_sha256'],'raw_response_sha256':raw['response_sha256'],
        'response':replay_node(row,tree,review,parent,raw)}


def restore(rows,trees,reviews,records,raw_records,complete=False):
    if len(rows) != 64 or len({r['id'] for r in rows}) != 64:
        raise ValueError('Use the full fixed64-question training cohort')
    by_id = {r['id']:r for r in rows}
    expected = {}
    for row in rows:
        for parent in PREFIXES:
            key = old.digest({'endpoint':'/api/generate','payload':payload(row,trees[row['id']],reviews[row['id'],''],parent)})
            if key in expected: raise ValueError('Duplicate expected request')
            expected[key] = row['id'],parent
    raw = {}
    for item in raw_records:
        key = item['request_sha256']
        if key not in expected or key in raw: raise ValueError('Foreign or duplicate physical generation')
        qid,parent = expected[key]
        replay_node(by_id[qid],trees[qid],reviews[qid,''],parent,item)
        raw[key] = item
    known = {}
    for record in records:
        pair = record['id'],record['parent']
        key = record['request_sha256']
        if pair in known or key not in raw or expected[key] != pair:
            raise ValueError('Missing physical receipt or repeated logical endpoint')
        qid,parent = pair
        if record != logical(by_id[qid],trees[qid],reviews[qid,''],parent,raw[key]):
            raise ValueError('Logical endpoint differs from its physical request and paid prefix')
        known[pair] = record
    if complete and (len(known) != 128 or set(raw) != set(expected)):
        raise ValueError('Seal all128 physical and logical endpoints before grading')
    return known,{'questions':64,'physical_calls':len(raw),'logical_endpoints':len(known)}


def extra_solver_answers(args,rows):
    result = source.extra_solver_answers(args,rows)
    for name in ('contrastive-three-phase-root-review-pilot-v1','dense-grounded1024-root-review-pilot-v1'):
        for record in protocol.strict_records(args.data.parent/name/'train.jsonl'):
            if record['id'] in result and record['kind'] == 'solver':
                if record['response']['model_digest'] != protocol.old.MODEL_DIGEST: raise ValueError('Changed compact weights')
                result[record['id']].append(record['response']['answer'])
    return result
