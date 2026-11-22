"""TRAIN-only option audit, counterexample check, and bounded review decision.

The frozen reviewer spends640+384+512 output tokens in three paid phases.
Every original option is retained in its original order and lettering. Only
the first phase is blind to the already-paid root solution. Both compact
continuations use the unchanged retained-review template and seeds.
"""
from collections import Counter
import hashlib
import json

import blind_first_review_protocol as original
import long_budget_review_protocol as inputs
from long_review_pilot import phase
from review_fields import parse as fields

previous, old, io, protocol = original.previous, original.old, original.io, original.protocol
SCHEMA = 'contrastive-three-phase-root-review-TRAIN-v1'
COUNTS = {'chemistry': 32, 'physics': 32}
PATHS = ('contrastive-v1-guided', 'contrastive-v1-guided/reuse')
PHASES = ('contrastive-option-audit-v1', 'contrastive-counterexample-v1', 'contrastive-decision-v1')
CAPS = (640, 384, 512)
JSON_PREFIX = '{"answer": "'
EXTRA_BANKS = (*inputs.EXTRA_BANKS, 'long4096-root-review-pilot-v1',
    'grounded1024-root-review-pilot-v2', 'advanced-grounded1024-root-review-pilot-v1',
    'arithmetic1024-root-review-pilot-v1', 'example1024-root-review-pilot-v1',
    'active-textbook1024-root-review-pilot-v1', 'active-textbook1024-root-review-pilot-v2',
    'evidence-first-textbook-root-review-pilot-v3', 'broad-evidence-first-textbook-root-review-pilot-v4')
SCREEN = {'novel_questions_beyond_registered_known_union_at_least': 6,
          'mean_second_trajectory_compute_over_rootstrong_retained_at_most': 1.25,
          'scope': 'TRAIN answer availability only; requires learned POMDP integration, strongest matched B and fresh validation'}
public, paid_reasoning, MissingPhysical = original.public, original.paid_reasoning, original.MissingPhysical


def reasoning_prompt(row):
    request = ('Audit the options in this scientific question. Preserve the original meaning and lettering. '
        'First identify the governing scientific constraint. Then give one concise discriminating test '
        'for each offered option: a decisive supporting fact, contradiction, unit check, limiting case, '
        'or violated assumption. Compare the two strongest surviving options explicitly. '
        'Keep the audit concise enough to reach a provisional best option within the budget. '
        'Do not assume that an option is true merely because it sounds familiar.\n\nQUESTION:\n'+public(row))
    return '<|im_start|>user\n'+request+'<|im_end|>\n<|im_start|>assistant\n<think>\n'


def counterexample_prompt(row, current, first):
    request = ('Check the following paid option audit for its most consequential error. '
        'Focus on the two strongest alternatives. Seek a concrete counterexample, missing assumption, '
        'incorrect physical or chemical rule, arithmetic error, or mismatch with the actual wording. '
        'Check whether CURRENT SOLUTION reveals a valid competing argument. '
        'Preserve a conclusion if the counterargument fails; avoid changing it merely to disagree. '
        'State the decisive check and your resulting best original option letter.\n\nQUESTION:\n'+public(row)
        +'\n\nPAID OPTION AUDIT:\n'+paid_reasoning(first)
        +'\n\nCURRENT SOLUTION:\n'+current['text'])
    return '<|im_start|>user\n'+request+'<|im_end|>\n<|im_start|>assistant\n<think>\n'


def decision_prompt(row, current, first, second):
    request = ('Use both paid analyses to decide the best answer to the ORIGINAL QUESTION. '
        'Resolve their substantive disagreement using the question and scientific constraints. '
        'Return ONLY JSON fields in this order: "answer" (one original offered option letter), '
        '"p_correct" (probability CURRENT SOLUTION is correct), "action" ("stop", "repair", or "retry"), '
        '"check" (at most55 words of decisive justification). STOP submits the current answer; '
        'REPAIR revises using this review; RETRY solves the original question afresh.\n\nQUESTION:\n'+public(row)
        +'\n\nCURRENT SOLUTION:\n'+current['text']
        +'\n\nPAID OPTION AUDIT:\n'+paid_reasoning(first)
        +'\n\nPAID COUNTEREXAMPLE CHECK:\n'+paid_reasoning(second))
    return '<|im_start|>user\n'+request+'<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n'+JSON_PREFIX


def validate_review_charge(judge):
    calls = judge.get('calls', [])
    if (judge.get('physical_api_calls') != 3 or len(calls) != 3
            or tuple(c['phase'] for c in calls) != PHASES
            or tuple(c['output_cap'] for c in calls) != CAPS or judge['output_cap'] != sum(CAPS)):
        raise ValueError('All three declared contrastive phases are mandatory')
    for name in ('input_tokens', 'output_tokens'):
        if judge[name] != sum(c[name] for c in calls):
            raise ValueError('Omitted paid review phase')
    if any(not 0 <= c['output_tokens'] <= c['output_cap'] for c in calls):
        raise ValueError('Exceeded a registered phase budget')


def review(row, tree, lookup, used=None):
    public(row)
    used = set() if used is None else used
    captured = []
    request = previous.spec.saved_request(lookup, used, captured)
    first = phase(request, row, reasoning_prompt(row), PHASES[0], CAPS[0])
    second = phase(request, row, counterexample_prompt(row, tree[''], first), PHASES[1], CAPS[1])
    final = phase(request, row, decision_prompt(row, tree[''], first, second), PHASES[2], CAPS[2])
    if len(captured) != 3:
        raise ValueError('Three physical reviewer phases required')
    calls = [first, second, final]
    for call, raw in zip(calls, captured):
        call['latency_seconds'] = raw['returned_unix']-raw['started_unix']
    body = JSON_PREFIX+final['text']
    obj, valid, complete = fields(body)
    p, action = obj.get('p_correct', .5), obj.get('action', 'repair')
    if type(p) not in (int, float) or not 0 <= p <= 1: p = .5
    if action not in ('stop', 'repair', 'retry'): action = 'repair'
    thought = paid_reasoning(first)+'\nPAID COUNTEREXAMPLE CHECK:\n'+paid_reasoning(second)
    result = {'model': protocol.old.JUDGE, 'model_digest': protocol.old.JUDGE_DIGEST, 'protocol': SCHEMA,
        'text': thought+'\n</think>\n'+body,
        'prompt_sha256': hashlib.sha256(json.dumps([v['prompt_sha256'] for v in calls]).encode()).hexdigest(),
        'input_tokens': sum(v['input_tokens'] for v in calls), 'output_tokens': sum(v['output_tokens'] for v in calls),
        'output_cap': sum(CAPS), 'physical_api_calls': 3, 'calls': calls,
        'latency_seconds': sum(v['latency_seconds'] for v in calls), 'raw_verdict': obj,
        'verdict': {**obj, 'solution': thought}, 'reasoning_source': 'Paid640 option audit +384 counterexample check +512 decision',
        'answer': protocol.old.parse_answer(row['domain'], 'Final: '+str(obj.get('answer', ''))),
        'p_correct': p, 'action': action, 'parse_valid': valid, 'json_complete': complete, 'finish': final['finish']}
    validate_review_charge(result)
    return result, captured


def solver_payload(row, current, teacher, path):
    public(row)
    if path not in PATHS: raise ValueError('Only the two registered compact continuations')
    logical_path = 'repair' if path == PATHS[0] else 'repair/repair'
    return {'model': protocol.old.MODEL, 'prompt': previous.spec.short_repair(row, current, teacher),
        'raw': True, 'stream': False, 'logprobs': True, 'keep_alive': '30m',
        'options': {'temperature': .65, 'seed': protocol.seed(row['id'], logical_path), 'num_predict': 128, 'num_ctx': 4096}}


def solver(row, current, teacher, path, lookup, used=None):
    used = set() if used is None else used
    captured = []
    payload = solver_payload(row, current, teacher, path)
    node = protocol.actor(previous.spec.saved_request(lookup, used, captured), row,
                          payload['prompt'], payload['options']['seed'], 128)
    if len(captured) != 1: raise ValueError('One paid compact call required')
    node['latency_seconds'] = captured[0]['returned_unix']-captured[0]['started_unix']
    return node, captured


def logical(row, root, kind, path, node, raw, current=None, teacher=None):
    return {'id': row['id'], 'subject': row['subject'], 'kind': kind, 'path': path, 'protocol': SCHEMA,
        'paid_prefix': {'root_sha256': old.digest(root), 'current_sha256': old.digest(current) if current is not None else None,
                        'new_review_sha256': old.digest(teacher) if teacher is not None else None},
        'physical_requests': [r['request_sha256'] for r in raw],
        'physical_responses': [r['response_sha256'] for r in raw], 'response': node}


def restore(rows, trees, records, raw_records, complete=False):
    if len(rows) != 64 or len({r['id'] for r in rows}) != 64 or Counter(r['subject'] for r in rows) != COUNTS:
        raise ValueError('The entire fixed64-question training cohort is required')
    lookup = {r['request_sha256']: r for r in raw_records}
    if len(lookup) != len(raw_records): raise ValueError('Repeated physical request')
    expected, used = {}, set()
    for row in rows:
        tree = trees[row['id']]
        current = tree['']
        try: teacher, raw = review(row, tree, lookup, used)
        except MissingPhysical: continue
        expected[row['id'], 'judge', ''] = logical(row, tree[''], 'judge', '', teacher, raw)
        for path in PATHS:
            try: node, raw = solver(row, current, teacher, path, lookup, used)
            except MissingPhysical: break
            expected[row['id'], 'solver', path] = logical(row, tree[''], 'solver', path, node, raw, current, teacher)
            current = node
    if set(lookup) != used: raise ValueError('Foreign physical call or unpaid future prefix')
    known = {}
    for record in records:
        key = record['id'], record['kind'], record['path']
        if key in known or expected.get(key) != record: raise ValueError('Logical/physical replay mismatch')
        if key[1] == 'solver':
            predecessor = (key[0], 'judge', '') if key[2] == PATHS[0] else (key[0], 'solver', PATHS[0])
            if predecessor not in known: raise ValueError('Missing durable paid predecessor')
        known[key] = record
    coverage = {'questions': 64, 'logical_reviews': sum(k[1] == 'judge' for k in known),
        'first_guided': sum(k[2] == PATHS[0] for k in known), 'second_repairs': sum(k[2] == PATHS[1] for k in known),
        'physical_teacher_phases': sum(r['payload']['model'] == protocol.old.JUDGE for r in raw_records),
        'physical_compact_calls': sum(r['payload']['model'] == protocol.old.MODEL for r in raw_records), 'physical_calls': len(raw_records)}
    if complete and (len(known) != 192 or coverage != {'questions': 64, 'logical_reviews': 64, 'first_guided': 64,
            'second_repairs': 64, 'physical_teacher_phases': 192, 'physical_compact_calls': 128, 'physical_calls': 320}):
        raise ValueError('Seal all320 physical calls and192 logical nodes before grading')
    return known, coverage


def public_inputs(args):
    *bundle, evidence = inputs.public_inputs(args)
    for name in EXTRA_BANKS:
        folder = args.data.parent/name
        if not (folder/'train.jsonl').is_file():
            raise ValueError('Missing previous training path bank: '+name)
        for file in ('train.jsonl', 'complete.json', 'collection-complete.json', 'registration.json', 'contract.json', 'summary.json'):
            path = folder/file
            if path.exists(): evidence[str(path.resolve())] = io.sha(path)
    return *bundle, evidence


def extra_solver_answers(args, rows):
    result = {r['id']: [] for r in rows}
    for name in EXTRA_BANKS:
        for record in protocol.strict_records(args.data.parent/name/'train.jsonl'):
            if record['id'] not in result or record.get('kind', 'solver') != 'solver': continue
            node = record['response']
            if node.get('model_digest') != protocol.old.MODEL_DIGEST: raise ValueError('Changed prior compact solver')
            result[record['id']].append(node['answer'])
    return result


def context(*_):
    # The generic collector records no retrieved context for this protocol.
    return ''
