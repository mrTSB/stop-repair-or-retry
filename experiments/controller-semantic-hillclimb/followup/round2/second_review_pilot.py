"""Registered training-only pilot of an independent second critical review.

All 64 new science questions from the existing mentor cohort are retained,
regardless of either review's correctness. This is paired action-data
collection, not a fitted POMDP or a fresh-data accuracy claim.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import committed
from long_review_pilot import phase
from review_fields import parse as fields
from reasoning_pilot import short_repair
from transfer_variants import actor
from transport import resilient
from scoring import correct
from prepare_mixed_mentor import plain_data, fingerprint, sha
import extended_acquire as old
import mentor_pairs

APP = 'srr-followup-solver-20261120'
PROTOCOL = 'second-critical-review-v1'
THOUGHT_PHASE = 'second-critical-thought-v1'
ANSWER_PHASE = 'second-critical-answer-v1'


def selection(contract):
    selected = [r for r in contract['selection'] if r['bank'] == 'new' and r['subject'] in ('chemistry', 'physics')]
    if len({r['id'] for r in selected}) != 64 or Counter(r['subject'] for r in selected) != {'chemistry': 32, 'physics': 32}:
        raise ValueError('Use all64 new science questions from the registered mentor cohort')
    return selected


def critical_prompt(row, root_review, current):
    # Only these three paid/public inputs enter the prompt. No reference,
    # unpurchased branch, or correctness-derived decision is consulted.
    request = (
        'Independently re-check the original scientific problem from first principles. '
        'Identify the pivotal assumption or calculation, and check whether it actually applies. '
        'Avoid anchoring on the previous review or CURRENT answer: both may repeat the same error. '
        'Consider the strongest alternative option and explain the decisive distinction. '
        'Keep your reasoning under 220 words. Then assess the CURRENT final answer. '
        'After reasoning return ONLY JSON with fields in this order: '
        '"answer" (one option letter), "p_correct" (probability CURRENT answer is correct), '
        '"action" ("stop", "repair", or "retry"), '
        '"check" (at most 55 words with the decisive justification). '
        'STOP submits CURRENT, REPAIR revises CURRENT using your review, RETRY solves afresh.\n\n'
        + row['question'].removesuffix(old.SCIENCE_SUFFIX)
        + '\n\nPREVIOUS PAID REVIEW (fallible):\n' + root_review['text']
        + '\n\nCURRENT compact solution:\n' + current['text'])
    return '<|im_start|>user\n' + request + '<|im_end|>\n<|im_start|>assistant\n<think>\n'


def review(request, row, root_review, current, think_cap=1024, answer_cap=512):
    """Same bounded two-phase mechanism, with a distinct prompt and phase seeds."""
    original = critical_prompt(row, root_review, current)
    first = phase(request, row, original, THOUGHT_PHASE, think_cap)
    thought, end, body = first['text'].rpartition('</think>')
    if not end:
        thought = first['thinking'] or first['text']
        body = first['text'] if first['text'].lstrip().startswith('{') else ''
    obj, valid, complete = fields(body)
    calls = [first]
    if not (valid and complete and obj.get('answer')):
        if end:
            thought = first['text'].rsplit('</think>', 1)[0]
        prefix = '{"answer": "'
        final_prompt = original + thought + '\n</think>\n\n' + prefix
        final = phase(request, row, final_prompt, ANSWER_PHASE, answer_cap)
        calls.append(final); body = prefix + final['text']
        obj, valid, complete = fields(body)
    probability, action = obj.get('p_correct', .5), obj.get('action', 'repair')
    if type(probability) not in (int, float) or not 0 <= probability <= 1:
        probability = .5
    if action not in ('stop', 'repair', 'retry'):
        action = 'repair'
    answer = old.parse_answer(row['domain'], 'Final: ' + str(obj.get('answer', '')))
    return {'model': old.JUDGE, 'model_digest': old.JUDGE_DIGEST, 'protocol': PROTOCOL,
            'text': thought + '\n</think>\n' + body,
            'prompt_sha256': hashlib.sha256(json.dumps([r['prompt_sha256'] for r in calls]).encode()).hexdigest(),
            'input_tokens': sum(r['input_tokens'] for r in calls), 'output_tokens': sum(r['output_tokens'] for r in calls),
            'output_cap': sum(r['output_cap'] for r in calls), 'physical_api_calls': len(calls), 'calls': calls,
            'latency_seconds': sum(r['latency_seconds'] for r in calls), 'raw_verdict': obj,
            'verdict': {**obj, 'solution': thought}, 'reasoning_source': 'observed bounded critical scratchpad',
            'answer': answer, 'p_correct': probability, 'action': action, 'parse_valid': valid,
            'json_complete': complete, 'finish': calls[-1]['finish']}


def verify_review(row, root_review, current, node):
    if node['protocol'] != PROTOCOL or node['model_digest'] != old.JUDGE_DIGEST:
        raise ValueError('Changed critical-review protocol or model')
    remaining = iter(node['calls']); used = []

    def replay(endpoint, payload):
        call = next(remaining)
        expected = (hashlib.sha256(payload['prompt'].encode()).hexdigest(),
                    payload['options']['seed'], payload['options']['num_predict'],
                    payload['options']['num_ctx'], payload['options']['temperature'])
        actual = (call['prompt_sha256'], call['seed'], call['output_cap'], call['context_length'], call['temperature'])
        if endpoint != '/api/generate' or expected != actual or not 0 <= call['output_tokens'] <= call['output_cap']:
            raise ValueError('Changed critical-review physical phase')
        used.append(call)
        return {'done': True, 'model': old.JUDGE, 'response': call['text'], 'thinking': call.get('thinking', ''),
                'prompt_eval_count': call['input_tokens'], 'eval_count': call['output_tokens'],
                'done_reason': call['finish'], 'srr_backend': call.get('backend')}

    actual = review(replay, row, root_review, current)
    if len(used) != len(node['calls']) or len(used) != node['physical_api_calls']:
        raise ValueError('Unaccounted critical-review phase')
    for key in ('text', 'answer', 'raw_verdict', 'verdict', 'input_tokens', 'output_tokens',
                'prompt_sha256', 'output_cap', 'parse_valid', 'json_complete', 'p_correct', 'action'):
        if actual[key] != node[key]:
            raise ValueError('Critical-review receipt replay differs: ' + key)


def register(args):
    parent = json.loads((args.mentor / 'contract.json').read_bytes())
    chosen = selection(parent)
    contract = {'source_sha256': sha(__file__), 'parent_contract_sha256': sha(args.mentor / 'contract.json'),
                'selection': chosen, 'questions': 64, 'subjects': {'chemistry': 32, 'physics': 32},
                'selection_rule': 'Every new science question in the already registered strong mentor cohort; no outcome filtering.',
                'training_question_sha256': sha(args.data / 'splits/train-questions.jsonl'),
                'phase_source_sha256': sha(sys.modules[phase.__module__].__file__),
                'short_repair_source_sha256': sha(sys.modules[short_repair.__module__].__file__),
                'solver_digest': old.MODEL_DIGEST, 'reviewer_digest': old.JUDGE_DIGEST,
                'protocol': PROTOCOL, 'think_cap': 1024, 'answer_cap': 512, 'solver_cap': 128,
                'review_context_length': 8192, 'solver_context_length': 4096,
                'review_temperature': 0., 'solver_temperature': .65,
                'review_seed_phases': [THOUGHT_PHASE, ANSWER_PHASE],
                'solver_seed_path': 'repair/repair', 'maximum_additional_compact_calls': 2,
                'logical_reviews_per_completed_treatment': 2, 'all_physical_phases_charged': True,
                'inference_inputs': ['original question', 'already paid root review', 'current guided compact response'],
                'app': APP, 'maximum_cloud_gpus': 2, 'split': 'training only', 'test_used': False,
                'fitted_controller': False, 'reference_answers_supplied': False}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'contract.json'
    if path.exists() and json.loads(path.read_bytes()) != contract:
        raise ValueError('Changed critical-review registration; use a new output directory')
    path.write_text(json.dumps(contract, indent=2) + '\n')
    return chosen, contract


def run(args):
    chosen, contract = register(args)
    if args.prepare_only:
        print(json.dumps({'questions': 64, 'contract_sha256': sha(args.output / 'contract.json'),
                          'new_inference_calls': 0, 'test_used': False}), flush=True)
        return
    while not (args.mentor / 'complete.json').exists():
        if not args.wait:
            raise ValueError('Finish every original strong mentor pair or use --wait')
        time.sleep(30)
    completion = json.loads((args.mentor / 'complete.json').read_bytes())
    if (completion['bank_sha256'], completion['contract_sha256']) != (
            sha(args.mentor / 'train.jsonl'), contract['parent_contract_sha256']):
        raise ValueError('Original mentor responses changed')
    all_rows, trees = plain_data(args.data, 'train')
    by_id = {r['id']: r for r in all_rows}
    rows = [by_id[r['id']] for r in chosen]
    parent = {(r['id'], r['kind'], r['path']): r['response'] for r in committed(args.mentor / 'train.jsonl')}
    controls = {}
    for row in rows:
        qid = row['id']; safe = {**row, 'answer': ''}
        root = parent[qid, 'judge', '']; guided = parent[qid, 'solver', 'guided']
        mentor_pairs.verify_review(safe, trees[qid][''], root, 1024, 512)
        prompt = short_repair(safe, trees[qid][''], root)
        if (guided['prompt_sha256'], guided['seed'], guided['output_cap'], guided['model_digest']) != (
                hashlib.sha256(prompt.encode()).hexdigest(), mentor_pairs.seed_for(qid, 'guided'), 128, old.MODEL_DIGEST):
            raise ValueError('Original guided control differs from its paid prefix')
        controls[qid] = {'root_solver_sha256': fingerprint(trees[qid]['']),
                         'root_review_sha256': fingerprint(root), 'guided_solver_sha256': fingerprint(guided)}
    receipt = {'parent_bank_sha256': sha(args.mentor / 'train.jsonl'), 'contract_sha256': sha(args.output / 'contract.json'),
               'controls': controls, 'questions': 64, 'references_supplied': False, 'test_used': False}
    path = args.output / 'prefix-receipt.json'
    if path.exists() and json.loads(path.read_bytes()) != receipt:
        raise ValueError('Changed paid prefix for critical-review pilot')
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    import modal
    request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
    tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
    if tags.get(old.MODEL) != old.MODEL_DIGEST or tags.get(old.JUDGE) != old.JUDGE_DIGEST:
        raise ValueError('Changed solver or reviewer')
    bank = args.output / 'train.jsonl'
    if bank.exists() and bank.stat().st_size and not bank.read_bytes().endswith(b'\n'):
        raise ValueError('Preserve and repair the partial final receipt before resuming')
    known = {(r['id'], r['kind']): r['response'] for r in committed(bank)}
    if len(known) != len(committed(bank)) or set(known) - {(r['id'], k) for r in rows for k in ('judge', 'solver')}:
        raise ValueError('Duplicate or unexpected critical-review receipt')
    lock = threading.Lock(); done = 0
    with bank.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def add(qid, kind, function):
            key = qid, kind
            if key not in known:
                node = function()
                with lock:
                    stream.write(json.dumps({'id': qid, 'kind': kind, 'path': 'guided' if kind == 'judge' else 'guided/critical-repair',
                                             'response': node}) + '\n')
                    stream.flush(); known[key] = node
            return known[key]

        def one(row):
            nonlocal done
            qid = row['id']; safe = {**row, 'answer': ''}
            root, current = parent[qid, 'judge', ''], parent[qid, 'solver', 'guided']
            critical = add(qid, 'judge', lambda: review(request, safe, root, current))
            verify_review(safe, root, current, critical)
            prompt = short_repair(safe, current, critical)
            seed = mentor_pairs.seed_for(qid, 'guided/reuse')
            node = add(qid, 'solver', lambda: actor(request, safe, prompt, seed, 128))
            if (node['prompt_sha256'], node['seed'], node['output_cap'], node['model_digest'], node['temperature'], node['context_length']) != (
                    hashlib.sha256(prompt.encode()).hexdigest(), seed, 128, old.MODEL_DIGEST, .65, 4096):
                raise ValueError('Critical guided response differs from its paid prompt')
            with lock:
                done += 1
                if done % 8 == 0 or done == 64:
                    print(f'Critical second-review training pairs {done}/64', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, rows))
    if len(known) != 128:
        raise ValueError('Do not summarize incomplete paired treatment outcomes')

    def aggregate(group):
        summary = {}
        for treatment in (False, True):
            teacher_correct = student_correct = solver_tokens = judge_tokens = physical_calls = 0
            for row in group:
                qid = row['id']
                reviews = [parent[qid, 'judge', '']]
                solvers = [trees[qid][''], parent[qid, 'solver', 'guided']]
                if treatment:
                    reviews.append(known[qid, 'judge']); solvers.append(known[qid, 'solver'])
                teacher_correct += int(correct(reviews[-1], row['answer']))
                student_correct += int(correct(solvers[-1], row['answer']))
                solver_tokens += sum(n['input_tokens'] + n['output_tokens'] for n in solvers)
                judge_tokens += sum(n['input_tokens'] + n['output_tokens'] for n in reviews)
                physical_calls += sum(n['physical_api_calls'] for n in reviews)
            n = len(group)
            summary['second_review_guided' if treatment else 'root_review_guided'] = {
                'n': n, 'student_correct': student_correct, 'student_accuracy': student_correct / n,
                'last_teacher_correct': teacher_correct, 'last_teacher_accuracy': teacher_correct / n,
                'mean_solver_tokens': solver_tokens / n, 'mean_review_tokens': judge_tokens / n,
                'mean_solver_calls': 3 if treatment else 2, 'mean_logical_reviews': 2 if treatment else 1,
                'mean_physical_review_calls': physical_calls / n,
                'mean_compute': (4 * solver_tokens + 3 * judge_tokens) / (3.09 * n),
                'mean_total_parameter_compute': (4 * solver_tokens + 35 * judge_tokens) / (3.09 * n)}
        wins = losses = 0
        for row in group:
            qid = row['id']
            before = correct(parent[qid, 'solver', 'guided'], row['answer'])
            after = correct(known[qid, 'solver'], row['answer'])
            wins += bool(after and not before); losses += bool(before and not after)
        return {'methods': summary, 'student_wins': wins, 'student_losses': losses}

    result = {'split': 'training only', 'questions': 64, 'test_used': False, 'fitted_controller': False,
              'contract_sha256': sha(args.output / 'contract.json'), 'prefix_receipt_sha256': sha(path),
              'bank_sha256': sha(bank), 'all': aggregate(rows),
              'subjects': {s: aggregate([r for r in rows if r['subject'] == s]) for s in ('chemistry', 'physics')},
              'interpretation': 'Matched fixed-cohort action pilot; all outcomes retained. Added review information and its full cost are charged. No POMDP or fresh-performance claim.'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'mentor', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--wait', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    run(p.parse_args())
