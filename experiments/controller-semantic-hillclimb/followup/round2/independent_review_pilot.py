"""Training-only independent second review on all64 registered science pairs.

Phase one sees the question alone. Phase two sees that same question, its own
paid reasoning, and only the current final answer. Both phases always execute.
The already paid root review and guided compact call remain trajectory costs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

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
import second_review_pilot as critical

APP = 'srr-followup-solver-20261120'
PROTOCOL = 'independent-second-review-v1'
THOUGHT_PHASE = 'independent-question-thought-v1'
ANSWER_PHASE = 'independent-current-answer-v1'
THOUGHT_CAP, ANSWER_CAP = 2048, 256
JSON_PREFIX = '{"answer": "'


def read(path):
    return json.loads(Path(path).read_bytes())


def question(row):
    if row['domain'] != 'science':
        raise ValueError('This registered pilot is science only')
    return row['question'].removesuffix(old.SCIENCE_SUFFIX)


def reasoning_prompt(row):
    request = (
        'Solve this scientific multiple-choice problem independently from first principles. '
        'Identify the pivotal physical or chemical assumption, check the calculation and units when relevant, '
        'and compare the plausible answer options. Explain the decisive distinction and determine the best option. '
        'Use the available reasoning budget to check your conclusion.\n\n' + question(row))
    return '<|im_start|>user\n' + request + '<|im_end|>\n<|im_start|>assistant\n<think>\n'


def paid_reasoning(first):
    # Preserve every returned reasoning/text segment rather than selecting a
    # fragment based on the proposed answer or any correctness judgement.
    return '\n'.join(x for x in (first.get('thinking', ''), first['text']) if x)


def decision_prompt(row, current, first):
    request = (
        'Use your paid independent reasoning below to give the best answer to the original question. '
        'Then assess whether CURRENT FINAL ANSWER is correct. Return ONLY JSON with fields in this order: '
        '"answer" (one option letter), "p_correct" (probability CURRENT FINAL ANSWER is correct), '
        '"action" ("stop", "repair", or "retry"), and "check" (at most 55 words with the decisive justification). '
        'STOP submits CURRENT FINAL ANSWER, REPAIR revises the current solution using your review, '
        'and RETRY solves the original question afresh.\n\nORIGINAL QUESTION:\n' + question(row)
        + '\n\nYOUR PAID INDEPENDENT REASONING:\n' + paid_reasoning(first)
        + '\n\nCURRENT FINAL ANSWER:\n' + current['answer'])
    return '<|im_start|>user\n' + request + '<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n' + JSON_PREFIX


def validate_phase(row, prompt, name, cap, node):
    seed = int(hashlib.sha256(f'{old.SEED}:{row["id"]}:{name}'.encode()).hexdigest()[:8], 16) % 2**31
    if (node['phase'], node['prompt_sha256'], node['seed'], node['output_cap'], node['context_length'], node['temperature']) != (
            name, hashlib.sha256(prompt.encode()).hexdigest(), seed, cap, 8192, 0):
        raise ValueError('Changed independent-review physical phase')
    if (type(node['input_tokens']) is not int or node['input_tokens'] <= 0 or
            type(node['output_tokens']) is not int or not 0 <= node['output_tokens'] <= cap):
        raise ValueError('Invalid independent-review phase token receipt')


def review(request, row, current, saved_phases=None, save_phase=None):
    saved_phases = {} if saved_phases is None else saved_phases

    def acquire(prompt, name, cap):
        node = saved_phases.get(name)
        if node is None:
            node = phase(request, row, prompt, name, cap)
            # Preserve even a completed phase with a bad cap/count receipt;
            # validation then fails closed instead of silently regenerating it.
            if save_phase is not None:
                save_phase(name, node)
            validate_phase(row, prompt, name, cap, node)
        else:
            validate_phase(row, prompt, name, cap, node)
        return node

    first = acquire(reasoning_prompt(row), THOUGHT_PHASE, THOUGHT_CAP)
    # Unconditional second phase: no answer, parse status, or reference can
    # suppress the registered decision call.
    final = acquire(decision_prompt(row, current, first), ANSWER_PHASE, ANSWER_CAP)
    body = JSON_PREFIX + final['text']; obj, valid, complete = fields(body)
    probability, action = obj.get('p_correct', .5), obj.get('action', 'repair')
    if type(probability) not in (int, float) or not 0 <= probability <= 1:
        probability = .5
    if action not in ('stop', 'repair', 'retry'):
        action = 'repair'
    answer = old.parse_answer(row['domain'], 'Final: ' + str(obj.get('answer', '')))
    calls = [first, final]; thought = paid_reasoning(first)
    return {'model': old.JUDGE, 'model_digest': old.JUDGE_DIGEST, 'protocol': PROTOCOL,
            'text': thought + '\n</think>\n' + body,
            'prompt_sha256': hashlib.sha256(json.dumps([r['prompt_sha256'] for r in calls]).encode()).hexdigest(),
            'input_tokens': sum(r['input_tokens'] for r in calls), 'output_tokens': sum(r['output_tokens'] for r in calls),
            'output_cap': THOUGHT_CAP + ANSWER_CAP, 'physical_api_calls': 2, 'calls': calls,
            'latency_seconds': sum(r['latency_seconds'] for r in calls), 'raw_verdict': obj,
            'verdict': {**obj, 'solution': thought}, 'reasoning_source': 'Question-only independent2048token scratchpad',
            'answer': answer, 'p_correct': probability, 'action': action, 'parse_valid': valid,
            'json_complete': complete, 'finish': final['finish']}


def verify_review(row, current, node):
    if (node['protocol'], node['model_digest'], node['physical_api_calls']) != (PROTOCOL, old.JUDGE_DIGEST, 2):
        raise ValueError('Changed independent-review protocol or phase count')
    if [n['phase'] for n in node['calls']] != [THOUGHT_PHASE, ANSWER_PHASE]:
        raise ValueError('Independent review must retain its two ordered physical phases')
    def forbidden(*args, **kwargs):
        raise AssertionError('Receipt verification attempted inference')
    actual = review(forbidden, row, current, {n['phase']: n for n in node['calls']})
    for key in ('text', 'answer', 'raw_verdict', 'verdict', 'input_tokens', 'output_tokens', 'prompt_sha256',
                'output_cap', 'parse_valid', 'json_complete', 'p_correct', 'action', 'finish'):
        if actual[key] != node[key]:
            raise ValueError('Independent-review receipt replay differs: ' + key)


def verify_solver(row, current, teacher, node):
    prompt = short_repair(row, current, teacher)
    if (node['prompt_sha256'], node['seed'], node['output_cap'], node['model_digest'], node['temperature'], node['context_length']) != (
            hashlib.sha256(prompt.encode()).hexdigest(), mentor_pairs.seed_for(row['id'], 'guided/reuse'),
            128, old.MODEL_DIGEST, .65, 4096):
        raise ValueError('Changed bounded compact second-repair control')
    if type(node['input_tokens']) is not int or node['input_tokens'] <= 0 or type(node['output_tokens']) is not int or not 0 <= node['output_tokens'] <= 128:
        raise ValueError('Invalid compact second-repair token receipt')


def controls(args):
    parent, parent_done = read(args.mentor / 'contract.json'), read(args.mentor / 'complete.json')
    old_contract, old_summary = read(args.critical / 'contract.json'), read(args.critical / 'summary.json')
    old_prefix = read(args.critical / 'prefix-receipt.json')
    chosen = critical.selection(parent)
    if (parent_done['bank_sha256'], parent_done['contract_sha256']) != (sha(args.mentor / 'train.jsonl'), sha(args.mentor / 'contract.json')):
        raise ValueError('Original mentor controls changed')
    if (old_summary['questions'], old_summary['test_used'], old_summary['bank_sha256'], old_summary['contract_sha256'],
            old_summary['prefix_receipt_sha256']) != (64, False, sha(args.critical / 'train.jsonl'), sha(args.critical / 'contract.json'),
                                                    sha(args.critical / 'prefix-receipt.json')):
        raise ValueError('Finish and preserve the entire critical-review pilot first')
    if (old_contract['selection'], old_contract['source_sha256'], old_contract['parent_contract_sha256'],
            old_contract['training_question_sha256']) != (chosen, sha(critical.__file__), sha(args.mentor / 'contract.json'),
                                                         sha(args.data / 'splits/train-questions.jsonl')):
        raise ValueError('Independent review must use the exact registered science cohort')
    if old_prefix['parent_bank_sha256'] != sha(args.mentor / 'train.jsonl') or old_contract['test_used']:
        raise ValueError('Critical controls used another parent bank or split')
    rows, trees = plain_data(args.data, 'train'); by_id = {r['id']: r for r in rows}
    strong_records, critical_records = committed(args.mentor / 'train.jsonl'), committed(args.critical / 'train.jsonl')
    strong = {(r['id'], r['kind'], r['path']): r['response'] for r in strong_records}
    earlier = {(r['id'], r['kind']): r['response'] for r in critical_records}
    if len(strong) != len(strong_records) or len(earlier) != 128 or len(critical_records) != 128:
        raise ValueError('Incomplete or duplicate paired controls')
    selected, hashes = [], {}
    for record in chosen:
        row = by_id[record['id']]; qid = row['id']; safe = {**row, 'answer': ''}
        root = strong[qid, 'judge', '']; current = strong[qid, 'solver', 'guided']; retained = strong[qid, 'solver', 'guided/reuse']
        mentor_pairs.verify_review(safe, trees[qid][''], root, 1024, 512)
        prompt = short_repair(safe, trees[qid][''], root)
        if (current['prompt_sha256'], current['seed'], current['output_cap'], current['model_digest']) != (
                hashlib.sha256(prompt.encode()).hexdigest(), mentor_pairs.seed_for(qid, 'guided'), 128, old.MODEL_DIGEST):
            raise ValueError('Changed paid first guided response')
        if (current['temperature'], current['context_length']) != (.65, 4096) or not 0 <= current['output_tokens'] <= 128:
            raise ValueError('Changed first guided generation settings or token cap')
        verify_solver(safe, current, root, retained)
        critical.verify_review(safe, root, current, earlier[qid, 'judge'])
        verify_solver(safe, current, earlier[qid, 'judge'], earlier[qid, 'solver'])
        values = {'root_solver': trees[qid][''], 'root_review': root, 'first_guided': current, 'retained_repair': retained,
                  'critical_review': earlier[qid, 'judge'], 'critical_repair': earlier[qid, 'solver']}
        hashes[qid] = {k + '_sha256': fingerprint(v) for k, v in values.items()}
        selected.append(row)
    return chosen, selected, trees, strong, earlier, hashes


def register(args, chosen, hashes):
    contract = {'source_sha256': sha(__file__), 'parent_contract_sha256': sha(args.mentor / 'contract.json'),
                'parent_bank_sha256': sha(args.mentor / 'train.jsonl'), 'critical_contract_sha256': sha(args.critical / 'contract.json'),
                'critical_summary_sha256': sha(args.critical / 'summary.json'), 'critical_bank_sha256': sha(args.critical / 'train.jsonl'),
                'training_question_sha256': sha(args.data / 'splits/train-questions.jsonl'),
                'selection': chosen, 'controls': hashes, 'questions': 64, 'subjects': {'chemistry': 32, 'physics': 32},
                'selection_rule': 'All64science questions of the existing critical pilot; no outcome filtering',
                'phase_source_sha256': sha(sys.modules[phase.__module__].__file__),
                'short_repair_source_sha256': sha(sys.modules[short_repair.__module__].__file__),
                'actor_source_sha256': sha(sys.modules[actor.__module__].__file__),
                'solver_digest': old.MODEL_DIGEST, 'reviewer_digest': old.JUDGE_DIGEST, 'protocol': PROTOCOL,
                'thinking_cap': THOUGHT_CAP, 'answer_cap': ANSWER_CAP, 'always_two_phases': True,
                'review_phase_names': [THOUGHT_PHASE, ANSWER_PHASE], 'context_length': 8192, 'review_temperature': 0,
                'first_phase_inputs': ['original question'],
                'second_phase_inputs': ['original question', 'own paid first-phase reasoning', 'current parsed final answer only'],
                'compact_prompt': 'Unchanged short_repair with current full solution and newly paid review answer/check',
                'compact_cap': 128, 'compact_temperature': .65, 'compact_context_length': 4096, 'compact_seed_path': 'repair/repair',
                'maximum_additional_compact_calls': 2, 'logical_reviews_per_treatment': 2,
                'all_trajectory_and_physical_phase_costs_charged': True, 'matched_B_required_if_promoted': True,
                'app': APP, 'maximum_cloud_gpus': 2, 'split': 'training only', 'test_used': False,
                'fitted_controller': False, 'reference_answers_supplied': False}
    args.output.mkdir(parents=True, exist_ok=True); path = args.output / 'contract.json'
    if path.exists() and read(path) != contract:
        raise ValueError('Changed independent-review registration; use a new output directory')
    path.write_text(json.dumps(contract, indent=2) + '\n')
    return contract


def aggregate(rows, trees, strong, earlier, known):
    treatments, comparison = {}, {}
    for name in ('first_guided', 'retained_repair', 'critical_repair', 'independent_repair'):
        solver_input = solver_output = review_input = review_output = physical = student = teacher = 0
        for row in rows:
            qid = row['id']; root_review = strong[qid, 'judge', '']
            solvers = [trees[qid][''], strong[qid, 'solver', 'guided']]; reviews = [root_review]
            if name == 'retained_repair':
                solvers.append(strong[qid, 'solver', 'guided/reuse'])
            elif name in ('critical_repair', 'independent_repair'):
                source = earlier if name == 'critical_repair' else known
                solvers.append(source[qid, 'solver']); reviews.append(source[qid, 'judge'])
            student += int(correct(solvers[-1], row['answer'])); teacher += int(correct(reviews[-1], row['answer']))
            solver_input += sum(n['input_tokens'] for n in solvers); solver_output += sum(n['output_tokens'] for n in solvers)
            review_input += sum(n['input_tokens'] for n in reviews); review_output += sum(n['output_tokens'] for n in reviews)
            physical += sum(n['physical_api_calls'] for n in reviews)
        n = len(rows); solver_tokens, review_tokens = solver_input + solver_output, review_input + review_output
        treatments[name] = {'n': n, 'student_correct': student, 'student_accuracy': student / n,
                            'last_teacher_correct': teacher, 'last_teacher_accuracy': teacher / n,
                            'mean_solver_input_tokens': solver_input / n, 'mean_solver_output_tokens': solver_output / n,
                            'mean_review_input_tokens': review_input / n, 'mean_review_output_tokens': review_output / n,
                            'mean_solver_tokens': solver_tokens / n, 'mean_review_tokens': review_tokens / n,
                            'mean_solver_calls': 2 if name == 'first_guided' else 3,
                            'mean_logical_reviews': 2 if name in ('critical_repair', 'independent_repair') else 1,
                            'mean_physical_review_calls': physical / n,
                            'mean_compute': (4 * solver_tokens + 3 * review_tokens) / (3.09 * n),
                            'mean_total_parameter_compute': (4 * solver_tokens + 35 * review_tokens) / (3.09 * n)}
    for name in ('first_guided', 'retained_repair', 'critical_repair'):
        wins = losses = 0
        for row in rows:
            qid = row['id']
            old_node = (strong[qid, 'solver', 'guided'] if name == 'first_guided' else
                        strong[qid, 'solver', 'guided/reuse'] if name == 'retained_repair' else earlier[qid, 'solver'])
            before, after = correct(old_node, row['answer']), correct(known[qid, 'solver'], row['answer'])
            wins += bool(after and not before); losses += bool(before and not after)
        comparison[name] = {'student_wins': wins, 'student_losses': losses}
    return {'methods': treatments, 'independent_vs_controls': comparison}


def run(args):
    chosen, rows, trees, strong, earlier, hashes = controls(args)
    register(args, chosen, hashes)
    if args.prepare_only:
        print(json.dumps({'questions': 64, 'contract_sha256': sha(args.output / 'contract.json'), 'new_inference_calls': 0}), flush=True)
        return
    bank, phase_bank = args.output / 'train.jsonl', args.output / 'physical-phases.jsonl'
    lock = threading.Lock(); done = 0
    with bank.open('a') as stream, phase_bank.open('a') as phase_stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(phase_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Lock before taking either resume snapshot. Otherwise a collector
        # that just finished could leave this process with stale saved calls.
        for path in (bank, phase_bank):
            if path.stat().st_size and not path.read_bytes().endswith(b'\n'):
                raise ValueError('Preserve and repair the partial final receipt before resuming')
        raw, raw_phases = committed(bank), committed(phase_bank)
        known = {(r['id'], r['kind']): r['response'] for r in raw}
        phases = {(r['id'], r['phase']): r['response'] for r in raw_phases}
        expected = {(r['id'], k) for r in rows for k in ('judge', 'solver')}
        expected_phases = {(r['id'], p) for r in rows for p in (THOUGHT_PHASE, ANSWER_PHASE)}
        if (len(known) != len(raw) or len(phases) != len(raw_phases) or set(known) - expected or set(phases) - expected_phases or
                any(r['path'] != ('guided' if r['kind'] == 'judge' else 'guided/independent-repair') for r in raw)):
            raise ValueError('Unexpected or duplicate independent pilot receipt')
        for row in rows:
            qid = row['id']
            if ((qid, ANSWER_PHASE) in phases and (qid, THOUGHT_PHASE) not in phases or
                    (qid, 'judge') in known and any((qid, p) not in phases for p in (THOUGHT_PHASE, ANSWER_PHASE)) or
                    (qid, 'solver') in known and (qid, 'judge') not in known):
                raise ValueError('Incomplete paid prefix in the independent-review resume ledger')
        import modal
        request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        if tags.get(old.MODEL) != old.MODEL_DIGEST or tags.get(old.JUDGE) != old.JUDGE_DIGEST:
            raise ValueError('Changed compact solver or reviewer')

        def add(qid, kind, function):
            if (qid, kind) not in known:
                node = function()
                with lock:
                    stream.write(json.dumps({'id': qid, 'kind': kind, 'path': 'guided' if kind == 'judge' else 'guided/independent-repair',
                                             'response': node}) + '\n'); stream.flush()
                    known[qid, kind] = node
            return known[qid, kind]

        def one(row):
            nonlocal done
            qid = row['id']; safe = {**row, 'answer': ''}; current = strong[qid, 'solver', 'guided']
            def save(name, node):
                with lock:
                    if (qid, name) in phases:
                        raise ValueError('Repeated completed physical phase')
                    phase_stream.write(json.dumps({'id': qid, 'phase': name, 'response': node}) + '\n'); phase_stream.flush()
                    os.fsync(phase_stream.fileno())
                    phases[qid, name] = node
            saved = {name: phases[qid, name] for name in (THOUGHT_PHASE, ANSWER_PHASE) if (qid, name) in phases}
            teacher = add(qid, 'judge', lambda: review(request, safe, current, saved, save))
            verify_review(safe, current, teacher)
            if any(phases.get((qid, call['phase'])) != call for call in teacher['calls']):
                raise ValueError('Logical review disagrees with the physical phase ledger')
            node = add(qid, 'solver', lambda: actor(request, safe, short_repair(safe, current, teacher), mentor_pairs.seed_for(qid, 'guided/reuse'), 128))
            verify_solver(safe, current, teacher, node)
            with lock:
                done += 1
                if done % 8 == 0 or done == 64:
                    print(f'Independent second-review training pairs {done}/64', flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, rows))
    if set(known) != expected or set(phases) != expected_phases:
        raise ValueError('Complete all64pairs and all128physical phases before grading')
    result = {'split': 'training only', 'questions': 64, 'test_used': False, 'fitted_controller': False,
              'contract_sha256': sha(args.output / 'contract.json'), 'bank_sha256': sha(bank),
              'physical_phases_sha256': sha(phase_bank), 'physical_review_phases': 128,
              'all': aggregate(rows, trees, strong, earlier, known),
              'subjects': {s: aggregate([r for r in rows if r['subject'] == s], trees, strong, earlier, known)
                           for s in ('chemistry', 'physics')},
              'interpretation': 'All registered training pairs retained. New independent information protocol; both prior paid calls and every physical phase charged. Not a fitted controller or fresh-data result. Matched LLM-only B required before any C-specific claim.'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'mentor', 'critical', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    run(p.parse_args())
