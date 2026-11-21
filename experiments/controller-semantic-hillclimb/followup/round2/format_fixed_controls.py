"""Training-only matched answer-first controls, with no controller or reviewer.

Only the final formatting sentence changes in each original fixed-action
prompt. All128 new mentor-cohort questions and all four continuation branches
are retained, regardless of correctness/validity. Scoring is a separate phase
that requires a sealed, validated512-response grid.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transfer_variants import actor
from transport import resilient
import extended_acquire as old

APP = 'srr-followup-solver-20261120'
PROTOCOL = 'matched-answer-first-fixed-controls-v1'
PATHS = ('repair', 'repair/repair', 'retry', 'retry/retry')
CONTROL_PATHS = ('', *PATHS)
COUNTS = {'math': 64, 'chemistry': 32, 'physics': 32}
FIXED = {'Stop': ('',), 'Repair once': ('', 'repair'),
         'Repair twice': ('', 'repair', 'repair/repair'),
         'Retry once': ('', 'retry'), 'Retry twice': ('', 'retry', 'retry/retry')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def lines(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Incomplete final receipt requires explicit recovery: ' + str(path))
    return [json.loads(line) for line in raw.splitlines()]


def public_rows(data):
    # References are discarded before cohort selection, prompt construction,
    # collection or validation. Only grade() retains reference values.
    return [{**{k: row[k] for k in ('id', 'question', 'domain', 'subject')}, 'answer': ''}
            for row in lines(data / 'splits/train-questions.jsonl')]


def selection(rows, mentor_contract):
    if len(rows) != 512 or len({r['id'] for r in rows}) != 512:
        raise ValueError('Use the entire registered512training pool')
    chosen = [{k: item[k] for k in ('id', 'subject')}
              for item in mentor_contract['selection'] if item['bank'] == 'new']
    if len(chosen) != 128 or len({r['id'] for r in chosen}) != 128 or Counter(r['subject'] for r in chosen) != COUNTS:
        raise ValueError('Use all128new mentor questions, with64/32/32subject counts')
    expected = {(r['id'], subject) for subject, count in COUNTS.items()
                for r in [row for row in rows if row['subject'] == subject][:count]}
    if {(r['id'], r['subject']) for r in chosen} != expected:
        raise ValueError('Mentor cohort differs from the original outcome-blind prefix quotas')
    return chosen


def logical_seed(qid, path):
    if path not in CONTROL_PATHS:
        raise ValueError('Unexpected fixed-action path')
    return int(hashlib.sha256(f'{old.SEED}:{qid}:{path}'.encode()).hexdigest()[:8], 16) % 2**31


def prompt(row, path, tree):
    if path not in PATHS:
        raise ValueError('Only the four registered continuation paths are changed')
    safe = {k: row[k] for k in ('id', 'question', 'domain', 'subject')}
    safe['answer'] = ''
    original = old.solver_prompt(safe, path, tree, {})
    unit = 'one option letter' if safe['domain'] == 'science' else 'a number'
    before = f'Give at most 90 words of scratch work, then a line "Final: X" where X is {unit}.'
    after = f'Write a line "Final: X" FIRST, where X is {unit}, then at most 35 words justifying it.'
    if original.count(before) != 1:
        raise ValueError('Original fixed prompt no longer has the registered format sentence')
    return original.replace(before, after)


def validate_original(row, path, tree):
    node = tree[path]
    expected = hashlib.sha256(old.solver_prompt(row, path, tree, {}).encode()).hexdigest()
    cap = 128 if not path else 256
    if (node['model'], node['model_digest'], node['prompt_sha256'], node['seed'], node['temperature']) != (
            old.MODEL, old.MODEL_DIGEST, expected, logical_seed(row['id'], path), 0 if not path else .65):
        raise ValueError('Original matched fixed response identity changed')
    if node.get('output_cap', cap) != cap or node.get('context_length', 4096) != 4096:
        raise ValueError('Original matched generation budget changed')
    if node['answer'] != old.parse_answer(row['domain'], node['text']):
        raise ValueError('Original parsed answer differs from its saved response')
    if node['input_tokens'] <= 0 or not 0 <= node['output_tokens'] <= cap:
        raise ValueError('Original matched token receipt is invalid')


def controls(data, rows, chosen):
    complete = json.loads((data / 'train-plain-complete.json').read_bytes())
    with (data / 'train-observations.jsonl').open('rb') as stream:
        prefix = b''.join(next(stream) for _ in range(complete['base_records']))
    if hashlib.sha256(prefix).hexdigest() != complete['base_bank_sha256']:
        raise ValueError('Sealed original training response prefix changed')
    by_id = {r['id']: r for r in rows}; trees = {r['id']: {} for r in chosen}
    for line in prefix.splitlines():
        item = json.loads(line)
        if item['id'] in trees and item['kind'] == 'solver' and item['path'] in CONTROL_PATHS:
            if item['path'] in trees[item['id']]:
                raise ValueError('Duplicate original fixed response')
            trees[item['id']][item['path']] = item['response']
    for qid, tree in trees.items():
        if set(tree) != set(CONTROL_PATHS):
            raise ValueError('Incomplete original fixed matched controls')
        for path in CONTROL_PATHS:
            validate_original(by_id[qid], path, tree)
    return trees, {'base_records': complete['base_records'], 'base_bank_sha256': complete['base_bank_sha256'],
                   'matched_controls_sha256': digest(trees)}


def source_files():
    import scoring
    functions = (actor, resilient, old.solver_prompt, old.repair_prompt, old.parse_answer,
                 old.confidence, old.chat, scoring.correct)
    paths = {Path(__file__).resolve(), *[Path(inspect.getfile(f)).resolve() for f in functions]}
    return {str(path): sha(path) for path in sorted(paths)}


def register(args):
    if args.output.exists():
        raise ValueError('Use a new format-control registration directory')
    rows = public_rows(args.data)
    mentor = json.loads((args.mentor / 'contract.json').read_bytes())
    chosen = selection(rows, mentor)
    _, control_receipt = controls(args.data, rows, chosen)
    contract = {'protocol': PROTOCOL, 'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_sha256': source_files(), 'mentor_contract_sha256': sha(args.mentor / 'contract.json'),
                'question_split_sha256': sha(args.data / 'splits/train-questions.jsonl'),
                'selection': chosen, 'subject_counts': COUNTS, 'questions': 128,
                'selection_rule': 'All128new questions in the existing mentor cohort; original64/32/32prefix quotas; no outcome filtering',
                'original_controls': control_receipt, 'continuation_paths': list(PATHS),
                'model': old.MODEL, 'model_digest': old.MODEL_DIGEST,
                'temperature': .65, 'output_cap': 256, 'context_length': 4096,
                'seed_rule': f'SHA256({old.SEED}:question_id:original_logical_path) first8hex modulo2**31',
                'prompt_change': 'Replace only the original final formatting sentence: answer first, then at most35words justification',
                'root': 'Exact original cached root reused; original root temperature0/cap128',
                'no_reviews_or_feedback': True, 'app': APP, 'client_workers': 2, 'max_cloud_gpus': 2,
                'training_only': True, 'test_used': False, 'correctness_used_to_collect_or_select': False,
                'score_after': 'All512new nodes and all raw API receipts validated and hash-sealed',
                'scope': 'Matched fixed-baseline format diagnostic, not an A/B/C result or frozen-benchmark replacement'}
    args.output.mkdir(parents=True)
    (args.output / 'registration.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(json.dumps({'registered': 128, 'planned_new_solver_calls': 512,
                      'registration_sha256': sha(args.output / 'registration.json'), 'inference_started': False}), flush=True)
    return contract


def registered_view(args):
    contract = json.loads((args.output / 'registration.json').read_bytes())
    if contract['source_sha256'] != source_files():
        raise ValueError('Registered format-control sources changed')
    if contract['question_split_sha256'] != sha(args.data / 'splits/train-questions.jsonl') or contract['mentor_contract_sha256'] != sha(args.mentor / 'contract.json'):
        raise ValueError('Registered training pool or cohort contract changed')
    rows = public_rows(args.data)
    chosen = selection(rows, json.loads((args.mentor / 'contract.json').read_bytes()))
    original, receipt = controls(args.data, rows, chosen)
    if chosen != contract['selection'] or receipt != contract['original_controls']:
        raise ValueError('Registered matched cohort or original controls changed')
    if (contract['protocol'], contract['continuation_paths'], contract['model_digest'], contract['temperature'],
            contract['output_cap'], contract['context_length'], contract['client_workers']) != (
                PROTOCOL, list(PATHS), old.MODEL_DIGEST, .65, 256, 4096, 2):
        raise ValueError('Registered format-control protocol changed')
    return contract, {r['id']: r for r in rows}, original


def index_records(records, chosen):
    allowed = {(r['id'], p) for r in chosen for p in PATHS}
    indexed = {}
    for record in records:
        key = record['id'], record['path']
        if key not in allowed or key in indexed:
            raise ValueError('Duplicate or unregistered fixed-control receipt')
        indexed[key] = record
    return indexed


def request_identity(qid, path, text, registration_sha):
    return {'id': qid, 'path': path, 'registration_sha256': registration_sha,
            'model': old.MODEL, 'model_digest': old.MODEL_DIGEST,
            'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'seed': logical_seed(qid, path),
            'temperature': .65, 'output_cap': 256, 'context_length': 4096}


def replay_node(row, path, text, raw, registration_sha):
    expected = request_identity(row['id'], path, text, registration_sha)
    if raw['request'] != expected:
        raise ValueError('Raw saved call identity differs; never regenerate it silently')
    def replay(endpoint, payload):
        if endpoint != '/api/generate' or payload['prompt'] != text or payload['options'] != {
                'temperature': .65, 'seed': logical_seed(row['id'], path), 'num_predict': 256, 'num_ctx': 4096}:
            raise ValueError('Unexpected compact actor payload')
        return raw['result']
    node = actor(replay, row, text, logical_seed(row['id'], path), 256)
    node['protocol'] = PROTOCOL
    node['latency_seconds'] = raw['latency_seconds']
    if node['input_tokens'] <= 0 or not 0 <= node['output_tokens'] <= 256:
        raise ValueError('Invalid physical solver-token accounting')
    return node


def validate_node(saved, replayed, parent_sha, registration_sha):
    if saved['registration_sha256'] != registration_sha or saved['parent_response_sha256'] != parent_sha:
        raise ValueError('Saved derived response belongs to another registered paid prefix')
    left = {k: v for k, v in saved['response'].items() if k != 'latency_seconds'}
    right = {k: v for k, v in replayed.items() if k != 'latency_seconds'}
    if left != right:
        raise ValueError('Saved response differs from its raw completed API receipt')


def validate_grid(contract, rows, original, saved, raw, registration_sha, require_complete=True):
    count = 0
    for item in contract['selection']:
        qid = item['id']; tree = {'': original[qid]['']}
        for path in PATHS:
            key = qid, path
            if key not in saved:
                if require_complete:
                    raise ValueError('Do not score an incomplete fixed-control grid')
                continue
            if key not in raw:
                raise ValueError('Derived response lacks its physical API receipt')
            parent = path.rsplit('/', 1)[0] if '/' in path else ''
            if path.endswith('repair') and parent not in tree:
                raise ValueError('Repair continuation lacks its paid parent')
            text = prompt(rows[qid], path, tree)
            rebuilt = replay_node(rows[qid], path, text, raw[key], registration_sha)
            parent_sha = digest(tree[parent]) if path.endswith('repair') else None
            validate_node(saved[key], rebuilt, parent_sha, registration_sha)
            tree[path] = saved[key]['response']; count += 1
    if require_complete and (count != 512 or len(raw) != 512 or len(saved) != 512):
        raise ValueError('Require all128questions times four continuations before scoring')


def _collect_locked(args, request=None):
    contract, rows, original = registered_view(args)
    registration_sha = sha(args.output / 'registration.json')
    saved = index_records(lines(args.output / 'train.jsonl'), contract['selection'])
    raw = index_records(lines(args.output / 'raw-calls.jsonl'), contract['selection'])
    validate_grid(contract, rows, original, saved, raw, registration_sha, require_complete=False)
    lock = threading.Lock(); worker_holder = {}; done = 0

    def remote(endpoint, payload):
        if request is not None:
            return request(endpoint, payload)
        with lock:
            if 'request' not in worker_holder:
                import modal
                worker_holder['request'] = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
                tags = {r['name']: r['digest'] for r in worker_holder['request']('/api/tags')['models']}
                if tags.get(old.MODEL) != old.MODEL_DIGEST:
                    raise ValueError('Deployed compact solver weights changed')
        return worker_holder['request'](endpoint, payload)

    with (args.output / 'raw-calls.jsonl').open('a') as raw_stream, (args.output / 'train.jsonl').open('a') as stream:
        def append(target, record):
            target.write(json.dumps(record) + '\n'); target.flush(); os.fsync(target.fileno())

        def one(item):
            nonlocal done
            qid = item['id']; row = rows[qid]; tree = {'': original[qid]['']}
            for path in PATHS:
                key = qid, path; text = prompt(row, path, tree)
                expected = request_identity(qid, path, text, registration_sha)
                if key not in raw:
                    def capture(endpoint, payload):
                        started = time.monotonic()
                        result = remote(endpoint, payload)
                        record = {'id': qid, 'path': path, 'request': expected, 'result': result,
                                  'latency_seconds': time.monotonic() - started}
                        # Save every returned result before parsing/validity checks.
                        # A completed wrong/invalid answer never triggers a retry.
                        with lock:
                            append(raw_stream, record); raw[key] = record
                        return result
                    actor(capture, row, text, logical_seed(qid, path), 256)
                rebuilt = replay_node(row, path, text, raw[key], registration_sha)
                parent = path.rsplit('/', 1)[0] if '/' in path else ''
                parent_sha = digest(tree[parent]) if path.endswith('repair') else None
                if key not in saved:
                    record = {'id': qid, 'path': path, 'registration_sha256': registration_sha,
                              'parent_response_sha256': parent_sha, 'response': rebuilt}
                    with lock:
                        append(stream, record); saved[key] = record
                validate_node(saved[key], rebuilt, parent_sha, registration_sha)
                tree[path] = saved[key]['response']
            with lock:
                done += 1
                if done % 8 == 0 or done == 128:
                    print(f'Format-only fixed training controls {done}/128; no accuracy scored', flush=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, contract['selection']))
    validate_grid(contract, rows, original, saved, raw, registration_sha)
    complete = {'protocol': PROTOCOL, 'questions': 128, 'solver_nodes': 512, 'physical_returned_api_calls': 512,
                'registration_sha256': registration_sha, 'bank_sha256': sha(args.output / 'train.jsonl'),
                'raw_calls_sha256': sha(args.output / 'raw-calls.jsonl'), 'training_only': True, 'scored': False}
    target = args.output / 'complete.json'
    if target.exists() and json.loads(target.read_bytes()) != complete:
        raise ValueError('Completed fixed-control bank changed')
    target.write_text(json.dumps(complete, indent=2) + '\n')
    print(json.dumps(complete), flush=True)
    return complete


def collect(args, request=None):
    # Release the collection lock on every failure, including an unusable raw
    # result that must be inspected rather than regenerated on resumption.
    with (args.output / 'collection.lock').open('a') as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _collect_locked(args, request)


def metric(rows, trees, paths, correct):
    inputs = []; outputs = []; correctness = []
    for row in rows:
        nodes = [trees[row['id']][p] for p in paths]
        inputs.append(sum(n['input_tokens'] for n in nodes)); outputs.append(sum(n['output_tokens'] for n in nodes))
        correctness.append(bool(correct(nodes[-1], row['answer'])))
    n = len(rows)
    return {'n': n, 'correct': sum(correctness), 'accuracy': sum(correctness) / n,
            'mean_input_tokens': sum(inputs) / n, 'mean_output_tokens': sum(outputs) / n,
            'mean_total_tokens': (sum(inputs) + sum(outputs)) / n,
            'mean_compute': 4 * (sum(inputs) + sum(outputs)) / (3.09 * n),
            'solver_calls_per_question': len(paths), 'review_calls': 0, 'encoder_calls': 0}, correctness


def grade(args):
    # Complete/hash guards precede the first retained reference-answer read.
    complete = json.loads((args.output / 'complete.json').read_bytes())
    if (complete['solver_nodes'], complete['questions'], complete['physical_returned_api_calls']) != (512, 128, 512):
        raise ValueError('All registered fixed controls must complete before scoring')
    for field, filename in (('registration_sha256', 'registration.json'), ('bank_sha256', 'train.jsonl'),
                            ('raw_calls_sha256', 'raw-calls.jsonl')):
        if complete[field] != sha(args.output / filename):
            raise ValueError('Sealed fixed-control input changed before grading')
    contract, public, original = registered_view(args)
    saved = index_records(lines(args.output / 'train.jsonl'), contract['selection'])
    raw = index_records(lines(args.output / 'raw-calls.jsonl'), contract['selection'])
    validate_grid(contract, public, original, saved, raw, complete['registration_sha256'])
    by_id = {r['id']: r for r in lines(args.data / 'splits/train-questions.jsonl')}
    rows = [by_id[r['id']] for r in contract['selection']]
    from scoring import correct
    formatted = {r['id']: {'': original[r['id']][''], **{p: saved[r['id'], p]['response'] for p in PATHS}} for r in rows}
    methods, paired, diagnostics, groups = {}, {}, {}, {}
    for label, group in [('all', rows), *[(s, [r for r in rows if r['subject'] == s]) for s in COUNTS]]:
        group_methods, group_pairs = {}, {}
        for name, paths in FIXED.items():
            before, a = metric(group, original, paths, correct)
            after, b = metric(group, formatted, paths, correct)
            group_methods['Original ' + name] = before; group_methods['Answer-first ' + name] = after
            group_pairs[name] = {'n': len(group), 'wins': sum(not x and y for x, y in zip(a, b)),
                                 'losses': sum(x and not y for x, y in zip(a, b)),
                                 'both_correct': sum(x and y for x, y in zip(a, b)),
                                 'both_wrong': sum(not x and not y for x, y in zip(a, b)),
                                 'accuracy_change_percentage_points': 100 * (after['accuracy'] - before['accuracy']),
                                 'mean_compute_change': after['mean_compute'] - before['mean_compute']}
        if label == 'all':
            methods, paired = group_methods, group_pairs
        else:
            groups[label] = {'methods': group_methods, 'paired': group_pairs}
        diagnostics[label] = {}
        for variant, bank in (('original', original), ('answer_first', formatted)):
            diagnostics[label][variant] = {}
            for path in CONTROL_PATHS:
                nodes = [bank[r['id']][path] for r in group]; cap = 128 if not path else 256
                diagnostics[label][variant][path or 'root'] = {'n': len(nodes),
                    'valid_answers': sum(bool(n['answer']) for n in nodes),
                    'at_token_cap': sum(n['output_tokens'] >= cap for n in nodes),
                    'finish_length': sum(n['finish'] == 'length' for n in nodes),
                    'input_tokens': sum(n['input_tokens'] for n in nodes),
                    'output_tokens': sum(n['output_tokens'] for n in nodes)}
    physical = [r['response'] for r in saved.values()]
    total_input = sum(n['input_tokens'] for n in physical); total_output = sum(n['output_tokens'] for n in physical)
    result = {'scope': contract['scope'], 'training_questions': 128, 'fresh_data_used': False,
              'methods': methods, 'paired_format_vs_original': paired, 'by_subject': groups, 'diagnostics': diagnostics,
              'acquisition': {'new_solver_calls': 512, 'new_review_calls': 0, 'new_encoder_calls': 0,
                  'input_tokens': total_input, 'output_tokens': total_output,
                  'compute': 4 * (total_input + total_output) / 3.09,
                  'original_root_and_controls_reused': True,
                  'transport_attempts_without_usable_receipt': len(lines(args.output / 'transport-errors.jsonl')),
                  'unknown_transport_compute_included': False},
              'compute_unit': '4B parameter-weighted input+output tokens divided by3.09B; policy costs include each paid root/prefix',
              'complete_sha256': sha(args.output / 'complete.json'), 'test_used': False,
              'interpretation': 'Formatting diagnostic only; not a learned-controller result and not a change to frozen benchmark baselines'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'collect', 'grade'))
    for name in ('data', 'mentor', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    {'register': register, 'collect': collect, 'grade': grade}[args.phase](args)
