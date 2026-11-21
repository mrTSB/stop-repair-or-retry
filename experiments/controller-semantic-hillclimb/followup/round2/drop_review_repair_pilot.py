"""TRAIN-only standard repair after a paid strong guided answer.

Exactly64 prespecified science questions, no new review or encoder call. The
second repair receives the current answer but no retained mentor feedback.
This is an action-availability diagnostic, never a fitted controller result.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import threading

import finish_once_pilot as original
import format_fixed_controls as fixed
import science_mentor_expansion as gates
import prospective_collect as protocol
from prospective_parallel import RequestJournal, KeyLocks, durable_line
from matched_review_b import validate_review_charge

SCHEMA = 'strong-guided-drop-review-standard-repair-v1'
PATH = 'guided/drop-review-repair'
COUNTS = {'chemistry': 32, 'physics': 32}
EXTRA = ('guided', 'guided/reuse', 'retry/reuse')
ORACLE_PATHS = (*protocol.PLAIN, *EXTRA)
WORKERS = 2
SCREEN = {'novel_correct_beyond_existing_ten_at_least': 6,
          'full_trajectory_compute_ratio_to_retained_at_most': 1.10,
          'scope': 'TRAIN action headroom only; no controller promotion or fresh-data claim'}
sha, digest, read, write = fixed.sha, fixed.digest, gates.read, protocol.atomic_json


def sources():
    """Exact static project dependency closure, independent of ambient imports."""
    project, here = Path(__file__).resolve().parents[4], Path(__file__).resolve().parent
    def local(path):
        if path is None:
            return None
        path = Path(path).resolve()
        if (path.is_file() and path.suffix == '.py' and path.is_relative_to(project)
                and not any(p.startswith('.') or p in ('site-packages', '__pycache__')
                            for p in path.relative_to(project).parts)):
            return path
        return None
    def locate(name):
        if name in sys.modules and (path := local(getattr(sys.modules[name], '__file__', None))):
            return path
        rel = Path(*name.split('.'))
        for folder in dict.fromkeys([str(here), *sys.path]):
            for path in (Path(folder or '.') / rel.with_suffix('.py'), Path(folder or '.') / rel / '__init__.py'):
                if (path := local(path)) is not None:
                    return path
        return None
    found, pending = set(), [(Path(__file__).resolve(), 'drop_review_repair_pilot')]
    while pending:
        path, module = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [v.name for v in node.names]
            elif isinstance(node, ast.ImportFrom):
                package = module.split('.')[:-node.level] if node.level else []
                name = '.'.join([*package, *([node.module] if node.module else [])])
                names = [name, *[name + '.' + v.name for v in node.names if v.name != '*']]
            for name in names:
                if name and (dependency := locate(name)) is not None:
                    pending.append((dependency, name))
    return {str(path): sha(path) for path in sorted(found)}


def public_inputs(args):
    rows = gates.public_train(args.data)
    parent, complete = read(args.mentor / 'contract.json'), read(args.mentor / 'complete.json')
    if (complete['questions'] != 192 or complete['records'] != 960
            or complete['contract_sha256'] != sha(args.mentor / 'contract.json')
            or complete['bank_sha256'] != sha(args.mentor / 'train.jsonl')
            or parent['new_question_split_sha256'] != sha(args.data / 'splits/train-questions.jsonl')):
        raise ValueError('Preserve the entire original sealed strong192 cohort')
    chosen = [v for v in fixed.selection(rows, parent) if v['subject'] in COUNTS]
    if len(chosen) != 64 or Counter(v['subject'] for v in chosen) != COUNTS:
        raise ValueError('Require all64 prespecified new TRAIN science questions')
    by_id = {r['id']: r for r in rows}; rows = [by_id[v['id']] for v in chosen]
    trees, prefix_sha = original.plain_controls(args.data, set(by_id[r['id']]['id'] for r in rows))
    bank = protocol.strict_records(args.mentor / 'train.jsonl')
    strong = {(v['id'], v['kind'], v['path']): v['response'] for v in bank}
    if len(bank) != 960 or len(strong) != 960:
        raise ValueError('Duplicate or incomplete original strong bank')
    reviews = {}
    for row in rows:
        qid, tree = row['id'], trees[row['id']]
        for path in protocol.PLAIN:
            protocol.validate_node(row, 'plain', 'solver', path, tree[path], tree, {})
        reviews[qid, ''] = strong[qid, 'judge', '']
        protocol.validate_node(row, 'strong', 'judge', '', reviews[qid, ''], tree, reviews)
        validate_review_charge(reviews[qid, ''])
        for path in (*EXTRA, 'guided/retry'):
            tree[path] = strong[qid, 'solver', path]
            protocol.validate_node(row, 'strong', 'solver', path, tree[path], tree, reviews)
        if tree['guided/retry'] != tree['repair/retry']:
            raise ValueError('The pure-retry alias must preserve its exact original response')
    evidence = {str(path.resolve()): sha(path) for path in (
        args.mentor / 'contract.json', args.mentor / 'complete.json', args.mentor / 'train.jsonl',
        args.data / 'splits/train-questions.jsonl', args.data / 'train-plain-complete.json')}
    return rows, trees, reviews, chosen, prefix_sha, evidence


def prompt(row, current):
    if row.get('answer') != '' or row['domain'] != 'science':
        raise ValueError('Only the public blank-reference science question reaches repair')
    return protocol.old.repair_prompt(row, current, '')


def request_payload(row, tree):
    return {'model': protocol.old.MODEL, 'prompt': prompt(row, tree['guided']), 'raw': True,
        'stream': False, 'logprobs': True, 'keep_alive': '30m', 'options': {
            'temperature': .65, 'seed': fixed.logical_seed(row['id'], 'repair/repair'),
            'num_predict': 256, 'num_ctx': 4096}}


def paid_prefix(tree, reviews, qid):
    return {'root_solver_sha256': digest(tree['']), 'root_review_sha256': digest(reviews[qid, '']),
            'first_guided_sha256': digest(tree['guided'])}


def replay_node(row, tree, raw):
    RequestJournal.validate(raw)
    payload = request_payload(row, tree)
    if raw['payload'] != payload:
        raise ValueError('Completed generation differs from the registered standard repair')
    response = raw['response']
    if (response.get('model') != protocol.old.MODEL or response.get('done') is not True
            or type(response.get('prompt_eval_count')) is not int or not 0 < response['prompt_eval_count'] <= 4096
            or type(response.get('eval_count')) is not int or not 0 <= response['eval_count'] <= 256):
        raise ValueError('Invalid full physical compact token receipt')
    elapsed = raw['returned_unix'] - raw['started_unix']
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid physical latency receipt')
    def saved(endpoint, actual):
        if endpoint != '/api/generate' or actual != payload:
            raise ValueError('Frozen actor requested another prompt or protocol')
        return copy.deepcopy(response)
    node = protocol.actor(saved, row, payload['prompt'], payload['options']['seed'], 256)
    node['latency_seconds'] = elapsed
    return node


def logical_record(row, tree, reviews, raw):
    return {'id': row['id'], 'subject': row['subject'], 'kind': 'solver', 'path': PATH,
        'protocol': SCHEMA, 'paid_prefix': paid_prefix(tree, reviews, row['id']),
        'request_sha256': raw['request_sha256'], 'raw_response_sha256': raw['response_sha256'],
        'response': replay_node(row, tree, raw)}


def restore(rows, trees, reviews, records, raw_records, complete=False):
    by_id, expected, raw = {r['id']: r for r in rows}, {}, {}
    for row in rows:
        key = digest({'endpoint': '/api/generate', 'payload': request_payload(row, trees[row['id']])})
        if key in expected:
            raise ValueError('Distinct questions unexpectedly share a physical request')
        expected[key] = row['id']
    for item in raw_records:
        key = item['request_sha256']
        if key not in expected or key in raw:
            raise ValueError('Foreign or duplicate completed physical repair')
        qid = expected[key]
        replay_node(by_id[qid], trees[qid], item)
        raw[key] = item
    known = {}
    for item in records:
        qid, key = item['id'], item['request_sha256']
        if qid not in by_id or qid in known or key not in raw or expected[key] != qid:
            raise ValueError('Duplicate logical repair or missing raw physical receipt')
        if item != logical_record(by_id[qid], trees[qid], reviews, raw[key]):
            raise ValueError('Saved logical repair differs from its physical receipt or paid prefix')
        known[qid] = item
    if complete and (set(raw) != set(expected) or set(known) != set(by_id)):
        raise ValueError('All64 physical/logical repairs must be sealed before grading')
    return known, {'questions': len(by_id), 'physical_calls': len(raw), 'logical_repairs': len(known)}


def registration(args):
    rows, trees, reviews, chosen, prefix_sha, evidence = public_inputs(args)
    return {'schema': SCHEMA, 'sources': sources(), 'input_evidence': evidence, 'selection': chosen,
        'questions': 64, 'subjects': COUNTS, 'new_solver_calls': 64, 'new_teacher_calls': 0,
        'new_neural_calls': 0, 'path': PATH, 'seed_path': 'repair/repair', 'screen': SCREEN,
        'solver': protocol.old.MODEL, 'solver_digest': protocol.old.MODEL_DIGEST,
        'reviewer_digest': protocol.old.JUDGE_DIGEST, 'output_cap': 256, 'temperature': .65,
        'context_length': 4096, 'workers': WORKERS, 'maximum_cloud_gpus': 2,
        'prompt': 'Exact original standard repair template applied to paid strong guided answer; empty feedback',
        'full_trajectory': ['root compact answer', 'paid strong root review (all physical phases)',
                            'paid first guided compact answer', 'standard256 second repair'],
        'maximum_answer_changing_continuations': 2, 'logical_reviews': 1,
        'existing_ten_paths': list(ORACLE_PATHS), 'guided_retry_alias': 'repair/retry',
        'controls': digest({'trees': trees, 'reviews': sorted((q, p, v) for (q, p), v in reviews.items())}),
        'sealed_plain_prefix_sha256': prefix_sha,
        'requests': [{'id': r['id'], 'payload_sha256': digest(request_payload(r, trees[r['id']])),
                      'paid_prefix': paid_prefix(trees[r['id']], reviews, r['id'])} for r in rows],
        'paths': {k: str(getattr(args, k).resolve()) for k in ('data', 'mentor', 'output')},
        'training_only': True, 'test_used': False, 'development_values_read': False,
        'references_supplied': False, 'grades_after': 'Complete validated hash-sealed64question raw/logical grid',
        'transport': 'One attempt per request; preserve unresolved remote errors as lower-bound acquisition costs',
        'no_quality_regeneration': True, 'scope': SCREEN['scope'],
        'promotion_requires': 'Matched B continuation and controls, fitted action model, separate fresh validation'}


def register(args):
    record = registration(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'registration.json', record)
    print(json.dumps({'registered': True, 'registration_sha256': sha(args.output / 'registration.json'),
                      'new_compact_calls': 64, 'calls_started': 0}), flush=True)


class Collector:
    def __init__(self, trees, reviews, known, stream, journal):
        self.trees, self.reviews, self.known, self.stream, self.journal = trees, reviews, known, stream, journal
        self.keys, self.lock, self.stopped = KeyLocks(), threading.RLock(), threading.Event()

    def one(self, row):
        try:
            with self.keys.get(row['id']):
                if self.stopped.is_set() or row['id'] in self.known:
                    return
                payload = request_payload(row, self.trees[row['id']])
                self.journal('/api/generate', payload)
                key = digest({'endpoint': '/api/generate', 'payload': payload})
                item = logical_record(row, self.trees[row['id']], self.reviews, self.journal.known[key])
                with self.lock:
                    durable_line(self.stream, item); self.known[row['id']] = item
                    if len(self.known) % 8 == 0:
                        print(f'Drop-review standard repairs {len(self.known)}/64', flush=True)
        except BaseException:
            self.stopped.set(); raise


def collect(args):
    with ExitStack() as stack:
        lock = stack.enter_context((args.output / 'collection.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'complete.json').exists():
            raise ValueError('Preserve the completed pilot; do not resample')
        contract = registration(args); registration_path = args.output / 'registration.json'
        if not args.registration_sha256 or sha(registration_path) != args.registration_sha256 or read(registration_path) != contract:
            raise ValueError('Externally recorded registration changed')
        rows, trees, reviews, *_ = public_inputs(args)
        bank, raw_path = args.output / 'train.jsonl', args.output / 'raw-generations.jsonl'
        stream, raw_stream = stack.enter_context(bank.open('a')), stack.enter_context(raw_path.open('a'))
        for handle in (stream, raw_stream):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw_records = protocol.strict_records(raw_path)
        known, _ = restore(rows, trees, reviews, protocol.strict_records(bank), raw_records)
        import modal
        worker = modal.Cls.from_name(protocol.APP, 'Ollama')()
        worker.request.autoscaler.update(max_containers=2)
        errors = args.output / 'transport-errors.jsonl'
        journal = RequestJournal(protocol.resilient(worker, errors, attempts=1), raw_stream, raw_records, slots=WORKERS)
        tags = {v['name']: v['digest'] for v in journal('/api/tags')['models']}
        if tags.get(protocol.old.MODEL) != protocol.old.MODEL_DIGEST:
            raise ValueError('Frozen compact solver manifest changed')
        collector = Collector(trees, reviews, known, stream, journal)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(collector.one, rows))
        known, coverage = restore(rows, trees, reviews, protocol.strict_records(bank), protocol.strict_records(raw_path), True)
        if coverage != {'questions': 64, 'physical_calls': 64, 'logical_repairs': 64} or registration(args) != contract:
            raise ValueError('Cohort, input evidence or source changed during collection')
        error_count = len(protocol.strict_records(errors))
        write(args.output / 'complete.json', {'schema': SCHEMA, 'coverage': coverage,
            'registration_sha256': sha(registration_path), 'bank_sha256': sha(bank), 'raw_sha256': sha(raw_path),
            'transport_errors_sha256': sha(errors) if errors.exists() else None,
            'transport_error_records': error_count, 'acquisition_compute_is_lower_bound': bool(error_count),
            'physical_solver_calls': 64, 'new_teacher_calls': 0, 'new_neural_calls': 0,
            'training_only': True, 'scored': False})
        print('All64 standard second repairs sealed; no accuracy scored', flush=True)


def validated_complete(args):
    done = read(args.output / 'complete.json')
    if (done['schema'] != SCHEMA or done['coverage'] != {'questions': 64, 'physical_calls': 64, 'logical_repairs': 64}
            or done['physical_solver_calls'] != 64 or done['new_teacher_calls'] != 0 or done['new_neural_calls'] != 0
            or done['training_only'] is not True or done['scored'] is not False):
        raise ValueError('Require the entire sealed ungraded64question grid')
    for name, key in (('registration.json', 'registration_sha256'), ('train.jsonl', 'bank_sha256'), ('raw-generations.jsonl', 'raw_sha256')):
        if sha(args.output / name) != done[key]:
            raise ValueError('Changed sealed pilot artifact')
    errors = args.output / 'transport-errors.jsonl'
    count = len(protocol.strict_records(errors))
    if ((sha(errors) if errors.exists() else None) != done['transport_errors_sha256']
            or count != done['transport_error_records'] or bool(count) != done['acquisition_compute_is_lower_bound']):
        raise ValueError('Changed transport accounting')
    if read(args.output / 'registration.json') != registration(args):
        raise ValueError('Changed registered source, input or feasibility screen')
    rows, trees, reviews, *_ = public_inputs(args)
    known, coverage = restore(rows, trees, reviews, protocol.strict_records(args.output / 'train.jsonl'),
                              protocol.strict_records(args.output / 'raw-generations.jsonl'), True)
    if coverage != done['coverage']:
        raise ValueError('Changed full-grid coverage')
    return rows, trees, reviews, known, done


def trajectory(nodes, review=None):
    solver_in, solver_out = sum(v['input_tokens'] for v in nodes), sum(v['output_tokens'] for v in nodes)
    review_in = review['input_tokens'] if review else 0
    review_out = review['output_tokens'] if review else 0
    return {'solver_input_tokens': solver_in, 'solver_output_tokens': solver_out,
        'review_input_tokens': review_in, 'review_output_tokens': review_out,
        'solver_calls': len(nodes), 'logical_review_calls': int(review is not None),
        'physical_review_calls': review['physical_api_calls'] if review else 0,
        'compute': (4*(solver_in+solver_out)+3*(review_in+review_out))/3.09,
        'total_parameter_compute': (4*(solver_in+solver_out)+35*(review_in+review_out))/3.09}


def aggregate(rows, trees, reviews, known, references):
    if len(rows) != 64 or Counter(r['subject'] for r in rows) != COUNTS or set(known) != {r['id'] for r in rows}:
        raise ValueError('Always report every registered64 science question')
    outcomes = []
    for row in rows:
        qid, tree, review = row['id'], trees[row['id']], reviews[row['id'], '']
        target = references[qid]
        methods = {name: ([tree[p] for p in paths], None) for name, paths in fixed.FIXED.items()}
        methods.update({'Strong first guided': ([tree[''], tree['guided']], review),
            'Strong retained second repair': ([tree[''], tree['guided'], tree['guided/reuse']], review),
            'Strong repair after retry': ([tree[''], tree['retry'], tree['retry/reuse']], review),
            'Drop-review standard second repair': ([tree[''], tree['guided'], known[qid]['response']], review)})
        values = {name: {'correct': bool(nodes[-1]['answer'] and nodes[-1]['answer'] == target),
                        'answer': nodes[-1]['answer'], **trajectory(nodes, paid)} for name, (nodes, paid) in methods.items()}
        original = {p: bool(tree[p]['answer'] and tree[p]['answer'] == target) for p in ORACLE_PATHS}
        outcomes.append({'id': qid, 'subject': row['subject'], 'methods': values, 'existing_paths': original,
            'existing_ten_oracle_correct': any(original.values()),
            'novel_correct': values['Drop-review standard second repair']['correct'] and not any(original.values())})
    reports = {}
    for panel in ('all_science', 'chemistry', 'physics'):
        selected = [v for v in outcomes if panel == 'all_science' or v['subject'] == panel]
        n = len(selected); methods = {}
        for name in selected[0]['methods']:
            values = [v['methods'][name] for v in selected]
            methods[name] = {'n': n, 'correct': sum(v['correct'] for v in values),
                'accuracy': sum(v['correct'] for v in values)/n,
                **{'mean_'+key: sum(v[key] for v in values)/n for key in trajectory([], None)}}
        paired = {}
        for name in methods:
            if name == 'Drop-review standard second repair':
                continue
            paired[name] = {'wins': sum(v['methods']['Drop-review standard second repair']['correct'] and not v['methods'][name]['correct'] for v in selected),
                'losses': sum(not v['methods']['Drop-review standard second repair']['correct'] and v['methods'][name]['correct'] for v in selected)}
        reports[panel] = {'n': n, 'methods': methods, 'drop_review_paired_vs': paired,
            'existing_ten_path_oracle_correct': sum(v['existing_ten_oracle_correct'] for v in selected),
            'novel_correct_beyond_existing_ten': sum(v['novel_correct'] for v in selected),
            'combined_eleven_path_oracle_correct': sum(v['existing_ten_oracle_correct'] or v['novel_correct'] for v in selected)}
    all_rows = reports['all_science']; methods = all_rows['methods']
    ratio = methods['Drop-review standard second repair']['mean_compute']/methods['Strong retained second repair']['mean_compute']
    checks = {'novel_correct_at_least6': all_rows['novel_correct_beyond_existing_ten'] >= 6,
              'mean_full_compute_at_most_1p10_retained': ratio <= 1.10}
    return reports, outcomes, {'criteria': SCREEN, 'checks': checks, 'passed': all(checks.values()),
        'full_trajectory_compute_ratio_to_retained': ratio, 'controller_promoted': False}


def grade(args):
    # Full physical/logical/source seal must pass before TRAIN labels are parsed.
    rows, trees, reviews, known, done = validated_complete(args)
    target = args.output / 'graded'
    if target.exists():
        raise ValueError('Preserve the first complete diagnostic and its fixed screen')
    labels = {v['id']: v for v in fixed.lines(args.data / 'splits/train-questions.jsonl')}
    for row in rows:
        if protocol.public_row(labels[row['id']]) != row:
            raise ValueError('Training target row differs from the sealed public prompt')
    reports, outcomes, screen = aggregate(rows, trees, reviews, known, {r['id']: labels[r['id']]['answer'] for r in rows})
    validated_complete(args)
    target.mkdir(); write(target / 'private-outcomes.json', outcomes)
    write(target / 'summary.json', {'schema': SCHEMA, 'reports': reports, 'screen': screen,
        'complete_sha256': sha(args.output / 'complete.json'), 'registration_sha256': sha(args.output / 'registration.json'),
        'private_outcomes_sha256': sha(target / 'private-outcomes.json'), 'questions': 64,
        'new_compact_calls': 64, 'new_teacher_calls': 0, 'new_neural_calls': 0,
        'transport_error_records': done['transport_error_records'],
        'acquisition_compute_is_lower_bound': done['acquisition_compute_is_lower_bound'],
        'training_only': True, 'test_used': False, 'controller_accuracy_claim': False,
        'scope': SCREEN['scope'], 'matched_B_required_if_promoted': True})
    print(json.dumps({'summary_sha256': sha(target / 'summary.json'), 'screen': screen, 'reports': reports}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'register', 'collect', 'grade'))
    for name in ('data', 'mentor', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    if args.phase == 'plan':
        record = registration(args)
        print(json.dumps({'questions': record['questions'], 'new_compact_calls': 64, 'new_teacher_calls': 0,
            'new_neural_calls': 0, 'screen': SCREEN, 'source_sha256': sha(__file__), 'registered': False}), flush=True)
    else:
        {'register': register, 'collect': collect, 'grade': grade}[args.phase](args)
