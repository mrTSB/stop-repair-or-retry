"""Stage all192 remaining TRAIN science pairs behind completed core comparisons.

New response banks only; no fitted models, neural inference, or evaluation.
References are lexically skipped. Every member receives the same registered
counterfactuals regardless of answer validity, confidence, or correctness.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import prospective_collect as protocol
from prospective_parallel import RequestJournal, KeyLocks, durable_line, raw_coverage
import replay_new_source_mentor as controls
import mentor_pairs

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
SCHEMA = 'round2-science-mentor-expansion-v1'
SALT = 'round2-science-strong-expansion-v1'
PATHS = (('judge', ''), ('solver', 'guided'), ('solver', 'guided/reuse'),
         ('solver', 'retry/reuse'), ('solver', 'guided/retry'))
WORKERS = 4
read = controls.read
sha = controls.sha
write = protocol.atomic_json


def source_manifest():
    """Static import closure; unrelated experiments and tests are not roots."""
    def allowed(path):
        if not path:
            return None
        path = Path(path).resolve()
        if (not path.is_file() or path.suffix != '.py' or not path.is_relative_to(PROJECT)
                or any(part.startswith('.') or part in ('site-packages', '__pycache__')
                       for part in path.relative_to(PROJECT).parts)):
            return None
        return path
    def locate(name):
        if not name:
            return None
        if name in sys.modules:
            return allowed(getattr(sys.modules[name], '__file__', None))
        relative = Path(*name.split('.'))
        for folder in dict.fromkeys([str(HERE), *sys.path]):
            for path in (Path(folder or '.') / relative.with_suffix('.py'),
                         Path(folder or '.') / relative / '__init__.py'):
                if path.is_file():
                    return allowed(path)
        return None
    pending = [(Path(__file__).resolve(), 'science_mentor_expansion'),
               (HERE.parent / 'modal_solver.py', 'modal_solver'),
               (HERE / 'development_metadata_review.py', 'development_metadata_review')]
    found = set()
    while pending:
        path, module = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                package = module.split('.')[:-node.level] if node.level else []
                name = '.'.join([*package, *([node.module] if node.module else [])])
                names = [name, *[name + '.' + alias.name for alias in node.names if alias.name != '*']]
            for name in names:
                dependency = locate(name)
                if dependency is not None:
                    pending.append((dependency, name))
    return {str(path.resolve()): sha(path) for path in sorted(found)}


def public_train(data):
    path = data / 'splits/train-questions.jsonl'
    rows = [protocol.parse_public_line(line) for line in path.read_text().splitlines()]
    if (len(rows) != 512 or len({r['id'] for r in rows}) != 512
            or Counter(r['subject'] for r in rows) != {'math': 256, 'chemistry': 128, 'physics': 128}):
        raise ValueError('Preserve the original512-question TRAIN reservation')
    return rows


def selection(rows, parent_contract):
    original = parent_contract['selection']
    prior = [r for r in original if r['bank'] == 'new']
    if (len(original) != 192 or len({r['id'] for r in original}) != 192 or len(prior) != 128
            or Counter(r['subject'] for r in prior) != {'math': 64, 'chemistry': 32, 'physics': 32}):
        raise ValueError('Changed existing source-matched pairing cohort')
    by_id = {r['id']: r for r in rows}
    if any(r['id'] not in by_id or by_id[r['id']]['subject'] != r['subject'] for r in prior):
        raise ValueError('Existing new pairs no longer match TRAIN membership')
    old = {r['id'] for r in prior}
    groups = []
    for subject in ('chemistry', 'physics'):
        values = [r for r in rows if r['subject'] == subject and r['id'] not in old]
        if len(values) != 96:
            raise ValueError('Include all96 remaining questions of each science subject')
        groups.append(sorted(values, key=lambda r: hashlib.sha256(f'{SALT}:{r["id"]}'.encode()).digest()))
    return [{'id': row['id'], 'subject': row['subject'], 'bank': 'new'} for pair in zip(*groups) for row in pair]


def plain_prefix(data, ids):
    receipt = read(data / 'train-plain-complete.json')
    path = data / 'train-observations.jsonl'
    with path.open('rb') as stream:
        # The ordinary collector may still append after this immutable prefix.
        raw = b''.join(stream.readline() for _ in range(receipt['base_records']))
    if not raw.endswith(b'\n') or hashlib.sha256(raw).hexdigest() != receipt['base_bank_sha256']:
        raise ValueError('Sealed plain TRAIN prefix changed')
    trees = {qid: {} for qid in ids}
    for line in raw.splitlines():
        item = json.loads(line)
        if item['id'] not in trees:
            continue
        if item['kind'] != 'solver' or item['path'] not in protocol.PLAIN or item['path'] in trees[item['id']]:
            raise ValueError('Unexpected/duplicate sealed TRAIN response')
        trees[item['id']][item['path']] = item['response']
    if any(set(tree) != set(protocol.PLAIN) for tree in trees.values()):
        raise ValueError('Missing an existing plain TRAIN counterfactual')
    return trees, receipt['base_bank_sha256']


def inputs(args):
    if sha(args.proposal / 'proposal.json') != args.proposal_sha256:
        raise ValueError('Approved expansion proposal changed')
    proposal = read(args.proposal / 'proposal.json')
    if sha(args.proposal / 'selection.json') != proposal['selection_sha256']:
        raise ValueError('Approved outcome-independent membership changed')
    rows = public_train(args.data)
    parent, completion = read(args.parent_bank / 'contract.json'), read(args.parent_bank / 'complete.json')
    if (sha(args.parent_bank / 'contract.json') != completion['contract_sha256']
            or sha(args.parent_bank / 'train.jsonl') != completion['bank_sha256']
            or completion['questions'] != 192 or completion['records'] != 960
            or completion['contract_sha256'] != proposal['parent_contract_sha256']
            or completion['bank_sha256'] != proposal['parent_bank_sha256']
            or parent['new_question_split_sha256'] != sha(args.data / 'splits/train-questions.jsonl')
            or proposal['train_question_sha256'] != parent['new_question_split_sha256']):
        raise ValueError('Original TRAIN reservation or completed parent bank changed')
    chosen = selection(rows, parent)
    if chosen != read(args.proposal / 'selection.json'):
        raise ValueError('Proposal is not the complete fixed-hash science complement')
    by_id = {r['id']: r for r in rows}
    trees, prefix_sha = plain_prefix(args.data, {r['id'] for r in chosen})
    if prefix_sha != proposal['plain_prefix_sha256']:
        raise ValueError('Approved plain-prefix controls changed')
    for item in chosen:
        row, tree = by_id[item['id']], trees[item['id']]
        for path, node in tree.items():
            protocol.validate_node(row, 'plain', 'solver', path, node, tree, {})
        # Existing pure-retry control is exactly the future guided/retry action.
        protocol.validate_node(row, 'strong', 'solver', 'guided/retry', tree['repair/retry'], tree, {})
    if (parent['solver_digest'], parent['reviewer_digest']) != (protocol.old.MODEL_DIGEST, protocol.old.JUDGE_DIGEST):
        raise ValueError('Frozen compact solver or reviewer changed')
    return [by_id[r['id']] for r in chosen], trees, chosen, proposal


def registration(args):
    required_priority = (args.data.parent / 'development-metadata-newsource-v1' /
                         'metadata-replay-complete.json').resolve()
    if required_priority not in [path.resolve() for path in args.priority_completion]:
        raise ValueError('Require the final metadata-newsource completion gate before registration')
    _, _, chosen, proposal = inputs(args)
    return {'schema': SCHEMA, 'source_files': source_manifest(),
        'proposal_sha256': args.proposal_sha256, 'selection': chosen,
        'selection_sha256': proposal['selection_sha256'], 'train_question_sha256': proposal['train_question_sha256'],
        'parent_bank_sha256': proposal['parent_bank_sha256'], 'parent_contract_sha256': proposal['parent_contract_sha256'],
        'plain_prefix_sha256': proposal['plain_prefix_sha256'],
        'paths': {name: str(getattr(args, name).resolve()) for name in ('data', 'parent_bank', 'proposal', 'b_reference', 'output')},
        'priority_completions': [str(p.resolve()) for p in args.priority_completion],
        'required_gates': {'ordinary_train_questions': 512, 'ordinary_dev_questions': 256, 'matched_B_summary': True,
                           'final_metadata_newsource_completion': str(required_priority)},
        'solver_digest': protocol.old.MODEL_DIGEST, 'reviewer_digest': protocol.old.JUDGE_DIGEST,
        'review_caps': [1024, 512], 'guided_cap': 128, 'temperature': {'solver': .65, 'review': 0.},
        'contexts': {'solver': 4096, 'review': 8192}, 'new_questions': 192, 'new_logical_records': 960,
        'counterfactuals': [list(p) for p in PATHS], 'workers': WORKERS, 'max_cloud_gpus': 2,
        'max_cloud_requests': WORKERS, 'reference_answers_supplied': False,
        'training_labels_opened_for_collection': False, 'development_or_test_values_decoded': False,
        'outcome_based_filtering': False, 'neural_training_or_inference': False,
        'fitting_contract': {'schema': 'science-expanded-source-matched-pairs-v1', 'training_questions': 512,
            'review_pairs': 320, 'review_subjects': {'math': 64, 'chemistry': 128, 'physics': 128},
            'pooled_review_pair_weights': {'math': 4., 'chemistry': 1., 'physics': 1.},
            'subject_local_question_weight': 1., 'ordinary_full_training_question_weight': 1.,
            'metadata_sibling_weight': 'Equal total base weight per question:1/number of sibling records.',
            'metadata_subject_target_weight': {'math': 4., 'chemistry': 1., 'physics': 1.},
            'metadata_effective_record_weight': 'question-normalized sibling weight times observed-subject target weight.',
            'existing_fitters_modified': False, 'new_fitters_required': True,
            'new_neural_work_requires_separate_scheduling': True}}


def verify_sources(record):
    for path, expected in record['source_files'].items():
        if sha(path) != expected:
            raise ValueError('Registered collector dependency changed: ' + path)


def register(args):
    if args.output.exists():
        raise ValueError('Use a new expansion area; never overwrite registration')
    record = registration(args)
    args.output.mkdir(parents=True)
    write(args.output / 'registration.json', record)
    print(json.dumps({'registered': True, 'registration_sha256': sha(args.output / 'registration.json'),
                      'new_questions': 192, 'remote_calls_started': False}), flush=True)


def completed_json(path):
    if not path.exists():
        return None
    try:
        return read(path)
    except json.JSONDecodeError:
        if time.time() - path.stat().st_mtime < 10:
            return None
        raise ValueError('Invalid completed prerequisite receipt: ' + str(path))


def locked_digest(path):
    with path.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        digest = hashlib.sha256()
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
        return digest.hexdigest()


def prerequisite_receipts(args, record):
    result, missing = {}, []
    for split, count in (('train', 512), ('dev', 256)):
        path = args.data / (split + '-ordinary-complete.json')
        complete = completed_json(path)
        if complete is None:
            missing.append(str(path)); continue
        contract_path = args.data / (split + '-ordinary-contract.json')
        contract = read(contract_path)
        if (complete['questions'] != count or complete['base_records'] != count * 11
                or complete['late_records'] != count * 6 or complete['references_supplied'] is not False
                or sha(contract_path) != complete['contract_sha256']
                or contract['mode'] != 'ordinary' or contract['source_sha256'] != sha(HERE / 'collect.py')
                or contract['solver_digest'] != record['solver_digest']
                or contract['reviewer_digest'] != record['reviewer_digest']):
            raise ValueError('Incomplete or changed ordinary prerequisite: ' + split)
        if split == 'train' and contract['split_sha256'] != record['train_question_sha256']:
            raise ValueError('Ordinary TRAIN belongs to a different reservation')
        for field, suffix in (('base_bank_sha256', '-observations.jsonl'), ('late_bank_sha256', '-late-observations.jsonl')):
            if locked_digest(args.data / (split + suffix)) != complete[field]:
                raise ValueError('Completed ordinary prerequisite bank changed')
        result[split] = {'completion_sha256': sha(path), 'contract_sha256': sha(contract_path),
                         'question_sha256': contract['split_sha256'],
                         'base_bank_sha256': complete['base_bank_sha256'], 'late_bank_sha256': complete['late_bank_sha256']}
    b = completed_json(args.b_reference)
    if b is None:
        missing.append(str(args.b_reference))
    elif 'dev' in result:
        identity_path = args.b_reference.parent / 'identity.json'
        controls.validate_b(b, read(identity_path), result['dev']['question_sha256'])
        if locked_digest(args.b_reference.parent / 'dev-observations.jsonl') != b['bank_sha256']:
            raise ValueError('Completed matched-B response bank changed')
        result['matched_B'] = {'summary_sha256': sha(args.b_reference), 'identity_sha256': sha(identity_path),
                               'bank_sha256': b['bank_sha256']}
    priority = {}
    for path in args.priority_completion:
        marker = completed_json(path)
        if marker is None:
            missing.append(str(path)); continue
        # Priority markers contain only hashes; their sibling question/response
        # values are never decoded by this training collector.
        if 'files' in marker:
            required = {'registration.json', 'identity.json', 'cache-import.json', 'summary.json', 'search.json',
                        'dev-observations.jsonl', 'private-neural-cache.pkl', 'raw-generations.jsonl'}
            if (marker.get('schema') != 'metadata-review-development-v1'
                    or marker.get('source_sha256') != sha(HERE / 'development_metadata_review.py')
                    or set(marker['files']) not in (required, required | {'transport-errors.jsonl'})):
                raise ValueError('Invalid final metadata comparison seal')
            for name, expected in marker['files'].items():
                if sha(path.parent / name) != expected:
                    raise ValueError('Completed priority comparison changed')
        else:
            if (marker.get('schema') != controls.SCHEMA
                    or marker.get('source_sha256') != sha(controls.__file__)):
                raise ValueError('Invalid priority completion marker')
            for field, name in (('summary_sha256', 'summary.json'), ('identity_sha256', 'identity.json'),
                                 ('import_sha256', 'cache-import.json')):
                if field not in marker or sha(path.parent / name) != marker[field]:
                    raise ValueError('Invalid priority completion marker')
        priority[str(path.resolve())] = sha(path)
    result['priority_completions'] = priority
    return result, missing


def validate_record(row, tree, reviews, record):
    if (record['id'] != row['id'] or record['dose'] != 1024
            or (record['kind'], record['path']) not in PATHS):
        raise ValueError('Unexpected expansion response key')
    kind, path, node = record['kind'], record['path'], record['response']
    protocol.validate_node(row, 'strong', kind, path, node, tree, reviews)
    if kind == 'judge':
        controls.control.validate_review_charge(node)
        for i, phase in enumerate(node['calls']):
            if (phase['temperature'], phase['context_length'], phase['output_cap']) != (0., 8192, 1024 if i == 0 else 512):
                raise ValueError('Strong-review physical decoding changed')
    alias = kind == 'solver' and path == 'guided/retry'
    if record.get('exact_control_reused') is not alias:
        raise ValueError('Only the identical pure retry is a reused response')
    if alias and (node != tree['repair/retry'] or record.get('plain_source_path') != 'repair/retry'):
        raise ValueError('Pure retry did not exactly preserve the original compact response')


def restore(rows, plain, records, raw):
    by_id = {row['id']: row for row in rows}
    trees, reviews, known = {qid: dict(tree) for qid, tree in plain.items()}, {}, {}
    for item in records:
        qid, kind, path = item['id'], item['kind'], item['path']
        key = qid, kind, path
        if qid not in by_id or key in known:
            raise ValueError('Nonselected or duplicate expansion record')
        # Within each question, prerequisites must already be durably present.
        if kind == 'solver' and path != 'guided/retry' and (qid, '') not in reviews:
            raise ValueError('Solver receipt precedes its paid root review')
        validate_record(by_id[qid], trees[qid], reviews, item)
        known[key] = item
        if kind == 'judge':
            reviews[qid, path] = item['response']
        else:
            trees[qid][path] = item['response']
    coverage = raw_coverage({'strong': [r for r in records if not r['exact_control_reused']]}, raw)
    return trees, reviews, known, coverage


class PairCollector:
    def __init__(self, rows, trees, reviews, known, stream, request):
        self.rows, self.trees, self.reviews, self.known = rows, trees, reviews, known
        self.stream, self.request = stream, request
        self.lock, self.keys = threading.RLock(), KeyLocks()
        self.stopped = threading.Event()

    def one(self, row):
        try:
            return self._one(row)
        except BaseException:
            self.stopped.set()
            raise

    def _one(self, row):
        if row.get('answer') != '':
            raise ValueError('Training references must remain blank')
        qid = row['id']
        with self.keys.get(qid):
            tree = self.trees[qid]
            for kind, path in PATHS:
                if self.stopped.is_set():
                    return
                key = qid, kind, path
                if key in self.known:
                    continue
                alias = path == 'guided/retry'
                if alias:
                    node = tree['repair/retry']
                elif kind == 'judge':
                    node = protocol.canonical_review(protocol.review(self.request, row, tree[''], 1024, 512))
                else:
                    parent = path.rsplit('/', 1)[0] if '/' in path else ''
                    prompt = protocol.short_repair(row, tree[parent], self.reviews[qid, ''])
                    node = protocol.actor(self.request, row, prompt, mentor_pairs.seed_for(qid, path), 128)
                record = {'id': qid, 'dose': 1024, 'kind': kind, 'path': path, 'response': node,
                          'exact_control_reused': alias, 'plain_source_path': 'repair/retry' if alias else None}
                validate_record(row, tree, self.reviews, record)
                with self.lock:
                    durable_line(self.stream, record)
                    self.known[key] = record
                    if kind == 'judge':
                        self.reviews[qid, path] = node
                    else:
                        tree[path] = node


def collect(args):
    path = args.output / 'registration.json'
    if not args.registration_sha256 or sha(path) != args.registration_sha256:
        raise ValueError('Registration differs from its externally recorded digest')
    record = read(path)
    # Reconstruct the fixed selection, sources and contract before any remote call.
    if record != registration(args):
        raise ValueError('Registered expansion inputs or implementation changed')
    with ExitStack() as stack:
        lock = stack.enter_context((args.output / 'collection.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'complete.json').exists():
            raise ValueError('Expansion is already complete; do not recollect')
        previous = None
        while True:
            verify_sources(record)
            gates, missing = prerequisite_receipts(args, record)
            if not missing:
                break
            if not args.wait:
                raise ValueError('Core prerequisites are unfinished: ' + json.dumps(missing))
            if missing != previous:
                print(json.dumps({'waiting_for': missing, 'physical_calls_started': False}), flush=True)
                previous = missing
            time.sleep(30)
        gate_path = args.output / 'prerequisites.json'
        if gate_path.exists() and read(gate_path) != gates:
            raise ValueError('Previously completed core prerequisite changed')
        if not gate_path.exists():
            write(gate_path, gates)
        rows, plain, _, _ = inputs(args)
        bank_path, raw_path = args.output / 'train.jsonl', args.output / 'raw-generations.jsonl'
        saved, raw = protocol.strict_records(bank_path), protocol.strict_records(raw_path)
        trees, reviews, known, _ = restore(rows, plain, saved, raw)
        bank_stream = stack.enter_context(bank_path.open('a'))
        raw_stream = stack.enter_context(raw_path.open('a'))
        for stream in (bank_stream, raw_stream):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # No cloud client, model tag lookup, or generation exists before all gates.
        import modal
        worker = modal.Cls.from_name(protocol.APP, 'Ollama')()
        worker.request.autoscaler.update(max_containers=2)
        request = RequestJournal(protocol.resilient(worker, args.output / 'transport-errors.jsonl'), raw_stream, raw, slots=WORKERS)
        tags = {item['name']: item['digest'] for item in request('/api/tags')['models']}
        if tags.get(protocol.old.MODEL) != record['solver_digest'] or tags.get(protocol.old.JUDGE) != record['reviewer_digest']:
            raise ValueError('Frozen compact/reviewer model tags changed')
        collector = PairCollector(rows, trees, reviews, known, bank_stream, request)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(collector.one, row) for row in rows]
            try:
                for count, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if count % 16 == 0:
                        print(f'Science expansion: {count}/192 questions, {len(known)}/960 records', flush=True)
            except BaseException:
                collector.stopped.set()
                for future in futures:
                    future.cancel()
                # Join running calls so their returned raw responses are durable
                # before any stream closes; do not start the remaining queue.
                raise
        expected = {(row['id'], kind, path) for row in rows for kind, path in PATHS}
        if set(known) != expected:
            raise ValueError('Incomplete fixed all192-question counterfactual grid')
        final_records, final_raw = protocol.strict_records(bank_path), protocol.strict_records(raw_path)
        _, _, _, coverage = restore(rows, plain, final_records, final_raw)
        verify_sources(record)
        current_gates, pending = prerequisite_receipts(args, record)
        if pending or current_gates != gates:
            raise ValueError('Core prerequisite changed during expansion')
        if inputs(args)[2] != record['selection']:
            raise ValueError('TRAIN selection changed during expansion')
        digests = {'train.jsonl': sha(bank_path), 'raw-generations.jsonl': sha(raw_path),
                   'registration.json': sha(path), 'prerequisites.json': sha(gate_path)}
        if (args.output / 'transport-errors.jsonl').exists():
            digests['transport-errors.jsonl'] = sha(args.output / 'transport-errors.jsonl')
        write(args.output / 'complete.json', {'schema': SCHEMA, 'questions': 192, 'records': 960,
            'source_matched_total_pairs': 320, 'registration_sha256': args.registration_sha256,
            'bank_sha256': sha(bank_path), 'digests': digests, 'raw_coverage': coverage,
            'unique_physical_generations': len(request.known), 'references_supplied': False,
            'training_labels_scored': False, 'development_or_test_values_decoded': False,
            'neural_calls': 0, 'outcome_filtering': False, 'fitting_contract': record['fitting_contract']})
        print(json.dumps({'complete': True, 'questions': 192, 'records': 960,
                          'physical_generations': len(request.known), 'labels_scored': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'collect'))
    for name in ('data', 'parent-bank', 'proposal', 'b-reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--proposal-sha256', required=True)
    parser.add_argument('--registration-sha256')
    parser.add_argument('--priority-completion', action='append', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    (register if args.phase == 'register' else collect)(args)


if __name__ == '__main__':
    main()
