"""Registered four-question prospective runner; frozen policy semantics unchanged.

No work starts on import. Register after a qualifying policy freeze, audit all
development decisions through this runner, then explicitly collect and grade.
The legacy collector, audit, freeze, and grader remain unmodified.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time

import prospective_collect as serial
from prospective_freeze import sha, verify, load_models

SCHEMA = 'round2-prospective-parallel-v1'
WORKERS = 4
CLOUD_CONTAINERS = 2
CLOUD_INPUTS = 4
METHODS = serial.METHODS
STORES = serial.STORES
HERE = Path(__file__).resolve().parent


def durable_line(stream, value):
    stream.write(json.dumps(value) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


class KeyLocks:
    """Locks live for the run, so waiters never split into two lock generations."""
    def __init__(self):
        self.guard = threading.Lock()
        self.locks = {}

    def get(self, key):
        with self.guard:
            return self.locks.setdefault(key, threading.RLock())


class ResponseStore(serial.ResponseStore):
    """Snapshot reads and durable check/write/publication share one short lock."""
    def __init__(self, streams=None, records=None):
        self.lock = threading.RLock()
        super().__init__(streams, records)

    def append(self, name, qid, kind, path, node, reused=None):
        with self.lock:
            return super().append(name, qid, kind, path, node, reused)

    def view(self, qid, name):
        with self.lock:
            return super().view(qid, name)


class Acquisition(serial.Acquisition):
    """Singleflight per canonical logical node, including recursive dependencies."""
    def __init__(self, store, request=None):
        super().__init__(store, request)
        self.keys = KeyLocks()

    def ensure(self, row, name, kind, path):
        if kind == 'solver' and path in serial.PLAIN:
            name = 'plain'
        with self.keys.get((row['id'], name, kind, path)):
            return super().ensure(row, name, kind, path)


class RequestJournal:
    """Singleflight exact payloads; persist every returned generation before parse.

Transport timeouts can leave a server-side attempt unresolved. The inherited
transport retry journal records those attempts; this is not a claim of remote
exactly-once execution. Completed, durable responses are never quality-retried.
"""
    def __init__(self, request, stream, records=(), slots=CLOUD_INPUTS):
        self.request, self.stream = request, stream
        self.keys, self.lock = KeyLocks(), threading.Lock()
        self.slots = threading.BoundedSemaphore(slots)
        self.known, self.failed = {}, {}
        for record in records:
            self.validate(record)
            key = record['request_sha256']
            if key in self.known:
                raise ValueError('Duplicate completed physical request receipt')
            self.known[key] = record

    @staticmethod
    def validate(record):
        if (record['endpoint'] != '/api/generate'
                or record['request_sha256'] != serial.fingerprint({
                    'endpoint': record['endpoint'], 'payload': record['payload']})
                or record['response_sha256'] != serial.fingerprint(record['response'])):
            raise ValueError('Physical request receipt identity changed')

    def __call__(self, endpoint, payload=None):
        # Model-tag checks must be live on each invocation/resume, not cached.
        if endpoint != '/api/generate':
            with self.slots:
                return self.request(endpoint, payload)
        key = serial.fingerprint({'endpoint': endpoint, 'payload': payload})
        with self.keys.get(key):
            with self.lock:
                previous = self.known.get(key)
                failure = self.failed.get(key)
            if failure is not None:
                raise RuntimeError('Physical request failed in this process; stop and inspect/resume receipts') from failure
            if previous is not None:
                # Parsing must not mutate a raw stored response or its identity.
                return json.loads(json.dumps(previous['response']))
            try:
                started = time.time()
                with self.slots:
                    response = self.request(endpoint, payload)
                record = {'endpoint': endpoint, 'payload': payload, 'request_sha256': key,
                          'response': response, 'response_sha256': serial.fingerprint(response),
                          'started_unix': started, 'returned_unix': time.time()}
                with self.lock:
                    durable_line(self.stream, record)
                    self.known[key] = record
            except BaseException as error:
                # In particular, a waiter cannot regenerate a returned answer
                # after its durable append failed. A partial tail fails closed
                # on restart; a complete tail can be read without another call.
                with self.lock:
                    self.failed[key] = error
                raise
            return json.loads(json.dumps(response))


class DecisionLedger:
    def __init__(self, stream=None, records=()):
        self.stream, self.lock, self.known = stream, threading.RLock(), {}
        for record in records:
            key = record['id'], record['method']
            if key in self.known:
                raise ValueError('Duplicate completed policy decision')
            self.known[key] = record

    def contains(self, key):
        with self.lock:
            return key in self.known

    def append(self, record):
        key = record['id'], record['method']
        with self.lock:
            if key in self.known:
                raise ValueError('Cannot replace a completed policy decision')
            if self.stream is not None:
                durable_line(self.stream, record)
            self.known[key] = record

    def snapshot(self):
        with self.lock:
            return dict(self.known)


def raw_coverage(banks, raw_records):
    """Bind each logical node/physical review phase to its persisted API result.

This uses response metadata only, never question references or a model. It also
fails before a resumed call if raw receipts for earlier logical nodes are lost.
"""
    index = {}
    for record in raw_records:
        RequestJournal.validate(record)
        payload, response = record['payload'], record['response']
        if not response.get('done'):
            continue
        options = payload['options']
        if response['model'] != payload['model']:
            raise ValueError('Raw generation returned a different model')
        key = (payload['model'], serial.hashlib.sha256(payload['prompt'].encode()).hexdigest(),
               options['seed'], options['temperature'], options['num_predict'], options['num_ctx'],
               response['response'], int(response['prompt_eval_count']), int(response.get('eval_count', 0)))
        index.setdefault(key, {})[record['request_sha256']] = response
    used, logical, physical = set(), 0, 0
    for name, records in banks.items():
        for record in records:
            logical += 1
            node, kind, path = record['response'], record['kind'], record['path']
            strong_review = name == 'strong' and kind == 'judge'
            phases = node['calls'] if strong_review else [node]
            expected_model = serial.old.JUDGE if kind == 'judge' else serial.old.MODEL
            default_cap = (512 if kind == 'judge' else 128 if not path
                           or (name == 'strong' and path not in serial.PLAIN and path != 'guided/retry') else 256)
            for phase in phases:
                physical += 1
                key = (node.get('model', expected_model), phase['prompt_sha256'], phase['seed'],
                       phase['temperature'], phase.get('output_cap', default_cap),
                       phase.get('context_length', 8192 if strong_review else 4096),
                       phase['text'], phase['input_tokens'], phase['output_tokens'])
                matching = {digest for digest, response in index.get(key, {}).items()
                            if (not strong_review or response.get('thinking', '') == phase.get('thinking', ''))
                            and ('finish' not in phase or response.get('done_reason') == phase['finish'])}
                if not matching:
                    raise ValueError('Logical response or review phase has no matching durable raw receipt')
                used.update(matching)
    return {'logical_response_nodes': logical, 'physical_response_nodes_including_exact_reuse': physical,
            'unique_raw_generations_covered': len(used), 'raw_generations_stored': len(raw_records)}


def run_questions(rows, make, ledger, workers=WORKERS):
    """One owner per qid; all methods sequential; failed runs never get a seal."""
    if workers not in (1, WORKERS):
        raise ValueError('Use one serial reference worker or four registered workers')
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids) or any(row.get('answer') != '' for row in rows):
        raise ValueError('Question ownership requires unique public-only rows')
    stopped = threading.Event()

    def question(row):
        for method in METHODS:
            if stopped.is_set():
                return
            if not ledger.contains((row['id'], method)):
                ledger.append(make(row, method))

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='prospective-q') as pool:
        futures = [pool.submit(question, row) for row in rows]
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 16 == 0:
                    print(f'Blind replay: {completed}/{len(rows)} questions; '
                          f'{len(ledger.snapshot())}/{len(rows) * len(METHODS)} decisions', flush=True)
        except BaseException:
            stopped.set()
            for future in futures:
                future.cancel()
            # Exiting the executor joins already-running calls before streams
            # close. Their returned raw receipts and decisions stay resumable.
            raise
    result = ledger.snapshot()
    if set(result) != {(qid, method) for qid in ids for method in METHODS}:
        raise ValueError('Incomplete prospective policy grid')
    return result


def make_decision(frozen, models, method, row, store, policy_lock, acquisition=None):
    """Same paid-prefix computation as serial.make_decision, with scoped NN lock."""
    if row.get('answer') != '':
        raise ValueError('Parallel decisions require blank reference answers')
    name = serial.method_store(frozen, method)
    model = models.get(method)
    function = serial.dispatch(frozen, models, method)
    if acquisition is not None:
        acquisition.ensure(row, 'plain', 'solver', '')
    for _ in range(8):
        tree, reviews = store.view(row['id'], name)
        try:
            # Includes observer loading, monkeypatch installation/restoration,
            # model/cache access, and every paid-prefix replay. Never a remote call.
            with policy_lock:
                item, ledger = serial.replay_paid_inputs(function, row, tree, reviews,
                    method if method in ('A', 'B', 'C') else 'baseline', model)
            break
        except serial.MissingObservation as missing:
            if acquisition is None:
                raise
            # The lock has unwound, including encoder_ledger's finally block.
            acquisition.ensure(row, name, missing.kind, missing.path)
    else:
        raise ValueError('Policy exceeded finite acquisition horizon')
    paid_tree = serial.prefix(tree, item['path'])
    paid_reviews = {(row['id'], d['path']): reviews[row['id'], d['path']]
                    for d in item.get('decisions', []) if d['queried']}
    for path, node in paid_tree.items():
        serial.validate_node(row, 'plain' if path in serial.PLAIN else name,
                             'solver', path, node, paid_tree, paid_reviews)
    for (_, path), node in paid_reviews.items():
        serial.validate_node(row, name, 'judge', path, node, paid_tree, paid_reviews)
    serial.validate_encoder_ledger(frozen, item, ledger)
    audited = serial.audit_costs(item, paid_tree, paid_reviews, ledger,
                                method if method in ('A', 'B', 'C') else 'baseline')
    item['judge_physical_calls'] = audited['physical_review_calls']
    receipts = [{'bank': 'plain' if path in serial.PLAIN else name, 'kind': 'solver',
                 'path': path, 'sha256': serial.fingerprint(node)} for path, node in paid_tree.items()]
    receipts += [{'bank': name, 'kind': 'judge', 'path': path, 'sha256': serial.fingerprint(node)}
                 for (_, path), node in paid_reviews.items()]
    return {'id': row['id'], 'method': method, 'bank': name, 'subject': row['subject'],
            'source': serial.source_group(row), 'item': item, 'neural_receipts': ledger,
            'paid_receipts': receipts,
            'final_response_sha256': serial.fingerprint(paid_tree[item['path']]),
            'blind_paid_prefix_audit': audited}


def protocol_payload(frozen, study, freeze_sha):
    root = Path(frozen['project_directory'])
    paths = [Path(__file__).resolve(), HERE / 'prospective_collect.py',
             HERE / 'prospective_audit.py', HERE / 'prospective_grade.py',
             HERE / 'prospective_freeze.py']
    sources = {}
    for path in paths:
        relative = str(path.relative_to(root))
        digest = sha(path)
        if frozen['runtime_sources'].get(relative) != digest or sha(study / 'runtime' / relative) != digest:
            raise ValueError('Parallel runner and dependencies must be archived in the original policy freeze')
        sources[relative] = digest
    return {'schema': SCHEMA, 'freeze_sha256': freeze_sha, 'source_sha256': sources,
            'question_workers': WORKERS, 'max_cloud_containers': CLOUD_CONTAINERS,
            'max_cloud_inputs': CLOUD_INPUTS, 'methods_in_question_order': list(METHODS),
            'question_ownership': 'One worker per question, all eight methods sequential; no speculative calls.',
            'observer_lock': 'Global lock covers complete replay_paid_inputs including encoder ledger; released before acquisition.',
            'physical_cache': 'Singleflight exact generate payload; fsync raw returned response before parsing; no quality retries.',
            'raw_coverage': 'Every logical node and strong-review physical phase must match a durable raw prompt/seed/model/text/token receipt before resume and completion.',
            'transport_uncertainty': 'Existing infrastructure-only retries retained; unresolved remote attempts journaled, not claimed exactly once.',
            'paid_compute': 'Unchanged per-policy paid-prefix and physical encoder ledger; cache hits still charged.',
            'development_route': 'prospective_parallel.py audit-development: all256x8 decisions and selected aggregate equality.',
            'collection_route': 'prospective_parallel.py collect: all1024x8 before completion seal.',
            'grading_route': 'prospective_parallel.py grade: protocol/raw-receipt guards then unchanged prospective_grade.run.',
            'test_records_read_at_registration': False}


def register(args):
    frozen = verify(args.study, args.freeze_sha256)
    data = Path(frozen['data_directory'])
    if ((data / 'test-collection-claim.json').exists() or (args.study / 'collection').exists()
            or (args.study / 'development-audit.json').exists()):
        raise ValueError('Register the parallel route before any collection claim or development replay')
    directory = args.study / 'parallel-protocol'
    if directory.exists():
        raise ValueError('Parallel registration already exists; do not overwrite')
    payload = protocol_payload(frozen, args.study, args.freeze_sha256)
    directory.mkdir()
    # The complete dependency surface is already preserved by prospective_freeze.
    # Keep this explicit entrypoint copy as an independently checked receipt too.
    shutil.copyfile(__file__, directory / 'prospective_parallel.py')
    serial.atomic_json(directory / 'registration.json', payload)
    digest = sha(directory / 'registration.json')
    print(json.dumps({'protocol_sha256': digest, 'test_records_read': False}), flush=True)


def verify_protocol(args):
    frozen = verify(args.study, args.freeze_sha256)
    directory = args.study / 'parallel-protocol'
    if not args.protocol_sha256 or sha(directory / 'registration.json') != args.protocol_sha256:
        raise ValueError('Parallel protocol differs from the externally recorded digest')
    payload = json.loads((directory / 'registration.json').read_bytes())
    if payload != protocol_payload(frozen, args.study, args.freeze_sha256):
        raise ValueError('Parallel runtime or concurrency contract changed')
    if sha(directory / 'prospective_parallel.py') != sha(__file__):
        raise ValueError('Archived parallel entrypoint changed')
    return frozen


def audit_development(args):
    frozen = verify_protocol(args)
    if (Path(frozen['data_directory']) / 'test-collection-claim.json').exists():
        raise ValueError('Development route must finish before claiming the test panel')
    path = args.study / 'development-audit.json'
    with (args.study / 'parallel-development.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or (args.study / 'development-replay.json').exists():
            raise ValueError('Development audit exists; never replace an earlier replay')
        frozen, models = load_models(args.study, args.device, args.freeze_sha256)
        full, stores = None, {}
        for method in METHODS:
            rows, store = serial.development_store(Path(frozen['data_directory']), frozen, method)
            if len(rows) != 256 or len({r['id'] for r in rows}) != 256:
                raise ValueError('Incomplete development panel')
            by_id = {row['id']: row for row in rows}
            if full is not None and by_id != full:
                raise ValueError('Methods disagree on the full development panel')
            full, stores[method] = by_id, store
        policy_lock = threading.Lock()
        def make(row, method):
            return make_decision(frozen, models, method, row, stores[method], policy_lock)
        records = run_questions([serial.public_row(row) for row in full.values()], make, DecisionLedger())
        reports = {}
        from scoring import correct
        for method in METHODS:
            values = []
            for qid, row in full.items():
                record = records[qid, method]
                final = serial.verify_decision(frozen, record, stores[method], serial.public_row(row))
                values.append({**record['item'], 'correct': int(correct(final, row['answer']))})
            measured = serial.statistics(values)
            measured['mean_judge_physical_calls'] = sum(v['judge_physical_calls'] for v in values) / len(values)
            expected = frozen['development_baselines'][method] if method in serial.BASELINES else frozen['selected'][method]['dev']
            if any(not math.isclose(measured[k], v, rel_tol=1e-7, abs_tol=1e-7) for k, v in expected.items()):
                raise ValueError('Parallel archived replay differs from selected metrics: ' + method)
            reports[method] = measured
        verify_protocol(args)
        replay = args.study / 'development-replay.json'
        serial.atomic_json(replay, [records[qid, method] for qid in full for method in METHODS])
        serial.atomic_json(path, {'passed': True, 'freeze_sha256': args.freeze_sha256,
            'parallel_protocol_sha256': args.protocol_sha256, 'questions': 256,
            'method_question_audits': len(records), 'methods': reports, 'test_used': False,
            'decision_receipts_sha256': sha(replay), 'source_sha256': sha(__file__),
            'observer_work_measured_from_actual_inputs': True, 'question_workers': WORKERS})
        print('All2048 parallel development paid-prefix audits passed; test records unopened.', flush=True)


def verify_development(args):
    audit = json.loads((args.study / 'development-audit.json').read_bytes())
    if (not audit['passed'] or audit['freeze_sha256'] != args.freeze_sha256
            or audit.get('parallel_protocol_sha256') != args.protocol_sha256
            or audit['method_question_audits'] != 2048 or audit['questions'] != 256
            or audit.get('question_workers') != WORKERS or audit['source_sha256'] != sha(__file__)
            or audit['decision_receipts_sha256'] != sha(args.study / 'development-replay.json')):
        raise ValueError('Finish the registered four-worker full development audit first')
    return audit


def collect(args):
    import modal
    frozen = verify_protocol(args)
    verify_development(args)
    data, target = Path(frozen['data_directory']), args.study / 'collection'
    with ExitStack() as stack:
        lock = stack.enter_context((args.study / 'collection.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target.mkdir(exist_ok=True)
        if (target / 'complete.json').exists():
            raise ValueError('Collection complete; never regenerate')
        claim = {'freeze_sha256': args.freeze_sha256, 'parallel_protocol_sha256': args.protocol_sha256,
                 'question_sha256': frozen['reserved_question_sha256'], 'study': str(args.study.resolve()),
                 'rules': 'All1024questions; no reference-conditioned calls or policy changes.'}
        claim_path = data / 'test-collection-claim.json'
        with (data / 'test-collection.lock').open('a') as claim_lock:
            fcntl.flock(claim_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if claim_path.exists() and json.loads(claim_path.read_bytes()) != claim:
                raise ValueError('Reserved panel already claimed by another freeze or runner')
            if not claim_path.exists():
                serial.atomic_json(claim_path, claim)
        # This is the first individual test-question read. The parser skips references.
        rows = serial.read_public_rows(data / 'splits/test-questions.jsonl')
        by_id = {row['id']: row for row in rows}
        if len(rows) != 1024 or len(by_id) != 1024:
            raise ValueError('Preserve all1024 unique reserved questions')
        frozen, models = load_models(args.study, args.device, args.freeze_sha256)
        streams, records = {}, {}
        for name in STORES:
            path = target / (name + '.jsonl')
            records[name] = serial.strict_records(path)
            stream = stack.enter_context(path.open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            streams[name] = stream
        store = ResponseStore(streams, records)
        if any(qid not in by_id for bank in store.known.values() for qid, _, _ in bank):
            raise ValueError('Response bank contains nonreserved questions')
        saved_path = target / 'ungraded-decisions.jsonl'
        saved = serial.strict_records(saved_path)
        for record in saved:
            if (record['freeze_sha256'] != args.freeze_sha256
                    or record.get('parallel_protocol_sha256') != args.protocol_sha256
                    or record['id'] not in by_id or record['method'] not in METHODS):
                raise ValueError('Unexpected or differently registered decision')
            serial.verify_decision(frozen, record, store, by_id[record['id']])
        output = stack.enter_context(saved_path.open('a'))
        fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = DecisionLedger(output, saved)
        raw_path = target / 'raw-generations.jsonl'
        raw_records = serial.strict_records(raw_path)
        raw_coverage(records, raw_records)
        raw_stream = stack.enter_context(raw_path.open('a'))
        fcntl.flock(raw_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = modal.Cls.from_name(serial.APP, 'Ollama')()
        worker.request.autoscaler.update(max_containers=CLOUD_CONTAINERS)
        request = RequestJournal(serial.resilient(worker, target / 'transport-errors.jsonl'), raw_stream, raw_records)
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        if tags.get(serial.old.MODEL) != frozen['solver_digest'] or tags.get(serial.old.JUDGE) != frozen['reviewer_digest']:
            raise ValueError('Inference model weights changed')
        # All old.call invocations see this same callable; never install worker-specific globals.
        acquisition, policy_lock = Acquisition(store, request), threading.Lock()
        def make(row, method):
            return {'freeze_sha256': args.freeze_sha256, 'parallel_protocol_sha256': args.protocol_sha256,
                    **make_decision(frozen, models, method, row, store, policy_lock, acquisition)}
        known = run_questions(rows, make, ledger)
        for record in known.values():
            serial.verify_decision(frozen, record, store, by_id[record['id']])
        verify_protocol(args)
        verify_development(args)
        coverage = raw_coverage({name: serial.strict_records(target / (name + '.jsonl')) for name in STORES},
                                serial.strict_records(raw_path))
        digests = {name + '.jsonl': sha(target / (name + '.jsonl')) for name in STORES}
        digests['ungraded-decisions.jsonl'] = sha(saved_path)
        auxiliary = {'raw-generations.jsonl': sha(raw_path)}
        if (target / 'transport-errors.jsonl').exists():
            auxiliary['transport-errors.jsonl'] = sha(target / 'transport-errors.jsonl')
        audit = {'passed': True, 'questions': 1024, 'method_question_audits': len(known),
                 'freeze_sha256': args.freeze_sha256, 'parallel_protocol_sha256': args.protocol_sha256,
                 'digests': digests, 'parallel_artifact_digests': auxiliary,
                 'completed_unique_physical_generations': len(request.known),
                 'raw_coverage': coverage,
                 'test_references_supplied_to_policies': False, 'grades_opened': False,
                 'source_sha256': sha(__file__), 'question_workers': WORKERS,
                 'max_cloud_containers': CLOUD_CONTAINERS, 'max_cloud_inputs': CLOUD_INPUTS}
        serial.atomic_json(target / 'collection-audit.json', audit)
        serial.atomic_json(target / 'complete.json', {**audit, 'methods': list(METHODS),
            'question_ids_sha256': serial.fingerprint(sorted(by_id)),
            'audit_sha256': sha(target / 'collection-audit.json'),
            'development_audit_sha256': sha(args.study / 'development-audit.json'),
            'required_grading_route': 'prospective_parallel.py grade'})
        print('All8192 parallel decisions complete and audited; grades remain closed.', flush=True)


def grade(args):
    verify_protocol(args)
    verify_development(args)
    target = args.study / 'collection'
    complete = json.loads((target / 'complete.json').read_bytes())
    audit = json.loads((target / 'collection-audit.json').read_bytes())
    if (complete.get('parallel_protocol_sha256') != args.protocol_sha256
            or audit.get('parallel_protocol_sha256') != args.protocol_sha256
            or complete.get('parallel_artifact_digests') != audit.get('parallel_artifact_digests')
            or complete.get('source_sha256') != sha(__file__)
            or complete.get('question_workers') != WORKERS):
        raise ValueError('Parallel completion protocol changed')
    extras = complete['parallel_artifact_digests']
    if set(extras) not in ({'raw-generations.jsonl'}, {'raw-generations.jsonl', 'transport-errors.jsonl'}):
        raise ValueError('Missing or unexpected parallel completion artifacts')
    for name, expected in extras.items():
        if sha(target / name) != expected:
            raise ValueError('Parallel raw/transport receipt changed')
    raw = serial.strict_records(target / 'raw-generations.jsonl')
    for record in raw:
        RequestJournal.validate(record)
    if (len({r['request_sha256'] for r in raw}) != len(raw)
            or len(raw) != complete['completed_unique_physical_generations']):
        raise ValueError('Duplicate or missing completed physical receipt')
    coverage = raw_coverage({name: serial.strict_records(target / (name + '.jsonl')) for name in STORES}, raw)
    if complete.get('raw_coverage') != coverage or audit.get('raw_coverage') != coverage:
        raise ValueError('Raw-to-logical response coverage changed')
    for record in serial.strict_records(target / 'ungraded-decisions.jsonl'):
        if record.get('parallel_protocol_sha256') != args.protocol_sha256:
            raise ValueError('Decision from another parallel registration')
    # The unchanged grader checks the full1024x8 grid, every independent paid
    # prefix, all frozen sources/assets, and only then parses any test reference.
    import prospective_grade
    prospective_grade.run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'audit-development', 'collect', 'grade'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--protocol-sha256', help='External digest printed by register; required after registration.')
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    if args.phase != 'register' and not args.protocol_sha256:
        parser.error('--protocol-sha256 is required after registration')
    globals()[args.phase.replace('-', '_')](args)


if __name__ == '__main__':
    main()
