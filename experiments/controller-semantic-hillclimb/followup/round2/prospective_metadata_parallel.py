"""Explicit metadata-C route around the unchanged four-question runner.

Register after a qualifying freeze; audit all256 development questions, then
collect all1024 reserved questions blind and grade once. No work on import.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import threading
from unittest.mock import patch

import prospective_parallel as parallel
import prospective_collect as serial
import prospective_grade as grader
import prospective_metadata_freeze as frozen_metadata
import metadata_paid_audit as metadata_audit
import metadata_review_hybrid as policy

HERE = Path(__file__).resolve().parent
SCHEMA = 'round2-prospective-metadata-parallel-v1'
sha = parallel.sha
ORIGINAL_DISPATCH = serial.dispatch
ORIGINAL_DECISION = parallel.make_decision
ORIGINAL_VERIFY_DECISION = serial.verify_decision
ORIGINAL_PAYLOAD = parallel.protocol_payload
ORIGINAL_STATISTICS = grader.statistics
ORIGINAL_ATOMIC_JSON = serial.atomic_json


def adapter(frozen):
    record = frozen['completed_A_B_registry']['metadata_C_provenance']['selected_adapter']
    metadata_audit.validate_adapter_state(record['state'], record['receipt'])
    return record


def dispatch(frozen, models, method):
    if method != 'C':
        return ORIGINAL_DISPATCH(frozen, models, method)
    config = frozen['selected']['C']
    if config['module'] != frozen_metadata.MODULE or type(models['C']) is not policy.Planner:
        raise ValueError('Metadata route requires explicit metadata_review_hybrid dispatch')
    expected = adapter(frozen)
    if models['C'].metadata_adapter.state() != expected['state'] or models['C'].metadata_adapter.receipt != expected['receipt']:
        raise ValueError('Selected metadata runtime learned state changed')
    return lambda row, tree, reviews: policy.evaluate(row, tree, reviews, models['C'], config['settings'], 'C')


def audit_record(frozen, record, store):
    item = record['item']
    if record['method'] != 'C':
        if item.get('metadata_calls', 0) or item.get('metadata_observations', []):
            raise ValueError('Only C may consult the metadata correctness adapter')
        return None
    tree, reviews = store.view(record['id'], record['bank'])
    tree = serial.prefix(tree, item['path'])
    reviews = {(record['id'], d['path']): reviews[record['id'], d['path']]
               for d in item['decisions'] if d['queried']}
    saved = adapter(frozen)
    return metadata_audit.audit_metadata(item, tree, reviews, record['neural_receipts'],
                                         saved['state'], saved['receipt'], record['subject'])


def make_decision(frozen, models, method, row, store, policy_lock, acquisition=None):
    record = ORIGINAL_DECISION(frozen, models, method, row, store, policy_lock, acquisition)
    if method == 'C':
        record['metadata_receipts_sha256'] = serial.fingerprint(record['item']['metadata_observations'])
        record['metadata_paid_audit'] = audit_record(frozen, record, store)
    else:
        audit_record(frozen, record, store)
    return record


def verify_decision(frozen, record, store, row=None):
    result = ORIGINAL_VERIFY_DECISION(frozen, record, store, row)
    evidence = audit_record(frozen, record, store)
    if record['method'] == 'C':
        if (record.get('metadata_receipts_sha256') != serial.fingerprint(record['item']['metadata_observations'])
                or record.get('metadata_paid_audit') != evidence):
            raise ValueError('Saved paid-metadata input/probability/timing receipt changed')
    return result


def statistics(values):
    result = ORIGINAL_STATISTICS(values)
    count = len(values)
    result.update(mean_metadata_calls=sum(v.get('metadata_calls', 0) for v in values)/count,
                  mean_metadata_cpu_ms=1000*sum(v.get('metadata_cpu_seconds', 0.) for v in values)/count,
                  mean_review_neural_calls=sum(v.get('review_neural_calls', 0) for v in values)/count,
                  mean_judge_physical_calls=sum(v['judge_physical_calls'] for v in values)/count)
    return result


def compare_development(measured, expected):
    """All deterministic outputs/costs match; measured wall time is not replayable."""
    for key, value in expected.items():
        if key == 'mean_metadata_cpu_ms':
            if not math.isfinite(measured[key]) or measured[key] < 0:
                raise ValueError('Invalid independently measured metadata CPU time')
            continue
        if key not in measured or not math.isclose(measured[key], value, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError('Metadata development replay changed deterministic metric: ' + key)


def protocol_payload(frozen, study, freeze_sha):
    payload = ORIGINAL_PAYLOAD(frozen, study, freeze_sha)
    if frozen['selected']['C']['module'] != frozen_metadata.MODULE:
        raise ValueError('This runner supports explicitly declared metadata C only')
    root = Path(frozen['project_directory']); sources = dict(payload['source_sha256'])
    for module in (frozen_metadata, metadata_audit, policy):
        path = Path(module.__file__).resolve(); relative = str(path.relative_to(root)); digest = sha(path)
        if frozen['runtime_sources'].get(relative) != digest or sha(study/'runtime'/relative) != digest:
            raise ValueError('Metadata dependency was not archived in the original freeze')
        sources[relative] = digest
    path = Path(__file__).resolve(); relative = str(path.relative_to(root)); digest = sha(path)
    if frozen['runtime_sources'].get(relative) != digest or sha(study/'runtime'/relative) != digest:
        raise ValueError('Metadata runner was not archived in the original freeze')
    sources[relative] = digest
    return {**payload, 'schema': SCHEMA, 'source_sha256': sources,
            'metadata_adapter_sha256': adapter(frozen)['receipt']['learned_state_sha256'],
            'metadata_paid_audit': 'Hash paid root/review inputs,32features,full learned state and measured CPU timing; independently recompute probability.',
            'review_encoder': 'Absent only in metadata C; actual solver/child component passes remain charged.',
            'development_route': 'prospective_metadata_parallel.py audit-development: all256x8 decisions; deterministic metrics equal, CPU wall time measured anew.',
            'collection_route': 'prospective_metadata_parallel.py collect: all1024x8 before completion seal.',
            'grading_route': 'prospective_metadata_parallel.py grade: metadata/raw/protocol audits before unchanged paired grading.'}


def atomic_json(path, value):
    if Path(path).name in ('collection-audit.json', 'complete.json') and Path(path).parent.name == 'collection':
        value = {**value, 'metadata_runner_sha256': sha(__file__), 'metadata_audit_sha256': sha(metadata_audit.__file__),
                 'metadata_receipts_audited': True, 'required_grading_route': 'prospective_metadata_parallel.py grade'}
    ORIGINAL_ATOMIC_JSON(path, value)


def audit_development(args):
    frozen = parallel.verify_protocol(args)
    if (Path(frozen['data_directory'])/'test-collection-claim.json').exists():
        raise ValueError('Audit development before claiming reserved questions')
    destination = args.study/'development-audit.json'
    with (args.study/'parallel-development.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if destination.exists() or (args.study/'development-replay.json').exists():
            raise ValueError('Never replace an earlier development audit')
        frozen, models = frozen_metadata.load_models(args.study, args.device, args.freeze_sha256)
        full, stores = None, {}
        for method in parallel.METHODS:
            rows, store = serial.development_store(Path(frozen['data_directory']), frozen, method)
            by_id = {r['id']: r for r in rows}
            if len(rows) != 256 or len(by_id) != 256 or (full is not None and by_id != full):
                raise ValueError('Preserve every same-panel development question')
            full, stores[method] = by_id, store
        lock = threading.Lock()
        def make(row, method): return make_decision(frozen, models, method, row, stores[method], lock)
        records = parallel.run_questions([serial.public_row(r) for r in full.values()], make, parallel.DecisionLedger())
        reports = {}
        from scoring import correct
        for method in parallel.METHODS:
            values = []
            for qid, row in full.items():
                record = records[qid, method]
                final = verify_decision(frozen, record, stores[method], serial.public_row(row))
                values.append({**record['item'], 'correct': int(correct(final, row['answer']))})
            measured = statistics(values)
            expected = frozen['development_baselines'][method] if method in serial.BASELINES else frozen['selected'][method]['dev']
            compare_development(measured, expected); reports[method] = measured
        parallel.verify_protocol(args)
        replay = args.study/'development-replay.json'
        serial.atomic_json(replay, [records[qid, method] for qid in full for method in parallel.METHODS])
        serial.atomic_json(destination, {'passed': True, 'freeze_sha256': args.freeze_sha256,
            'parallel_protocol_sha256': args.protocol_sha256, 'questions': 256, 'method_question_audits': len(records),
            'methods': reports, 'test_used': False, 'decision_receipts_sha256': sha(replay),
            'source_sha256': sha(__file__), 'metadata_audit_sha256': sha(metadata_audit.__file__),
            'metadata_receipts_audited': True, 'observer_work_measured_from_actual_inputs': True,
            'question_workers': parallel.WORKERS,
            'timing_comparison': 'CPU wall time is remeasured and hash-sealed; every deterministic selected metric must match.'})
        print('All2048 metadata-aware development paid-prefix audits passed; test records unopened.', flush=True)


def verify_development(args):
    audit = json.loads((args.study/'development-audit.json').read_bytes())
    if (audit.get('passed') is not True or audit['freeze_sha256'] != args.freeze_sha256
            or audit.get('parallel_protocol_sha256') != args.protocol_sha256
            or audit.get('questions') != 256 or audit.get('method_question_audits') != 2048
            or audit.get('question_workers') != parallel.WORKERS or audit.get('source_sha256') != sha(__file__)
            or audit.get('metadata_audit_sha256') != sha(metadata_audit.__file__)
            or audit.get('metadata_receipts_audited') is not True
            or audit.get('decision_receipts_sha256') != sha(args.study/'development-replay.json')):
        raise ValueError('Complete the registered metadata-aware full development audit first')
    return audit


def grade(args):
    parallel.verify_protocol(args); verify_development(args)
    for name in ('complete.json', 'collection-audit.json'):
        record = json.loads((args.study/'collection'/name).read_bytes())
        if (record.get('metadata_runner_sha256') != sha(__file__)
                or record.get('metadata_audit_sha256') != sha(metadata_audit.__file__)
                or record.get('metadata_receipts_audited') is not True
                or record.get('required_grading_route') != 'prospective_metadata_parallel.py grade'):
            raise ValueError('Require metadata-aware collection and paid-prefix audit seals before grading')
    parallel.grade(args)


@contextmanager
def installed():
    # Scope spans the whole parallel stage, so every worker sees the same hooks.
    # Frozen runner still owns question/cache locks and releases NN lock before cloud.
    with patch.object(parallel, 'verify', frozen_metadata.verify), \
            patch.object(parallel, 'load_models', frozen_metadata.load_models), \
            patch.object(parallel, 'protocol_payload', protocol_payload), \
            patch.object(parallel, 'make_decision', make_decision), \
            patch.object(parallel, 'verify_development', verify_development), \
            patch.object(serial, 'dispatch', dispatch), \
            patch.object(serial, 'replay_paid_inputs', metadata_audit.replay_paid_inputs), \
            patch.object(serial, 'verify_decision', verify_decision), \
            patch.object(serial, 'atomic_json', atomic_json), \
            patch.object(grader, 'verify', frozen_metadata.verify), \
            patch.object(grader, 'verify_decision', verify_decision), \
            patch.object(grader, 'statistics', statistics):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register','audit-development','collect','grade'))
    parser.add_argument('--study', type=Path, required=True); parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--protocol-sha256'); parser.add_argument('--device', choices=('cpu','mps'), default='mps')
    args = parser.parse_args()
    if args.phase != 'register' and not args.protocol_sha256: parser.error('External protocol SHA is required')
    with installed():
        {'register': parallel.register, 'audit-development': audit_development,
         'collect': parallel.collect, 'grade': grade}[args.phase](args)


if __name__ == '__main__': main()
