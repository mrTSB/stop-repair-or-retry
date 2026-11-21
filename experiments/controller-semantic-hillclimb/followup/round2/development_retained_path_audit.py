"""Complete every retained-review path for all128 DEV science questions.

This is a hindsight availability bound, never a controller result. Registration
and collection skip reference values. Grade is a separate phase requiring the
entire128x10 science grid and its paid physical receipts to be sealed first.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
from pathlib import Path
import threading

import development_metadata_review as previous
import prospective_collect as serial
from prospective_parallel import RequestJournal

read, write, sha = previous.read, serial.atomic_json, previous.sha
SCHEMA = 'development-retained-path-availability-v1'
PLAIN = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
ADDITIONAL = ('guided', 'guided/reuse', 'retry/reuse')
PATHS = (*PLAIN, *ADDITIONAL)
SCIENCE = {'chemistry': 64, 'physics': 64}
SCOPE = 'Hindsight answer availability on existing development only; not controller accuracy, a learned policy, or fresh validation.'


def plain_prefix(data):
    complete = read(data / 'dev-plain-complete.json')
    if complete['questions'] != 256 or complete['base_records'] != 1792:
        raise ValueError('Require the full original256x7 plain development grid')
    with (data / 'dev-observations.jsonl').open('rb') as stream:
        prefix = b''.join(next(stream) for _ in range(complete['base_records']))
    if (not prefix.endswith(b'\n') or hashlib.sha256(prefix).hexdigest() != complete['base_bank_sha256']):
        raise ValueError('Original plain development prefix changed')
    return prefix, complete


def public_view(data, strong_records):
    rows = serial.read_public_rows(data / 'splits/dev-questions.jsonl')
    if (len(rows) != 256 or len({r['id'] for r in rows}) != 256
            or Counter(r['subject'] for r in rows) != {'math': 128, **SCIENCE}
            or any(r['answer'] != '' for r in rows)):
        raise ValueError('Require the complete public256development panel with blank references')
    by_id = {r['id']: r for r in rows}; trees = {r['id']: {} for r in rows}; judges = {}
    prefix, _ = plain_prefix(data)
    for line in prefix.splitlines():
        record = json.loads(line); qid, path = record['id'], record['path']
        if (qid not in by_id or record['kind'] != 'solver' or path not in PLAIN or path in trees[qid]):
            raise ValueError('Unexpected or duplicate original plain response')
        trees[qid][path] = record['response']
    for row in rows:
        tree = trees[row['id']]
        if set(tree) != set(PLAIN):
            raise ValueError('Incomplete original seven-path controls')
        for path, node in tree.items():
            serial.validate_node(row, 'plain', 'solver', path, node, tree, {})
    seen = set()
    for record in strong_records:
        qid, kind, path = record['id'], record['kind'], record['path']
        if qid not in by_id or (qid, kind, path) in seen:
            raise ValueError('Unexpected or duplicate retained-review receipt')
        seen.add((qid, kind, path))
        if kind == 'judge' and path == '':
            judges[qid, path] = record['response']
        elif kind == 'solver' and path in (*ADDITIONAL, 'guided/retry'):
            trees[qid][path] = record['response']
        else:
            raise ValueError('Unknown retained-review path')
    for record in strong_records:
        qid, kind, path = record['id'], record['kind'], record['path']
        serial.validate_node(by_id[qid], 'strong', kind, path, record['response'], trees[qid], judges)
    previous.validate_nodes(rows, trees, judges, strong_records)
    science = [row for row in rows if row['subject'] in SCIENCE]
    if any((row['id'], '') not in judges for row in science):
        raise ValueError('Every science root review must already exist; no new teacher calls are permitted')
    return rows, trees, judges


def missing_plan(rows, trees):
    science = [r for r in rows if r['subject'] in SCIENCE]
    if Counter(r['subject'] for r in science) != SCIENCE:
        raise ValueError('Always retain all128science questions, regardless of outcome')
    missing = [{'id': row['id'], 'subject': row['subject'], 'path': path}
               for row in science for path in ADDITIONAL if path not in trees[row['id']]]
    return {'questions': 128, 'subject_counts': SCIENCE, 'required_paths': list(PATHS),
        'required_question_path_cells': 1280, 'missing': missing,
        'missing_by_path': {p: sum(r['path'] == p for r in missing) for p in ADDITIONAL},
        'maximum_new_compact_calls': len(missing), 'new_teacher_calls': 0, 'new_neural_calls': 0,
        'selection_by_correctness': False}


def source_inputs(args):
    panel = sha(args.data / 'splits/dev-questions.jsonl')
    payload, receipt = previous.completed_bank(args.source, panel)
    if (read(args.source / 'cache-import.json')['schema'] != previous.SCHEMA
            or read(args.source / 'identity.json')['source_sha256'] != sha(previous.__file__)):
        raise ValueError('Require the final explicitly sealed metadata-newsource development area')
    metadata_identity = read(args.source_models / 'identity.json')
    if (metadata_identity['cohort'] != 'newsource'
            or read(args.source / 'identity.json')['models_sha256'] != sha(args.source_models / 'complete.json')):
        raise ValueError('The source bank/cache must be bound to the exact newsource metadata models')
    rows, trees, judges = public_view(args.data, [json.loads(line) for line in payload.splitlines()])
    cache = (args.source / 'private-neural-cache.pkl').read_bytes()
    previous.inherited.cache_values(cache)  # Validate, copy, and preserve; never run an encoder.
    raw = (args.source / 'raw-generations.jsonl').read_bytes()
    raw_records = serial.strict_records(args.source / 'raw-generations.jsonl')
    previous.coverage([json.loads(line) for line in payload.splitlines()], raw_records)
    b = read(args.b_reference)
    previous.inherited.validate_b(b, read(args.b_reference.parent / 'identity.json'), panel)
    n, correct = b['B']['dev']['n'], b['B']['dev']['correct']
    if n != 256 or not math.isclose(b['B']['dev']['accuracy'], correct / n, abs_tol=1e-12):
        raise ValueError('Changed same-panel strongest-B accuracy evidence')
    needed = math.ceil(correct + .1 * n - 1e-12)
    evidence = {'source': receipt, 'dev_questions_sha256': panel,
        'plain_completion_sha256': sha(args.data / 'dev-plain-complete.json'),
        'source_model_complete_sha256': sha(args.source_models / 'complete.json'),
        'source_bank_sha256': hashlib.sha256(payload).hexdigest(),
        'source_raw_sha256': hashlib.sha256(raw).hexdigest(),
        'source_raw_generations': len(raw_records),
        'source_cache_sha256': hashlib.sha256(cache).hexdigest(),
        'B_summary_sha256': sha(args.b_reference), 'B_correct': correct, 'B_n': n,
        'minimum_C_correct_above_current_B': needed}
    return rows, trees, judges, payload, cache, raw, evidence


def plan(args):
    rows, trees, _, _, _, _, evidence = source_inputs(args)
    return {'schema': SCHEMA, 'scope': SCOPE, **missing_plan(rows, trees),
            'evidence': evidence, 'test_used': False, 'inference_started': False}


def register(args):
    rows, trees, _, payload, cache, raw, evidence = source_inputs(args)
    proposal = missing_plan(rows, trees)
    expected = {'schema': SCHEMA, 'scope': SCOPE, 'source_files': previous.fitting.source_files(),
        'paths': {name: str(getattr(args, name).resolve()) for name in ('data', 'source', 'source_models', 'b_reference', 'output')},
        'evidence': evidence, 'plan': proposal,
        'science_ids': [r['id'] for r in rows if r['subject'] in SCIENCE],
        'collection_rule': 'All128science questions and all10fixed paths; only absent guided/guided-reuse/retry-reuse nodes are acquired.',
        'output_cap': 128, 'temperature': .65, 'context_length': 4096,
        'model': previous.study.old.MODEL, 'model_digest': previous.study.old.MODEL_DIGEST,
        'teacher_model_digest': previous.study.old.JUDGE_DIGEST,
        'maximum_requests': 2, 'maximum_cloud_gpus': 2, 'neural_calls': 0, 'new_teacher_calls': 0,
        'test_used': False, 'grades_opened': False, 'score_after_complete_science_grid_only': True}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.replay.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / 'registration.json'
        if path.exists():
            if read(path) != expected:
                raise ValueError('Changed registered path audit; preserve this area')
        else:
            if any(p.name != '.replay.lock' for p in args.output.iterdir()):
                raise ValueError('Never overwrite unregistered work')
            write(path, expected)
        imported = {'schema': SCHEMA, 'source_sha256': sha(__file__), 'source': evidence,
            'neural_cache_sha256': evidence['source_cache_sha256'], 'raw_import_sha256': evidence['source_raw_sha256'],
            'selection_by_correctness': False, 'test_used': False}
        previous.import_prefix(args.output, payload, cache, raw, imported)
    return expected


def registered_view(args):
    contract = read(args.output / 'registration.json')
    if (contract['schema'] != SCHEMA or contract['source_files'] != previous.fitting.source_files()
            or contract['paths'] != {name: str(getattr(args, name).resolve()) for name in ('data', 'source', 'source_models', 'b_reference', 'output')}):
        raise ValueError('Changed registered source/path identity')
    previous.fitting.verify_hashes(contract['source_files'])
    rows, original, _, payload, cache, raw, evidence = source_inputs(args)
    if evidence != contract['evidence'] or missing_plan(rows, original) != contract['plan']:
        raise ValueError('Original complete source, control panel, or acquisition plan changed')
    if (contract['science_ids'] != [r['id'] for r in rows if r['subject'] in SCIENCE]
            or contract['scope'] != SCOPE
            or (contract['output_cap'], contract['temperature'], contract['context_length'],
                contract['model'], contract['model_digest'], contract['teacher_model_digest'],
                contract['maximum_requests'], contract['maximum_cloud_gpus']) !=
                (128, .65, 4096, previous.study.old.MODEL, previous.study.old.MODEL_DIGEST,
                 previous.study.old.JUDGE_DIGEST, 2, 2)
            or contract.get('new_teacher_calls') != 0 or contract.get('neural_calls') != 0
            or contract.get('test_used') is not False or contract.get('grades_opened') is not False
            or contract.get('score_after_complete_science_grid_only') is not True):
        raise ValueError('Changed registered cohort, generation budget, or blind completion contract')
    previous.import_prefix(args.output, payload, cache, raw, read(args.output / 'cache-import.json'))
    imported = read(args.output / 'cache-import.json')
    if (imported['source'] != evidence or imported['source_sha256'] != sha(__file__)
            or sha(args.output / 'private-neural-cache.pkl') != evidence['source_cache_sha256']):
        raise ValueError('Preserve the entire unmodified imported neural cache and source')
    records = serial.strict_records(args.output / 'dev-observations.jsonl')
    raw_records = serial.strict_records(args.output / 'raw-generations.jsonl')
    rows, trees, judges = public_view(args.data, records)
    source_records = [json.loads(line) for line in payload.splitlines()]
    suffix = records[len(source_records):]
    permitted = {(r['id'], r['path']) for r in contract['plan']['missing']}
    if any(r['kind'] != 'solver' or (r['id'], r['path']) not in permitted or not r.get('metadata_acquired') for r in suffix):
        raise ValueError('Acquired an unregistered question/path or a forbidden new teacher call')
    for record in raw_records[contract['evidence']['source_raw_generations']:]:
        RequestJournal.validate(record)
        payload = record['payload']; options = payload['options']
        if (payload['model'], options['num_predict'], options['temperature'], options['num_ctx']) != (
                previous.study.old.MODEL, 128, .65, 4096):
            raise ValueError('New raw receipt contains a teacher call or changed compact protocol')
    previous.coverage(records, raw_records)
    return contract, rows, trees, judges, records, raw_records


class ScienceAcquisition(previous.Acquisition):
    def acquire(self, row, missing):
        if (row.get('answer') != '' or row['subject'] not in SCIENCE
                or missing.kind != 'solver' or missing.path not in ADDITIONAL):
            raise ValueError('Only registered science compact continuations may be acquired')
        if (row['id'], '') not in self.judges:
            raise ValueError('A paid existing root review is required; never acquire a teacher call')
        return super().acquire(row, missing)


def completion_payload(args, contract, rows, trees, records, raw):
    missing = missing_plan(rows, trees)
    if missing['maximum_new_compact_calls'] != 0:
        raise ValueError('Complete all128science ten-path grids before sealing or grading')
    coverage = previous.coverage(records, raw)
    new_raw = raw[contract['evidence']['source_raw_generations']:]
    files = ('registration.json', 'cache-import.json', 'dev-observations.jsonl',
             'raw-generations.jsonl', 'private-neural-cache.pkl', 'imported-neural-cache.pkl')
    if (args.output / 'transport-errors.jsonl').exists():
        files += ('transport-errors.jsonl',)
    return {'schema': SCHEMA, 'scope': SCOPE, 'source_sha256': sha(__file__),
        'registration_sha256': sha(args.output / 'registration.json'),
        'questions': 128, 'science_question_path_cells': 1280, 'required_paths': list(PATHS),
        'science_ids_sha256': serial.fingerprint([r['id'] for r in rows if r['subject'] in SCIENCE]),
        'digests': {name: sha(args.output / name) for name in files}, 'raw_coverage': coverage,
        'new_logical_compact_nodes': contract['plan']['maximum_new_compact_calls'],
        'new_completed_physical_generations': len(new_raw),
        'new_compact_input_tokens': sum(int(r['response']['prompt_eval_count']) for r in new_raw),
        'new_compact_output_tokens': sum(int(r['response'].get('eval_count', 0)) for r in new_raw),
        'physical_cost_scope': 'Completed acquisition calls only; imported calls stay in the complete bank. This is not controller compute.',
        'new_teacher_calls': 0, 'neural_calls': 0, 'grades_opened': False, 'test_used': False}


def collect(args):
    with (args.output / '.replay.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'collection-complete.json').exists():
            verify_complete(args)
            print('Verified completed path grid; no acquisition repeated.', flush=True)
            return
        with ExitStack() as stack:
            streams = {}
            for name in ('dev-observations.jsonl', 'raw-generations.jsonl'):
                stream = stack.enter_context((args.output / name).open('a'))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); streams[name] = stream
            contract, rows, trees, judges, records, raw = registered_view(args)
            client, client_lock = None, threading.Lock()
            def request(endpoint, payload=None):
                nonlocal client
                if endpoint == '/api/generate' and (payload['model'] != previous.study.old.MODEL
                    or payload['options']['num_predict'] != 128 or payload['options']['temperature'] != .65
                    or payload['options']['num_ctx'] != 4096):
                    raise ValueError('Forbidden teacher call or changed compact generation contract')
                with client_lock:
                    if client is None:
                        import modal
                        worker = modal.Cls.from_name(previous.study.APP, 'Ollama')()
                        worker.request.autoscaler.update(max_containers=2)
                        candidate = previous.study.resilient(worker, args.output / 'transport-errors.jsonl')
                        tags = {r['name']: r['digest'] for r in candidate('/api/tags')['models']}
                        if tags.get(previous.study.old.MODEL) != previous.study.old.MODEL_DIGEST:
                            raise ValueError('Compact solver weights changed')
                        client = candidate
                return client(endpoint, payload)
            journal = RequestJournal(request, streams['raw-generations.jsonl'], raw, slots=2)
            acquire = ScienceAcquisition(rows, trees, judges, streams['dev-observations.jsonl'], journal, None)
            science = [row for row in rows if row['subject'] in SCIENCE]
            def one(row):
                for path in ADDITIONAL:
                    acquire.acquire(row, previous.study.MissingObservation('solver', path))
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(one, science))
            contract, rows, trees, _, records, raw = registered_view(args)
            write(args.output / 'collection-complete.json', completion_payload(args, contract, rows, trees, records, raw))
            print({'complete': True, 'questions': 128, 'science_path_cells': 1280,
                   'new_logical_nodes_this_run': acquire.added, 'new_teacher_calls': 0,
                   'grades_opened': False}, flush=True)


def verify_complete(args):
    contract, rows, trees, judges, records, raw = registered_view(args)
    marker = args.output / 'collection-complete.json'
    if not marker.exists() or read(marker) != completion_payload(args, contract, rows, trees, records, raw):
        raise ValueError('Require the unchanged fully sealed1280-cell science grid before grading')
    return contract, rows, trees, judges


def grade(args):
    with (args.output / '.replay.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract, public, trees, _ = verify_complete(args)
        destination = args.output / 'graded'
        if destination.exists():
            raise ValueError('Never overwrite or reselect an opened hindsight bound')
        destination.mkdir()
        write(destination / 'opening.json', {'collection_complete_sha256': sha(args.output / 'collection-complete.json'),
            'all1280science_path_cells_precede_reference_access': True, 'scope': SCOPE, 'test_used': False})
        # First reference-value decode in this tool; protected TEST is never read.
        path = args.data / 'splits/dev-questions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [serial.public_row(row) for row in rows] != public or sha(path) != contract['evidence']['dev_questions_sha256']:
            raise ValueError('Development reference membership changed after the complete grid')
        from scoring import correct
        private = [{'id': row['id'], 'subject': row['subject'],
            'path_correct': {path: int(correct(trees[row['id']][path], row['answer']))
                             for path in PATHS if path in trees[row['id']]}}
                   for row in rows]
        subjects = {subject: {'questions': sum(r['subject'] == subject for r in private),
            'known_union_correct': sum(any(r['path_correct'].values()) for r in private if r['subject'] == subject),
            'all_ten_paths_complete': all(len(r['path_correct']) == 10 for r in private if r['subject'] == subject)}
                    for subject in ('math', 'chemistry', 'physics')}
        science = sum(subjects[s]['known_union_correct'] for s in SCIENCE)
        optimistic = science + subjects['math']['questions']
        needed = contract['evidence']['minimum_C_correct_above_current_B']
        summary = {'scope': SCOPE, 'development_questions': 256, 'science_questions': 128,
            'subjects': subjects, 'complete_science_union_correct': science,
            'known_total_union_correct': science + subjects['math']['known_union_correct'],
            'optimistic_total_correct_allowing_all128math': optimistic,
            'optimistic_accuracy_upper_bound': optimistic / 256,
            'current_strongest_B_correct': contract['evidence']['B_correct'],
            'minimum_C_correct_above_current_B': needed,
            'current_single_review_tree_ruled_out_even_with_hindsight': optimistic < needed,
            'interpretation': 'Below the target rules out this fixed single-review response tree even with hindsight; reaching the bound does not establish a learnable or compute-efficient controller.',
            'allowed_paths': list(PATHS), 'test_used': False, 'fresh_validation': False,
            'collection_complete_sha256': sha(args.output / 'collection-complete.json'),
            'registration_sha256': sha(args.output / 'registration.json')}
        write(destination / 'private-per-question.json', private)
        write(destination / 'summary.json', summary)
        write(destination / 'complete.json', {'summary_sha256': sha(destination / 'summary.json'),
            'private_per_question_sha256': sha(destination / 'private-per-question.json'),
            'scope': SCOPE, 'test_used': False})
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'register', 'collect', 'grade'))
    for name in ('data', 'source', 'source-models', 'b-reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = {'plan': plan, 'register': register, 'collect': collect, 'grade': grade}[args.mode](args)
    if result is not None:
        public = {k: v for k, v in result.items() if k not in ('missing', 'science_ids', 'source_files')}
        if 'plan' in public:
            public['plan'] = {k: v for k, v in public['plan'].items() if k != 'missing'}
        print(json.dumps(public), flush=True)


if __name__ == '__main__':
    main()
