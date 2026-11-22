"""Blind DEV grid for the NEW terminal Complete action, after the fixed TRAIN fit.

All128 science questions and three prefixes are collected before DEV references
are decoded. The saved model and seven evaluation penalties are bound before
collection. This module never evaluates or changes a controller.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import science_complete_actions as actions
import development_retained_path_audit as ordinary
import fit_completion_pomdp as fitting
import expanded_mentor_io as io

SCHEMA = 'development-terminal-complete-grid-v1'
COUNTS = {'chemistry': 64, 'physics': 64}
PENALTIES = (0., .01, .02, .04, .08, .16, .24)
WORKERS = 2
read, write, sha, digest = actions.read, actions.write, actions.sha, actions.digest


def model_gate(args):
    """TRAIN completion fields/models only; no DEV outcomes and no model forward."""
    complete = fitting.verify_complete(args.model)
    registered = read(args.model/'registration.json')
    train = Path(registered['paths']['complete'])
    if (Path(registered['paths']['data']).resolve() != args.data.resolve()
            or registered['regularization'] != .1 or registered['smoothing'] != 32.
            or registered['models'] != 1 or registered['state_count'] != 16
            or registered['maximum_additional_calls'] != 2 or registered['complete_terminal'] is not True
            or registered['paid_scoring_features_used'] is not False
            or registered['math_unchanged'] is not True or registered['neural_calls'] != 0
            or registered['cloud_calls'] != 0 or registered['development_or_test_records_read'] is not False
            or registered['training_headroom']['novel'] < 16):
        raise ValueError('Require the prespecified single CPU-fit16-state science POMDP')
    summary = read(train/'graded/summary.json')
    headroom = registered['training_headroom']; panel = summary['panels']['all_science']
    if (summary['schema'] != actions.SCHEMA or summary['questions'] != 256
            or summary['training_only'] is not True or summary['test_used'] is not False
            or summary['controller_accuracy_claim'] is not False
            or summary['complete_sha256'] != registered['complete_grid_receipt']['complete_sha256']
            or summary['registration_sha256'] != sha(train/'registration.json')
            or summary['screen']['criteria'] != actions.SCREEN or summary['screen']['passed'] is not True
            or panel['original_seven_oracle_correct'] != headroom['plain7_oracle']
            or panel['combined_ten_oracle_correct'] != headroom['with_complete_oracle']
            or panel['novel_correct_beyond_original_seven'] != headroom['novel']):
        raise ValueError('The full TRAIN screen must pass before DEV registration')
    files = {str((args.model/name).resolve()): sha(args.model/name)
             for name in ('registration.json', 'complete.json', 'completion-A.pkl')}
    files[str((train/'graded/summary.json').resolve())] = sha(train/'graded/summary.json')
    return {'files': files, 'model_sha256': complete['model_sha256'],
        'model_registration_sha256': sha(args.model/'registration.json'),
        'train_grid_directory': str(train.resolve()), 'train_grid_complete_sha256': sha(train/'complete.json'),
        'train_novel_correct': headroom['novel'], 'fixed_regularization': .1, 'fixed_smoothing': 32.}


def public_inputs(args):
    gate = model_gate(args)
    rows = actions.pilot.protocol.read_public_rows(args.data/'splits/dev-questions.jsonl')
    if (len(rows) != 256 or len({row['id'] for row in rows}) != 256
            or Counter(row['subject'] for row in rows) != {'math': 128, **COUNTS}
            or any(row['answer'] != '' for row in rows)):
        raise ValueError('Require the full public256 DEV panel with blank references')
    by_id = {row['id']: row for row in rows}; trees = {qid: {} for qid in by_id}
    payload, completion = ordinary.plain_prefix(args.data)
    for line in payload.splitlines():
        value = json.loads(line); qid, path = value['id'], value['path']
        if (qid not in by_id or value['kind'] != 'solver' or path not in actions.pilot.protocol.PLAIN
                or path in trees[qid]):
            raise ValueError('Unexpected or duplicate sealed original DEV control')
        trees[qid][path] = value['response']
    for row in rows:
        tree = trees[row['id']]
        if set(tree) != set(actions.pilot.protocol.PLAIN):
            raise ValueError('Every original question keeps all seven fixed paths')
        for path, node in tree.items():
            actions.pilot.protocol.validate_node(row, 'plain', 'solver', path, node, tree, {})
    science = [row for row in rows if row['subject'] in COUNTS]
    training = read(Path(gate['train_grid_directory'])/'registration.json')
    if {row['id'] for row in science} & {row['id'] for row in training['selection']}:
        raise ValueError('DEV collection overlaps a Complete TRAIN question')
    for row in science:
        actions.secondary.offered_letters(row)
    selected_trees = {row['id']: trees[row['id']] for row in science}
    evidence = {'model_gate': gate, 'dev_questions_sha256': sha(args.data/'splits/dev-questions.jsonl'),
        'plain_completion_sha256': sha(args.data/'dev-plain-complete.json'),
        'sealed_plain_prefix_sha256': completion['base_bank_sha256'],
        'sealed_plain_prefix_records': completion['base_records']}
    return science, selected_trees, evidence


def specification(args):
    rows, trees, evidence = public_inputs(args)
    return {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'paths': {key: str(getattr(args, key).resolve()) for key in ('data', 'model', 'output')},
        'evidence': evidence, 'model_sha256': evidence['model_gate']['model_sha256'],
        'selection': [{'id': row['id'], 'subject': row['subject']} for row in rows],
        'questions': 128, 'subjects': COUNTS, 'prefixes': list(actions.PREFIXES),
        'required_cells': 384, 'new_compact_calls': 384, 'new_teacher_calls': 0, 'new_neural_calls': 0,
        'controls_sha256': digest(trees), 'original_seven_paths_retained': True,
        'requests': [{'id': row['id'], 'prefix_path': prefix, 'path': actions.action_path(prefix),
                      'payload_sha256': digest(actions.pilot.request_payload(row, prefix, trees[row['id']])[0])}
                     for row in rows for prefix in actions.PREFIXES],
        'request_protocol': actions.pilot.SCHEMA, 'action_protocol': actions.SCHEMA,
        'solver': actions.pilot.frozen_blob(), 'output_cap': 1, 'top_logprobs': 20, 'context_length': 4096,
        'truncate': False, 'shift': False, 'seed_temperature': 'Exact saved original prefix seed and temperature',
        'answer_rule': 'Exactly one uppercase offered option after whitespace; no extraction, fallback, or retry',
        'maximum_answer_changing_continuations': 2, 'complete_consumes_one_call_and_is_terminal': True,
        'paid_score_features_used': False, 'math_unchanged': True, 'references_supplied': False,
        'evaluation_penalties': list(PENALTIES), 'additional_hyperparameter_search': False,
        'original_baselines': list(actions.pilot.fixed.FIXED),
        'new_matched_controls': ['Complete', 'Repair then Complete', 'Retry then Complete',
            'Stop if valid else Complete', 'Repair then Complete if invalid', 'Retry then Complete if invalid'],
        'full256_control_composition': 'Science controls plus the exact same retained math A component, explicitly labeled',
        'all384_before_dev_outcomes': True, 'no_outcome_filtering': True, 'test_used': False,
        'development_only': True, 'workers': WORKERS, 'maximum_cloud_gpus': 2,
        'automatic_transport_attempts': 1, 'scope': 'New-action DEV collection/selection, never fresh validation'}


def register(args):
    record = specification(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'registration.json', record)
    print({'registered': True, 'registration_sha256': sha(args.output/'registration.json'),
           'new_calls_started': 0, 'required_calls': 384}, flush=True)


def registered_view(args):
    if read(args.output/'registration.json') != specification(args):
        raise ValueError('Changed registered model, DEV cohort, actions, or evaluation penalties')
    return public_inputs(args)


class Collector:
    def __init__(self, trees, known, stream, journal):
        self.trees, self.known, self.stream, self.journal = trees, known, stream, journal
        self.keys, self.lock, self.stopped = actions.pilot.KeyLocks(), threading.RLock(), threading.Event()

    def one(self, work):
        row, prefix = work; key = row['id'], prefix
        try:
            with self.keys.get(key):
                if self.stopped.is_set() or key in self.known:
                    return
                payload = actions.pilot.request_payload(row, prefix, self.trees[row['id']])[0]
                self.journal('/api/generate', payload)
                physical_key = digest({'endpoint': '/api/generate', 'payload': payload})
                value = actions.logical_record(row, prefix, self.trees[row['id']], self.journal.known[physical_key], False)
                with self.lock:
                    actions.pilot.durable_line(self.stream, value); self.known[key] = value
                    if len(self.known) % 32 == 0:
                        print(f'DEV Complete grid {len(self.known)}/384', flush=True)
        except BaseException:
            self.stopped.set(); raise


def collect(args):
    with ExitStack() as stack:
        lock = stack.enter_context((args.output/'collection.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output/'complete.json').exists():
            raise ValueError('Keep the complete DEV grid; never regenerate')
        if not args.registration_sha256 or sha(args.output/'registration.json') != args.registration_sha256:
            raise ValueError('Explicit reviewed DEV registration SHA required')
        rows, trees, _ = registered_view(args)
        bank, raw_path = args.output/'dev.jsonl', args.output/'raw-generations.jsonl'
        bank_stream, raw_stream = stack.enter_context(bank.open('a')), stack.enter_context(raw_path.open('a'))
        for stream in (bank_stream, raw_stream):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = actions.pilot.protocol.strict_records(raw_path)
        known, _ = actions.restore(rows, trees, set(), actions.pilot.protocol.strict_records(bank), raw)
        import modal
        worker = modal.Cls.from_name(actions.pilot.protocol.APP, 'Ollama')()
        worker.request.autoscaler.update(max_containers=2)
        errors = args.output/'transport-errors.jsonl'
        journal = actions.pilot.RequestJournal(actions.pilot.protocol.resilient(worker, errors, attempts=1), raw_stream, raw, slots=WORKERS)
        tags = {value['name']: value['digest'] for value in journal('/api/tags')['models']}
        if tags.get(actions.pilot.protocol.old.MODEL) != actions.pilot.protocol.old.MODEL_DIGEST:
            raise ValueError('Frozen compact solver manifest changed')
        collector = Collector(trees, known, bank_stream, journal)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(collector.one, [(row, prefix) for row in rows for prefix in actions.PREFIXES]))
        known, coverage = actions.restore(rows, trees, set(), actions.pilot.protocol.strict_records(bank),
                                          actions.pilot.protocol.strict_records(raw_path), True)
        registered_view(args)
        expected = {'expected_cells': 384, 'physical_calls': 384, 'logical_actions': 384,
                    'reused_physical_calls': 0, 'new_physical_calls': 384}
        if coverage != expected:
            raise ValueError('Entire384cell DEV grid required before labels')
        names = ['registration.json', 'dev.jsonl', 'raw-generations.jsonl']
        if errors.exists(): names.append('transport-errors.jsonl')
        count = len(actions.pilot.protocol.strict_records(errors))
        nodes = [value['response'] for value in known.values()]
        write(args.output/'complete.json', {'schema': SCHEMA, 'coverage': coverage, 'questions': 128,
            'digests': {name: sha(args.output/name) for name in names},
            'registration_sha256': sha(args.output/'registration.json'),
            'model_sha256': read(args.output/'registration.json')['model_sha256'],
            'physical_input_tokens': sum(node['input_tokens'] for node in nodes),
            'physical_output_tokens': sum(node['output_tokens'] for node in nodes),
            'summed_physical_latency_seconds': sum(node['latency_seconds'] for node in nodes),
            'transport_error_records': count, 'acquisition_compute_is_lower_bound': bool(count),
            'dev_outcomes_scored': False, 'development_only': True, 'test_used': False,
            'new_neural_calls': 0, 'new_teacher_calls': 0})
        print('All384 DEV Complete cells sealed; no DEV outcomes scored', flush=True)


def validated_complete(args):
    complete = read(args.output/'complete.json')
    expected = {'expected_cells': 384, 'physical_calls': 384, 'logical_actions': 384,
                'reused_physical_calls': 0, 'new_physical_calls': 384}
    if (complete['schema'] != SCHEMA or complete['coverage'] != expected or complete['questions'] != 128
            or complete['development_only'] is not True or complete['test_used'] is not False
            or complete['dev_outcomes_scored'] is not False
            or complete['new_neural_calls'] != 0 or complete['new_teacher_calls'] != 0
            or complete['registration_sha256'] != sha(args.output/'registration.json')
            or complete['model_sha256'] != read(args.output/'registration.json')['model_sha256']):
        raise ValueError('Require the full ungraded same-model384cell DEV seal')
    errors = args.output/'transport-errors.jsonl'
    required = {'registration.json', 'dev.jsonl', 'raw-generations.jsonl'}
    if errors.exists(): required.add('transport-errors.jsonl')
    if set(complete['digests']) != required or any(sha(args.output/name) != value for name, value in complete['digests'].items()):
        raise ValueError('Changed completed DEV artifacts')
    count = len(actions.pilot.protocol.strict_records(errors))
    if complete['transport_error_records'] != count or complete['acquisition_compute_is_lower_bound'] != bool(count):
        raise ValueError('Changed physical transport accounting')
    rows, trees, _ = registered_view(args)
    known, coverage = actions.restore(rows, trees, set(), actions.pilot.protocol.strict_records(args.output/'dev.jsonl'),
        actions.pilot.protocol.strict_records(args.output/'raw-generations.jsonl'), True)
    nodes = [value['response'] for value in known.values()]
    if (coverage != expected or complete['physical_input_tokens'] != sum(node['input_tokens'] for node in nodes)
            or complete['physical_output_tokens'] != sum(node['output_tokens'] for node in nodes)
            or complete['summed_physical_latency_seconds'] != sum(node['latency_seconds'] for node in nodes)):
        raise ValueError('Changed completed DEV grid/physical charges')
    return rows, trees, known, complete


def development_data(data_dir, complete_dir):
    data_dir, complete_dir = Path(data_dir).resolve(), Path(complete_dir).resolve()
    registered = read(complete_dir/'registration.json')
    paths = {key: Path(value).resolve() for key, value in registered['paths'].items()}
    if paths['data'] != data_dir or paths['output'] != complete_dir:
        raise ValueError('DEV grid belongs to another reservation/directory')
    args = SimpleNamespace(**paths)
    rows, trees, known, complete = validated_complete(args)
    labels = {row['id']: row for row in actions.pilot.fixed.lines(data_dir/'splits/dev-questions.jsonl')}
    for row in rows:
        if actions.pilot.protocol.public_row(labels[row['id']]) != row:
            raise ValueError('DEV target row differs from sealed public question')
        for prefix in actions.PREFIXES:
            trees[row['id']][actions.action_path(prefix)] = known[row['id'], prefix]['response']
    receipt = {**complete, 'complete_sha256': sha(complete_dir/'complete.json'),
        'directory': str(complete_dir), 'data_directory': str(data_dir), 'model_directory': str(paths['model']),
        'source_files': registered['source_files'], 'public_input_evidence': registered['evidence'],
        'evaluation_penalties': registered['evaluation_penalties']}
    return [labels[row['id']] for row in rows], trees, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'register', 'collect'))
    for name in ('data', 'model', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    if args.phase == 'plan':
        value = specification(args)
        print({'registered': False, 'questions': 128, 'new_calls': 384,
               'model_sha256': value['model_sha256'], 'penalties': list(PENALTIES)}, flush=True)
    else:
        {'register': register, 'collect': collect}[args.phase](args)
