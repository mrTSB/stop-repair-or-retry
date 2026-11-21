"""NEW science-only terminal Complete action: all256 TRAIN questions ×3 prefixes.

Reuse all192 sealed pilot requests exactly; acquire576 missing requests without
outcome selection. Complete consumes one continuation and immediately submits
its single offered letter (or an invalid answer). No scoring features are used.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import scoring_observation_pilot as pilot
import scoring_completion_availability as secondary

SCHEMA = 'science-terminal-complete-actions-v1'
COUNTS = {'chemistry': 128, 'physics': 128}
PREFIXES = pilot.PATHS
PATHS = ('complete', 'repair/complete', 'retry/complete')
WORKERS = 2
SCREEN = {'novel_correct_beyond_original_seven_at_least': 16,
          'scope': 'TRAIN feasibility gate for fitting a new action model only; no A/C promotion or target/fresh claim'}
sha, digest, read, write = pilot.sha, pilot.digest, pilot.read, pilot.write


def action_path(prefix):
    if prefix not in PREFIXES:
        raise ValueError('Complete is permitted only at root or after one repair/retry')
    return (prefix+'/' if prefix else '')+'complete'


def sources():
    return {**pilot.source_files(), str(Path(secondary.__file__).resolve()): sha(secondary.__file__),
            str(Path(__file__).resolve()): sha(__file__)}


def pilot_args(args):
    record = read(args.pilot/'registration.json')
    saved = {key: Path(value) for key, value in record['paths'].items()}
    if saved['data'].resolve() != args.data.resolve() or saved['output'].resolve() != args.pilot.resolve():
        raise ValueError('The reused pilot must belong to this exact TRAIN reservation')
    return SimpleNamespace(**saved)


def logical_record(row, prefix, tree, raw, reused):
    # Retain the exact original observation request protocol and its raw checks.
    # Its probability features are deliberately discarded, not model inputs.
    checked = pilot.observation(row, prefix, tree, raw)
    payload, prefix_receipt = pilot.request_payload(row, prefix, tree)
    text = raw['response']['response']
    answer = secondary.hypothetical_answer(text, secondary.offered_letters(row))
    node = {'model': pilot.protocol.old.MODEL, 'model_digest': pilot.protocol.old.MODEL_DIGEST,
        'text': text, 'answer': answer, 'prompt_sha256': prefix_receipt['prompt_sha256'],
        'seed': payload['options']['seed'], 'temperature': payload['options']['temperature'],
        'output_cap': 1, 'context_length': 4096,
        'input_tokens': checked['observer_input_tokens'], 'output_tokens': checked['observer_output_tokens'],
        'latency_seconds': checked['latency_seconds'], 'finish': raw['response'].get('done_reason'),
        'terminal': True, 'additional_answer_changing_calls': 1,
        'total_answer_changing_continuations': 1 if prefix == '' else 2,
        'new_neural_calls': 0, 'review_calls': 0,
        'backend_version': raw['response']['srr_backend']['ollama_version']}
    return {'id': row['id'], 'subject': row['subject'], 'kind': 'solver', 'path': action_path(prefix),
        'prefix_path': prefix, 'protocol': SCHEMA, 'paid_prefix': prefix_receipt,
        'reused_pilot_physical_response': bool(reused), 'request_sha256': raw['request_sha256'],
        'raw_response_sha256': raw['response_sha256'], 'response': node}


def public_inputs(args):
    inherited = pilot_args(args)
    old_rows, old_trees, old_known, old_complete = pilot.validated_complete(inherited)
    rows = [row for row in pilot.gates.public_train(args.data) if row['subject'] in COUNTS]
    if len(rows) != 256 or Counter(row['subject'] for row in rows) != COUNTS:
        raise ValueError('Require every128chemistry and128physics TRAIN question')
    by_id = {row['id']: row for row in rows}
    trees, prefix_sha = pilot.original.plain_controls(args.data, set(by_id))
    for row in rows:
        secondary.offered_letters(row)
        for path, node in trees[row['id']].items():
            pilot.protocol.validate_node(row, 'plain', 'solver', path, node, trees[row['id']], {})
    reused_raw = pilot.protocol.strict_records(args.pilot/'raw-generations.jsonl')
    raw_by_key = {record['request_sha256']: record for record in reused_raw}
    if len(reused_raw) != 192 or len(raw_by_key) != 192:
        raise ValueError('Import every exact192 pilot physical response')
    imported = []
    for row in old_rows:
        qid = row['id']
        if row != by_id[qid] or old_trees[qid] != trees[qid]:
            raise ValueError('Original pilot question or seven-path prefix changed')
        for prefix in PREFIXES:
            key = old_known[qid, prefix]['request_sha256']
            imported.append(logical_record(row, prefix, trees[qid], raw_by_key[key], True))
    old_ids = {row['id'] for row in old_rows}
    missing = [{'id': row['id'], 'subject': row['subject'], 'prefix_path': prefix,
        'path': action_path(prefix), 'payload_sha256': digest(pilot.request_payload(row, prefix, trees[row['id']])[0])}
        for row in rows if row['id'] not in old_ids for prefix in PREFIXES]
    if len(missing) != 576 or len(old_ids) != 64:
        raise ValueError('Exactly192 reused plus576 new physical requests are required')
    evidence = {str((args.pilot/name).resolve()): sha(args.pilot/name) for name in (
        'registration.json', 'complete.json', 'train.jsonl', 'raw-generations.jsonl',
        'diagnostic/summary.json', 'secondary-availability/registration.json', 'secondary-availability/summary.json')}
    evidence.update({str((args.data/name).resolve()): sha(args.data/name) for name in (
        'splits/train-questions.jsonl', 'train-plain-complete.json')})
    return rows, trees, imported, reused_raw, missing, {'files': evidence,
        'sealed_plain_prefix_sha256': prefix_sha, 'original_pilot_coverage': old_complete['coverage'],
        'original_pilot_transport_errors': old_complete['transport_error_records'],
        'pilot_ids_sha256': digest(sorted(old_ids)), 'all_science_ids_sha256': digest(sorted(by_id))}


def restore(rows, trees, reused_keys, records, raw_records, require_complete=False):
    by_id = {row['id']: row for row in rows}; expected, raw, known = {}, {}, {}
    for row in rows:
        for prefix in PREFIXES:
            key = digest({'endpoint': '/api/generate', 'payload': pilot.request_payload(row, prefix, trees[row['id']])[0]})
            if key in expected:
                raise ValueError('Two original prefixes unexpectedly share a physical request')
            expected[key] = row['id'], prefix
    if not set(reused_keys) <= set(expected):
        raise ValueError('An imported pilot request is not an exact current prefix')
    for value in raw_records:
        key = value['request_sha256']
        if key not in expected or key in raw:
            raise ValueError('Unknown or duplicated Complete physical request')
        qid, prefix = expected[key]
        logical_record(by_id[qid], prefix, trees[qid], value, key in reused_keys)
        raw[key] = value
    for value in records:
        key = value['request_sha256']
        qid, prefix = expected.get(key, (None, None))
        if key not in raw or (qid, prefix) in known:
            raise ValueError('Logical Complete lacks a unique raw receipt')
        expected_record = logical_record(by_id[qid], prefix, trees[qid], raw[key], key in reused_keys)
        if value != expected_record:
            raise ValueError('Complete action differs from exact paid prefix/token/seed/protocol')
        known[qid, prefix] = value
    if require_complete and (set(raw) != set(expected) or set(known) != set(expected.values())):
        raise ValueError('Seal all768 question/prefix cells before opening labels')
    return known, {'expected_cells': len(expected), 'physical_calls': len(raw), 'logical_actions': len(known),
                   'reused_physical_calls': len(set(raw) & set(reused_keys)),
                   'new_physical_calls': len(set(raw) - set(reused_keys))}


def specification(args):
    rows, trees, imported, raw, missing, evidence = public_inputs(args)
    return {'schema': SCHEMA, 'source_files': sources(), 'evidence': evidence,
        'paths': {key: str(getattr(args, key).resolve()) for key in ('data', 'pilot', 'output')},
        'selection': [{'id': row['id'], 'subject': row['subject']} for row in rows],
        'questions': 256, 'subjects': COUNTS, 'prefixes': list(PREFIXES), 'terminal_paths': list(PATHS),
        'raw_reuse': {'count': 192, 'sha256': sha(args.pilot/'raw-generations.jsonl'),
                     'logical_records_sha256': digest(imported)}, 'missing': missing,
        'new_calls': 576, 'total_calls': 768, 'new_teacher_calls': 0, 'new_neural_calls': 0,
        'math_rule': 'Math retains strongest existing A and original actions; no math Complete generation or invented label',
        'solver': pilot.frozen_blob(), 'request_protocol': pilot.SCHEMA,
        'output_cap': 1, 'top_logprobs': 20, 'context_length': 4096, 'truncate': False, 'shift': False,
        'temperature_seed': 'Exact original paid-prefix temperature and seed, unchanged from the pilot',
        'answer_rule': 'Only response.strip() exactly one uppercase offered option; all others invalid; no fallback/retry',
        'action_budget': 'Complete costs1 continuation and terminates:1 after root,2 total after repair/retry',
        'no_continuation_after_complete': True, 'score_features_used': False,
        'all_original_seven_paths_retained': True, 'controls_sha256': digest(trees),
        'matched_controls': ['Complete', 'Repair then Complete', 'Retry then Complete',
            'Stop if valid else Complete', 'Repair then Complete if invalid', 'Retry then Complete if invalid'],
        'original_five_baselines_unchanged': True, 'screen': SCREEN,
        'report_panels': ['all_science', 'chemistry', 'physics', 'original_pilot64', 'remaining192'],
        'comparison_scope': 'Science TRAIN only; describe every control accuracy/cost, gate solely on16novel oracle cases',
        'workers': WORKERS, 'maximum_cloud_gpus': 2, 'automatic_transport_attempts': 1,
        'no_outcome_filtering': True, 'training_only': True, 'test_used': False,
        'grades_only_after_full_grid': True, 'promotion_requires': 'New fitted16-state POMDP, matched controls/B availability, and separate fresh validation'}


def write_bytes(path, payload):
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def register(args):
    record = specification(args)
    rows, trees, imported, raw, _, _ = public_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'registration.json', record)
    write_bytes(args.output/'raw-generations.jsonl', (args.pilot/'raw-generations.jsonl').read_bytes())
    payload = ''.join(json.dumps(value, sort_keys=True)+'\n' for value in imported).encode()
    write_bytes(args.output/'train.jsonl', payload)
    write(args.output/'cache-import.json', {'schema': SCHEMA, 'raw_bytes': (args.pilot/'raw-generations.jsonl').stat().st_size,
        'raw_sha256': record['raw_reuse']['sha256'], 'logical_bytes': len(payload),
        'logical_prefix_sha256': hashlib.sha256(payload).hexdigest(), 'logical_records_sha256': digest(imported),
        'raw_records': 192, 'new_calls': 0})
    restored, coverage = restore(rows, trees, {v['request_sha256'] for v in raw}, imported, raw)
    if len(restored) != 192 or coverage['new_physical_calls'] != 0:
        raise ValueError('Import changed the exact pilot request grid')
    print({'registered': True, 'registration_sha256': sha(args.output/'registration.json'),
           'reused': 192, 'new_calls_started': 0}, flush=True)


def registered_view(args):
    record = read(args.output/'registration.json')
    if record != specification(args):
        raise ValueError('Changed frozen Complete action specification')
    rows, trees, imported, raw, missing, _ = public_inputs(args)
    receipt = read(args.output/'cache-import.json')
    if (receipt['schema'] != SCHEMA or receipt['raw_records'] != 192 or receipt['new_calls'] != 0
            or receipt['logical_records_sha256'] != digest(imported)
            or receipt['raw_sha256'] != sha(args.pilot/'raw-generations.jsonl')):
        raise ValueError('Changed exact cache-import receipt')
    for name, size, value in [('raw-generations.jsonl', receipt['raw_bytes'], receipt['raw_sha256']),
                              ('train.jsonl', receipt['logical_bytes'], receipt['logical_prefix_sha256'])]:
        with (args.output/name).open('rb') as stream:
            prefix = stream.read(size)
        if len(prefix) != size or not prefix.endswith(b'\n') or hashlib.sha256(prefix).hexdigest() != value:
            raise ValueError('The entire imported192-response prefix must remain unchanged')
    return rows, trees, {v['request_sha256'] for v in raw}, missing


class Collector:
    def __init__(self, trees, reused, known, stream, journal):
        self.trees, self.reused, self.known, self.stream, self.journal = trees, reused, known, stream, journal
        self.keys, self.lock, self.stopped = pilot.KeyLocks(), threading.RLock(), threading.Event()

    def one(self, work):
        row, prefix = work; key = row['id'], prefix
        try:
            with self.keys.get(key):
                if self.stopped.is_set() or key in self.known:
                    return
                payload = pilot.request_payload(row, prefix, self.trees[row['id']])[0]
                self.journal('/api/generate', payload)
                request_key = digest({'endpoint': '/api/generate', 'payload': payload})
                value = logical_record(row, prefix, self.trees[row['id']], self.journal.known[request_key], request_key in self.reused)
                with self.lock:
                    pilot.durable_line(self.stream, value); self.known[key] = value
                    if len(self.known) % 32 == 0:
                        print(f'Complete action grid {len(self.known)}/768', flush=True)
        except BaseException:
            self.stopped.set(); raise


def collect(args):
    with ExitStack() as stack:
        lock = stack.enter_context((args.output/'collection.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output/'complete.json').exists():
            raise ValueError('Keep the completed grid; never regenerate')
        if not args.registration_sha256 or sha(args.output/'registration.json') != args.registration_sha256:
            raise ValueError('Explicit reviewed registration SHA is required')
        rows, trees, reused, _ = registered_view(args)
        bank, raw_path = args.output/'train.jsonl', args.output/'raw-generations.jsonl'
        bank_stream, raw_stream = stack.enter_context(bank.open('a')), stack.enter_context(raw_path.open('a'))
        for stream in (bank_stream, raw_stream):
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = pilot.protocol.strict_records(raw_path)
        known, _ = restore(rows, trees, reused, pilot.protocol.strict_records(bank), raw)
        import modal
        worker = modal.Cls.from_name(pilot.protocol.APP, 'Ollama')()
        worker.request.autoscaler.update(max_containers=2)
        errors = args.output/'transport-errors.jsonl'
        journal = pilot.RequestJournal(pilot.protocol.resilient(worker, errors, attempts=1), raw_stream, raw, slots=WORKERS)
        tags = {v['name']: v['digest'] for v in journal('/api/tags')['models']}
        if tags.get(pilot.protocol.old.MODEL) != pilot.protocol.old.MODEL_DIGEST:
            raise ValueError('Frozen solver manifest changed')
        collector = Collector(trees, reused, known, bank_stream, journal)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(collector.one, [(row, prefix) for row in rows for prefix in PREFIXES]))
        _, coverage = restore(rows, trees, reused, pilot.protocol.strict_records(bank), pilot.protocol.strict_records(raw_path), True)
        registered_view(args)
        if coverage != {'expected_cells': 768, 'physical_calls': 768, 'logical_actions': 768,
                        'reused_physical_calls': 192, 'new_physical_calls': 576}:
            raise ValueError('The required full768 grid is incomplete')
        names = ['registration.json', 'cache-import.json', 'train.jsonl', 'raw-generations.jsonl']
        if errors.exists(): names.append('transport-errors.jsonl')
        inherited_errors = read(args.output/'registration.json')['evidence']['original_pilot_transport_errors']
        new_errors = len(pilot.protocol.strict_records(errors))
        write(args.output/'complete.json', {'schema': SCHEMA, 'coverage': coverage, 'questions': 256,
            'digests': {name: sha(args.output/name) for name in names},
            'registration_sha256': sha(args.output/'registration.json'),
            'inherited_transport_errors': inherited_errors, 'new_transport_errors': new_errors,
            'acquisition_compute_is_lower_bound': bool(inherited_errors or new_errors),
            'scored': False, 'training_only': True, 'test_used': False, 'new_neural_calls': 0, 'new_teacher_calls': 0})
        print('All768 Complete action cells sealed; no TRAIN labels scored', flush=True)


def validated_complete(args):
    complete = read(args.output/'complete.json')
    expected = {'expected_cells': 768, 'physical_calls': 768, 'logical_actions': 768,
                'reused_physical_calls': 192, 'new_physical_calls': 576}
    if (complete['schema'] != SCHEMA or complete['coverage'] != expected or complete['questions'] != 256
            or complete['training_only'] is not True or complete['test_used'] is not False
            or complete['scored'] is not False or complete['new_neural_calls'] != 0 or complete['new_teacher_calls'] != 0
            or complete['registration_sha256'] != sha(args.output/'registration.json')):
        raise ValueError('Require the full ungraded science Complete action grid')
    required = {'registration.json', 'cache-import.json', 'train.jsonl', 'raw-generations.jsonl'}
    errors = args.output/'transport-errors.jsonl'
    if errors.exists(): required.add('transport-errors.jsonl')
    if set(complete['digests']) != required or any(sha(args.output/name) != value for name, value in complete['digests'].items()):
        raise ValueError('Changed sealed acquisition artifacts')
    inherited = read(args.output/'registration.json')['evidence']['original_pilot_transport_errors']
    count = len(pilot.protocol.strict_records(errors))
    if (complete['inherited_transport_errors'] != inherited or complete['new_transport_errors'] != count
            or complete['acquisition_compute_is_lower_bound'] != bool(inherited or count)):
        raise ValueError('Changed transport uncertainty accounting')
    rows, trees, reused, _ = registered_view(args)
    known, coverage = restore(rows, trees, reused, pilot.protocol.strict_records(args.output/'train.jsonl'),
                              pilot.protocol.strict_records(args.output/'raw-generations.jsonl'), True)
    if coverage != expected:
        raise ValueError('Complete grid changed')
    return rows, trees, known, complete


def training_data(data_dir, complete_dir):
    """Fitter API, independent of its output directory; validates before labels."""
    data_dir, complete_dir = Path(data_dir).resolve(), Path(complete_dir).resolve()
    record = read(complete_dir/'registration.json')
    paths = {key: Path(value).resolve() for key, value in record['paths'].items()}
    if paths['data'] != data_dir or paths['output'] != complete_dir:
        raise ValueError('Completed action grid belongs to another reservation/directory')
    args = SimpleNamespace(**paths)
    rows, trees, known, complete = validated_complete(args)
    labels = {row['id']: row for row in pilot.fixed.lines(args.data/'splits/train-questions.jsonl')}
    for row in rows:
        if pilot.protocol.public_row(labels[row['id']]) != row:
            raise ValueError('TRAIN label row differs from sealed public prefix')
        for prefix in PREFIXES:
            trees[row['id']][action_path(prefix)] = known[row['id'], prefix]['response']
    receipt = {**complete, 'complete_sha256': sha(complete_dir/'complete.json'),
        'directory': str(complete_dir), 'data_directory': str(data_dir), 'pilot_directory': str(paths['pilot']),
        'source_files': record['source_files'], 'public_input_evidence': record['evidence']}
    return [labels[row['id']] for row in rows], trees, receipt


def aggregate(rows, trees, pilot_ids):
    if len(rows) != 256 or Counter(row['subject'] for row in rows) != COUNTS:
        raise ValueError('Grade all256science questions without filtering')
    outcomes = []
    for row in rows:
        qid, target = row['id'], row['answer']; tree = trees[qid]
        offered = secondary.offered_letters({**row, 'answer': ''})
        if target not in offered or any(path not in tree for path in (*pilot.protocol.PLAIN, *PATHS)):
            raise ValueError('Incomplete original/new action labels or invalid reference')
        methods = {name: [tree[path] for path in paths] for name, paths in pilot.fixed.FIXED.items()}
        for prefix, name, conditional in [('', 'Complete', 'Stop if valid else Complete'),
                ('repair', 'Repair then Complete', 'Repair then Complete if invalid'),
                ('retry', 'Retry then Complete', 'Retry then Complete if invalid')]:
            paid = [tree['']] + ([tree[prefix]] if prefix else [])
            methods[name] = [*paid, tree[action_path(prefix)]]
            methods[conditional] = paid if tree[prefix]['answer'] in offered else methods[name]
        values = {}
        for name, nodes in methods.items():
            tokens = sum(node['input_tokens']+node['output_tokens'] for node in nodes)
            values[name] = {'correct': bool(nodes[-1]['answer'] == target), 'valid': nodes[-1]['answer'] in offered,
                'tokens': tokens, 'compute': 4*tokens/3.09, 'solver_calls': len(nodes),
                'answer_changing_continuations': len(nodes)-1}
        old_oracle = any(tree[path]['answer'] == target for path in pilot.protocol.PLAIN)
        new_oracle = any(tree[path]['answer'] == target for path in (*pilot.protocol.PLAIN, *PATHS))
        outcomes.append({'id': qid, 'subject': row['subject'], 'reused_pilot_question': qid in pilot_ids,
            'methods': values, 'original_seven_oracle_correct': old_oracle, 'combined_ten_oracle_correct': new_oracle,
            'novel_correct': new_oracle and not old_oracle})
    panels = {}
    for name in ('all_science', 'chemistry', 'physics', 'original_pilot64', 'remaining192'):
        selected = [v for v in outcomes if name == 'all_science' or v['subject'] == name
            or (name == 'original_pilot64' and v['reused_pilot_question'])
            or (name == 'remaining192' and not v['reused_pilot_question'])]
        n = len(selected); methods = {}
        for method in selected[0]['methods']:
            values = [v['methods'][method] for v in selected]
            methods[method] = {'n': n, 'correct': sum(v['correct'] for v in values),
                'accuracy': sum(v['correct'] for v in values)/n, 'valid': sum(v['valid'] for v in values),
                **{'mean_'+key: sum(v[key] for v in values)/n for key in ('tokens', 'compute', 'solver_calls', 'answer_changing_continuations')}}
        best_original = max(pilot.fixed.FIXED, key=lambda method: (methods[method]['correct'], -methods[method]['mean_compute']))
        paired = {method: {'wins': sum(v['methods'][method]['correct'] and not v['methods'][best_original]['correct'] for v in selected),
            'losses': sum(not v['methods'][method]['correct'] and v['methods'][best_original]['correct'] for v in selected),
            'mean_token_difference': methods[method]['mean_tokens']-methods[best_original]['mean_tokens'],
            'mean_compute_ratio': methods[method]['mean_compute']/methods[best_original]['mean_compute']}
            for method in methods if method not in pilot.fixed.FIXED}
        panels[name] = {'questions': n, 'methods': methods, 'highest_original_baseline': best_original,
            'paired_vs_highest_original': paired,
            'original_seven_oracle_correct': sum(v['original_seven_oracle_correct'] for v in selected),
            'combined_ten_oracle_correct': sum(v['combined_ten_oracle_correct'] for v in selected),
            'novel_correct_beyond_original_seven': sum(v['novel_correct'] for v in selected)}
    return panels, outcomes, {'criteria': SCREEN, 'passed': panels['all_science']['novel_correct_beyond_original_seven'] >= 16,
        'controller_promoted': False, 'fixed_control_comparisons_descriptive_only': True}


def grade(args):
    rows, trees, complete = training_data(args.data, args.output)
    target = args.output/'graded'
    if target.exists():
        raise ValueError('Preserve completed feasibility result; no outcome-guided rescore')
    inherited = read(args.pilot/'registration.json')
    pilot_ids = {v['id'] for v in inherited['selection']}
    panels, outcomes, screen = aggregate(rows, trees, pilot_ids)
    acquisition = {}
    for name in ('all_grid', 'new', 'reused'):
        nodes = [trees[row['id']][path] for row in rows for path in PATHS
            if name == 'all_grid' or (name == 'reused') == (row['id'] in pilot_ids)]
        tokens = sum(node['input_tokens']+node['output_tokens'] for node in nodes)
        acquisition[name] = {'physical_calls': len(nodes), 'input_tokens': sum(node['input_tokens'] for node in nodes),
            'output_tokens': sum(node['output_tokens'] for node in nodes), 'active_parameter_token_proxy': 4*tokens/3.09,
            'summed_physical_latency_seconds': sum(node['latency_seconds'] for node in nodes)}
    validated_complete(args)
    target.mkdir(); write(target/'private-outcomes.json', outcomes)
    write(target/'summary.json', {'schema': SCHEMA, 'panels': panels, 'screen': screen,
        'complete_sha256': sha(args.output/'complete.json'), 'registration_sha256': sha(args.output/'registration.json'),
        'private_outcomes_sha256': sha(target/'private-outcomes.json'), 'questions': 256,
        'new_physical_calls': 576, 'reused_physical_calls': 192, 'new_neural_calls': 0, 'new_teacher_calls': 0,
        'acquisition_cost': acquisition,
        'acquisition_compute_is_lower_bound': complete['acquisition_compute_is_lower_bound'],
        'training_only': True, 'test_used': False, 'controller_accuracy_claim': False,
        'math_unchanged': True, 'primary_negative_scoring_screen_preserved': True})
    print(json.dumps({'complete': True, 'screen': screen, 'summary_sha256': sha(target/'summary.json'),
                     'all_science': panels['all_science']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'register', 'collect', 'grade'))
    for name in ('data', 'pilot', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    if args.phase == 'plan':
        record = specification(args)
        print(json.dumps({'questions': record['questions'], 'new_calls': record['new_calls'],
                          'reused_calls': record['raw_reuse']['count'], 'screen': SCREEN, 'registered': False}), flush=True)
    else:
        {'register': register, 'collect': collect, 'grade': grade}[args.phase](args)
