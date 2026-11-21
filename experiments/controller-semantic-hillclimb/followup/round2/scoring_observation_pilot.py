"""Unlaunched training-only paid conditional-continuation observation pilot.

All64 fixed science questions and all three original decision nodes are kept.
The one-token response is observed, never submitted as an answer. This is not
historical token recovery, full-vocabulary entropy, or an LLM judge.
"""
import argparse
import ast
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import threading

import finish_once_pilot as original
import format_fixed_controls as fixed
import science_mentor_expansion as gates
import prospective_collect as protocol
from prospective_parallel import RequestJournal, KeyLocks, durable_line

SCHEMA = 'paid-conditional-continuation-observation-v1'
PATHS = ('', 'repair', 'retry')
COUNTS = {'chemistry': 32, 'physics': 32}
MODEL_BLOB = '81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490'
MODEL_BYTES = 3389971840
WORKERS = 2
MARKER = re.compile(r'\b(?:final(?:\s+answer)?|answer)\s*\**\s*:', re.IGNORECASE)
LETTER = re.compile(r'\s*([A-J])\s*\Z')
FEATURES = ('marker_present', 'original_answer_valid', 'next_token_observed',
    'top_alternative_count_over20', 'returned_mass', 'unreturned_mass',
    'largest_returned_probability', 'largest_minus_second_returned_probability',
    'recognized_letter_mass', 'recognized_letter_count_over10',
    'current_letter_observed', 'current_letter_observed_mass_or_masked0',
    'current_letter_best_observed_logprob_or_masked0',
    'largest_other_letter_observed_mass', 'current_minus_other_mass_or_masked0',
    'current_letter_observed_rank_over10_or_masked0')
SCREEN = {'joint_logloss_improvement_at_least': .03,
    'paired_question_bootstrap_upper95_below': 0.,
    'current_repair_retry_brier_each_no_worse': True,
    'valid_original_current_letter_coverage_at_least': .8,
    'scope': 'Screen for further controlled training expansion only; never controller promotion'}
sha, digest, read, write = fixed.sha, fixed.digest, gates.read, protocol.atomic_json


def source_files():
    """Static local dependency closure, avoiding unrelated imported test files."""
    project = Path(__file__).resolve().parents[4]
    here = Path(__file__).resolve().parent
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
        if name in sys.modules:
            path = local(getattr(sys.modules[name], '__file__', None))
            if path is not None:
                return path
        rel = Path(*name.split('.'))
        for folder in dict.fromkeys([str(here), *sys.path]):
            for path in (Path(folder or '.') / rel.with_suffix('.py'), Path(folder or '.') / rel / '__init__.py'):
                if local(path) is not None:
                    return local(path)
        return None
    pending = [(Path(__file__).resolve(), 'scoring_observation_pilot')]
    found = set()
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


def frozen_blob():
    path = Path(__file__).resolve().parent.parent / 'modal_solver.py'
    constants = {}
    for node in ast.parse(path.read_bytes()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('MODELS', 'MANIFESTS_B64'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    payload = base64.b64decode(constants['MANIFESTS_B64'][protocol.old.MODEL])
    manifest = json.loads(payload)
    models = [r for r in manifest['layers'] if r['mediaType'] == 'application/vnd.ollama.image.model']
    if (constants['MODELS'][protocol.old.MODEL] != protocol.old.MODEL_DIGEST
            or hashlib.sha256(payload).hexdigest() != protocol.old.MODEL_DIGEST
            or len(models) != 1 or models[0]['digest'] != 'sha256:' + MODEL_BLOB
            or models[0]['size'] != MODEL_BYTES):
        raise ValueError('Frozen manifest or exact GGUF blob identity changed')
    return {'manifest_sha256': protocol.old.MODEL_DIGEST, 'model_blob_sha256': MODEL_BLOB,
            'model_blob_bytes': MODEL_BYTES, 'deployed_source': str(path), 'deployed_source_sha256': sha(path),
            'ollama_version': '0.40.16'}


def refined_gate(args):
    """Validate aggregate completion/provenance only, never per-question DEV data."""
    required = (args.data.parent / 'development-refined-observer-v1/summary.json').resolve()
    if args.refined_completion.resolve() != required or not required.is_file():
        raise ValueError('Current refined A must finish before pilot registration or collection')
    summary = read(required)
    folder = required.parent
    identity = read(folder / 'identity.json')
    provenance = read(folder / 'oof-provenance.json')
    verifier = (args.data.parent / 'verifier-stream-v4').resolve()
    if (summary.get('development_questions') != 256 or summary.get('fitting_questions') != 512
            or summary.get('test_used') is not False or identity.get('test_used') is not False
            or Path(identity['verifier']).resolve() != verifier
            or identity['source_sha256'] != sha(Path(__file__).with_name('development_refined_observer.py'))
            or identity['train_questions_sha256'] != sha(args.data / 'splits/train-questions.jsonl')
            or identity['dev_questions_sha256'] != sha(args.data / 'splits/dev-questions.jsonl')
            or summary['oof_provenance_sha256'] != sha(folder / 'oof-provenance.json')
            or provenance.get('new_questions') != 512 or provenance.get('all_new_nodes_verified') != 8704
            or provenance.get('decision_nodes_used') != 1536
            or provenance.get('development_test_question_overlap') != 0
            or provenance['alignment_sha256'] != sha(folder / 'oof-alignment.jsonl')):
        raise ValueError('Refined A completion is partial or not bound to V4 training OOF provenance')
    bindings = {'train.jsonl': 'metadata_sha256', 'oof.npz': 'oof_sha256',
        'contract.json': 'training_contract_sha256', 'training-summary.json': 'training_summary_sha256'}
    for name, field in bindings.items():
        if sha(verifier / name) != provenance[field]:
            raise ValueError('Refined A training OOF evidence changed: ' + name)
    if set(provenance['fold_recipe_sha256']) != {'0', '1', '2', 'full'}:
        raise ValueError('Refined A requires all three excluded folds and the full deployment recipe')
    for fold, value in provenance['fold_recipe_sha256'].items():
        if sha(verifier / fold / 'recipe.json') != value:
            raise ValueError('Refined A OOF recipe changed')
    paths = [required, folder / 'identity.json', folder / 'oof-provenance.json', folder / 'oof-alignment.jsonl',
             *[verifier / name for name in bindings], *[verifier / fold / 'recipe.json' for fold in ('0', '1', '2', 'full')]]
    return {str(path): sha(path) for path in paths}


def public_inputs(args):
    prerequisite = refined_gate(args)
    rows = gates.public_train(args.data)
    parent = read(args.mentor / 'contract.json'); complete = read(args.mentor / 'complete.json')
    if (complete['questions'] != 192 or complete['records'] != 960
            or complete['contract_sha256'] != sha(args.mentor / 'contract.json')
            or complete['bank_sha256'] != sha(args.mentor / 'train.jsonl')
            or parent['new_question_split_sha256'] != sha(args.data / 'splits/train-questions.jsonl')):
        raise ValueError('Changed sealed original mentor cohort')
    chosen = [v for v in fixed.selection(rows, parent) if v['subject'] in COUNTS]
    if len(chosen) != 64 or Counter(v['subject'] for v in chosen) != COUNTS:
        raise ValueError('Require all64 prespecified new science questions')
    by_id = {r['id']: r for r in rows}
    trees, prefix_sha = original.plain_controls(args.data, {r['id'] for r in chosen})
    for item in chosen:
        row, tree = by_id[item['id']], trees[item['id']]
        for path, node in tree.items():
            protocol.validate_node(row, 'plain', 'solver', path, node, tree, {})
    evidence = {**prerequisite, **{str((args.mentor / name).resolve()): sha(args.mentor / name)
                for name in ('contract.json', 'complete.json', 'train.jsonl')}}
    for path in (args.data / 'splits/train-questions.jsonl', args.data / 'train-plain-complete.json',
                 args.base / 'identity.json', args.base / 'C1-smooth8.pkl', args.base / 'training-neural.jsonl',
                 args.base / 'training-neural.npz', args.legacy_examples,
                 args.baseline_diagnostic / 'summary.json', args.baseline_diagnostic / 'registration.json',
                 args.baseline_diagnostic / 'private-oof.npz', args.baseline_diagnostic / 'fold-receipts.json',
                 args.baseline_diagnostic / 'training-alignment.json', *[
                     args.baseline_diagnostic / f'numeric_C1-fold{i}.pkl' for i in range(3)]):
        evidence[str(path.resolve())] = sha(path)
    return [by_id[v['id']] for v in chosen], trees, chosen, prefix_sha, evidence


def scoring_prompt(row, path, tree):
    if row.get('answer') != '' or row['domain'] != 'science' or path not in PATHS:
        raise ValueError('Observation requires a blank reference and a paid science decision prefix')
    paid = {p: tree[p] for p in ('', path)}
    current = paid[path]
    action_prompt = protocol.old.solver_prompt(row, path, paid, {})
    if (hashlib.sha256(action_prompt.encode()).hexdigest() != current['prompt_sha256']
            or current['seed'] != fixed.logical_seed(row['id'], path)):
        raise ValueError('Original action prompt or seed changed')
    markers = list(MARKER.finditer(current['text']))
    prefix = current['text'][:markers[-1].end()] if markers else current['text'] + '\nFinal:'
    text = action_prompt + prefix
    receipt = {'original_prompt_sha256': current['prompt_sha256'], 'original_seed': current['seed'],
        'paid_ancestry_sha256': digest(paid), 'current_response_sha256': digest(current),
        'marker_present': bool(markers), 'marker_rule': 'Last final/answer marker through colon, no trailing answer',
        'prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
        'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}
    return text, receipt


def request_payload(row, path, tree):
    prompt, receipt = scoring_prompt(row, path, tree)
    return {'model': protocol.old.MODEL, 'prompt': prompt, 'raw': True, 'stream': False,
            'logprobs': True, 'top_logprobs': 20, 'keep_alive': '30m', 'truncate': False, 'shift': False,
            'options': {'temperature': tree[path]['temperature'], 'seed': tree[path]['seed'],
                        'num_predict': 1, 'num_ctx': 4096}}, receipt


def alternatives(response):
    """Unique returned alternatives only; no invented missing-token mass."""
    tokens = response.get('logprobs') or []
    if not isinstance(tokens, list) or len(tokens) > 1:
        raise ValueError('One-token observer returned multiple token positions')
    if not tokens:
        return {}, False
    entry = tokens[0]; values = entry.get('top_logprobs') or []
    if not isinstance(values, list) or len(values) > 20:
        raise ValueError('Top20 observation contract changed')
    unique = {}
    # The selected token can be outside top20. Retain its genuine logprob too.
    for item in [*values, entry]:
        raw = bytes(item['bytes']) if 'bytes' in item else item['token'].encode()
        score = item['logprob']
        if (type(score) not in (int, float) or not math.isfinite(score) or score > 1e-7):
            raise ValueError('Nonfinite or positive returned token logprob')
        if raw in unique and not math.isclose(unique[raw], score, abs_tol=1e-7, rel_tol=0):
            raise ValueError('Duplicate returned token has conflicting likelihood')
        unique[raw] = float(score)
    if sum(math.exp(v) for v in unique.values()) > 1 + 1e-5:
        raise ValueError('Returned alternative probabilities exceed unit mass')
    return unique, True


def feature_record(current, prefix_receipt, response):
    values, present = alternatives(response)
    probabilities = sorted((math.exp(v) for v in values.values()), reverse=True)
    masses, best = {}, {}
    for token, logprob in values.items():
        match = LETTER.fullmatch(token.decode('utf-8', errors='replace'))
        if match:
            letter = match[1]; masses[letter] = masses.get(letter, 0.) + math.exp(logprob)
            best[letter] = max(best.get(letter, -math.inf), logprob)
    answer = current['answer']; valid = answer in 'ABCDEFGHIJ' and len(answer) == 1
    seen = valid and answer in masses
    mass = min(1., sum(probabilities)); other = max([p for a, p in masses.items() if a != answer], default=0.)
    rank = 1 + sum(p > masses[answer] for p in masses.values()) if seen else 0
    features = [float(prefix_receipt['marker_present']), float(valid), float(present),
        len((response.get('logprobs') or [{}])[0].get('top_logprobs') or []) / 20.,
        mass, 1.-mass, probabilities[0] if probabilities else 0.,
        probabilities[0] - (probabilities[1] if len(probabilities) > 1 else 0.) if probabilities else 0.,
        sum(masses.values()), len(masses)/10., float(seen), masses[answer] if seen else 0.,
        best[answer] if seen else 0., other, masses[answer]-other if seen else 0., rank/10.]
    return {'features': features, 'current_letter_observed': bool(seen), 'original_answer_valid': valid,
        'returned_letter_masses': masses, 'current_letter_mass': masses[answer] if seen else None,
        'current_letter_best_logprob': best[answer] if seen else None,
        'missing_option_rule': 'Unreturned letters are censored; masked numeric zeros are not zero probabilities',
        'entropy': None, 'entropy_note': 'No full-vocabulary or semantic answer entropy estimated'}


def observation(row, path, tree, raw):
    payload, prefix = request_payload(row, path, tree)
    RequestJournal.validate(raw)
    if raw['payload'] != payload:
        raise ValueError('Raw observation request differs from registered paid prefix')
    response = raw['response']
    if (response.get('model') != protocol.old.MODEL or response.get('done') is not True
            or response.get('srr_backend', {}).get('ollama_version') != '0.40.16'
            or type(response.get('prompt_eval_count')) is not int or not 0 < response['prompt_eval_count'] <= 4096
            or type(response.get('eval_count')) is not int or not 0 <= response['eval_count'] <= 1
            or not isinstance(response.get('response'), str)):
        raise ValueError('Changed one-token model/version/physical usage receipt')
    cached = response.get('prompt_eval_cached_count')
    if (cached is not None and (type(cached) is not int or not 0 <= cached <= response['prompt_eval_count'])):
        raise ValueError('Cached prompt tokens must be included within the full input count')
    if len(response.get('logprobs') or []) > response['eval_count']:
        raise ValueError('Observed token positions exceed physical generated-token count')
    elapsed = raw['returned_unix'] - raw['started_unix']
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid physical observation latency')
    features = feature_record(tree[path], prefix, response)
    tokens = response['prompt_eval_count'] + response['eval_count']
    return {'id': row['id'], 'subject': row['subject'], 'path': path, 'kind': 'solver-observation',
        'protocol': SCHEMA, 'prefix': prefix, 'request_sha256': raw['request_sha256'],
        'raw_response_sha256': raw['response_sha256'], 'original_answer': tree[path]['answer'],
        'observer_input_tokens': response['prompt_eval_count'], 'observer_output_tokens': response['eval_count'],
        'observer_compute': 4*tokens/3.09, 'observer_calls': 1, 'latency_seconds': elapsed,
        'answer_changing_continuations': 0, 'review_calls': 0, 'new_neural_calls': 0,
        'cached_prompt_tokens_reported': response.get('prompt_eval_cached_count'),
        'all_input_tokens_charged_even_if_cached': True, **features}


def restore(rows, trees, records, raw_records, require_complete=False):
    by_id = {r['id']: r for r in rows}; raw = {}
    expected = {}
    for row in rows:
        for path in PATHS:
            payload, _ = request_payload(row, path, trees[row['id']])
            key = digest({'endpoint': '/api/generate', 'payload': payload})
            if key in expected:
                raise ValueError('Distinct decision nodes unexpectedly share an observation request')
            expected[key] = row['id'], path
    for item in raw_records:
        RequestJournal.validate(item); key = item['request_sha256']
        if key not in expected or key in raw:
            raise ValueError('Foreign or duplicate raw observation')
        qid, path = expected[key]
        observation(by_id[qid], path, trees[qid], item)
        raw[key] = item
    known = {}
    for item in records:
        key = item['id'], item['path']
        if key in known or item['request_sha256'] not in raw:
            raise ValueError('Duplicate logical observation or missing durable physical receipt')
        if item != observation(by_id[item['id']], item['path'], trees[item['id']], raw[item['request_sha256']]):
            raise ValueError('Logical observation differs from paid raw result')
        known[key] = item
    if require_complete and (set(raw) != set(expected) or set(known) != set(expected.values())):
        raise ValueError('Do not score until every question and all three prefixes are complete')
    return known, {'expected_calls': len(expected), 'physical_calls': len(raw), 'logical_observations': len(known)}


def registration(args):
    rows, trees, chosen, prefix_sha, evidence = public_inputs(args)
    requests = [{'id': r['id'], 'path': p, 'payload_sha256': digest(request_payload(r, p, trees[r['id']])[0]),
                 'prefix': request_payload(r, p, trees[r['id']])[1]} for r in rows for p in PATHS]
    return {'schema': SCHEMA, 'source_files': source_files(), 'input_evidence': evidence,
        'selection': chosen, 'subjects': COUNTS, 'decision_paths': list(PATHS), 'questions': 64,
        'observer_calls': 192, 'requests': requests, 'solver': frozen_blob(),
        'original_seven_path_controls_sha256': digest(trees), 'sealed_plain_prefix_sha256': prefix_sha,
        'features': list(FEATURES), 'screen': SCREEN, 'temperature_seed': 'Unchanged original node temperature and logical seed',
        'output_cap': 1, 'top_logprobs': 20, 'context_length': 4096, 'workers': WORKERS,
        'truncate': False, 'shift': False, 'oversize_prefix_rule': 'Fail the run; never shorten, omit, or replace a question',
        'answer_changing_continuations': 0, 'maximum_existing_answer_changing_continuations': 2,
        'scope': 'NEW paid conditional-continuation observation; not exact historical logits, a judge, or a generated replacement answer',
        'feature_missingness': 'Censored missing letters retained with masks; no fabricated probabilities; no full entropy',
        'coverage_screen_denominator': 'Every originally valid current answer among all192 decision nodes',
        'raw_coverage_finding': {'ordinary_training_questions': 512, 'original_decision_nodes': 1536,
            'preserved_full_logprobs': 0, 'stored_confidence_features_per_node': 13,
            'confidence_already_used_by_A': True, 'stored_answer_span_aligned_nodes': 1100,
            'compatible_known_training_raw_receipts': 0, 'failed_protocol_records_substituted': False},
        'diagnostic': {'variants': ['numeric_C1', 'numeric_plus_scoring_C1'], 'C': 1., 'temperature': 1.,
            'folds': 'SHA256(question_id)[:8] modulo3; all3nodes excluded together',
            'scaling': 'StandardScaler fit only within training fold', 'equal_question_weight': True,
            'reference': 'Same64 cohort numeric+20cached-neural-logit C1 control; archived all512-training question-OOF numeric model also reported',
            'bootstrap': '2000 paired whole-question draws; fixed fitted folds, descriptive OOF interval',
            'panels': ['all', 'chemistry', 'physics', 'root', 'repair', 'retry', 'subject_by_path'],
            'POMDP_runtime_changed': False, 'new_neural_calls': 0},
        'cost': 'Every scoring input+output token charged4B/3.09B even when prompt cache hits; physical latency and observer calls separate',
        'cost_api_source': 'https://raw.githubusercontent.com/ollama/ollama/v0.40.16/llm/llama_server.go',
        'cost_api_semantics': 'promptEvalCount returns cache_n plus prompt_n; prompt_eval_cached_count is a subset and is never subtracted',
        'transport': 'No automatic transport retries (attempts1). Stop for receipt audit on failure; completed durable generations are never retried. Any unresolved remote attempt makes acquisition cost a lower bound.',
        'no_retry_or_extra_token_fallback': True, 'no_outcome_filtering': True,
        'training_only': True, 'test_used': False, 'development_outcomes_read': False,
        'paths': {name: str(getattr(args, name).resolve()) for name in (
            'data', 'mentor', 'base', 'legacy_examples', 'baseline_diagnostic', 'refined_completion', 'output')}}


def register(args):
    record = registration(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'registration.json', record)
    print({'registered': True, 'registration_sha256': sha(args.output / 'registration.json'), 'calls_started': 0}, flush=True)


class Collector:
    def __init__(self, trees, known, stream, journal):
        self.trees, self.known, self.stream, self.journal = trees, known, stream, journal
        self.keys, self.lock, self.stopped = KeyLocks(), threading.RLock(), threading.Event()

    def one(self, work):
        row, path = work; key = row['id'], path
        try:
            with self.keys.get(key):
                if self.stopped.is_set() or key in self.known:
                    return
                payload, _ = request_payload(row, path, self.trees[row['id']])
                self.journal('/api/generate', payload)
                raw_key = digest({'endpoint': '/api/generate', 'payload': payload})
                item = observation(row, path, self.trees[row['id']], self.journal.known[raw_key])
                with self.lock:
                    durable_line(self.stream, item); self.known[key] = item
        except BaseException:
            self.stopped.set(); raise


def collect(args):
    registered = args.output / 'registration.json'
    with ExitStack() as stack:
        lock = stack.enter_context((args.output / 'collection.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'complete.json').exists():
            raise ValueError('Pilot already complete; do not repeat calls')
        record = registration(args)
        if not args.registration_sha256 or sha(registered) != args.registration_sha256 or read(registered) != record:
            raise ValueError('Externally recorded registration changed')
        rows, trees, *_ = public_inputs(args)
        bank, raw_path = args.output / 'train.jsonl', args.output / 'raw-generations.jsonl'
        stream, raw_stream = stack.enter_context(bank.open('a')), stack.enter_context(raw_path.open('a'))
        for handle in (stream, raw_stream):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records, raw = protocol.strict_records(bank), protocol.strict_records(raw_path)
        known, _ = restore(rows, trees, records, raw)
        import modal
        worker = modal.Cls.from_name(protocol.APP, 'Ollama')()
        journal = RequestJournal(protocol.resilient(worker, args.output / 'transport-errors.jsonl', attempts=1), raw_stream, raw, slots=WORKERS)
        tags = {r['name']: r['digest'] for r in journal('/api/tags')['models']}
        if tags.get(protocol.old.MODEL) != protocol.old.MODEL_DIGEST:
            raise ValueError('Frozen deployed solver manifest changed')
        collector = Collector(trees, known, stream, journal)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(collector.one, [(r, p) for r in rows for p in PATHS]))
        _, coverage = restore(rows, trees, protocol.strict_records(bank), protocol.strict_records(raw_path), True)
        if coverage['expected_calls'] != 192 or registration(args) != record:
            raise ValueError('Registered full cohort or evidence changed')
        errors = args.output / 'transport-errors.jsonl'
        errors_count = len(protocol.strict_records(errors))
        write(args.output / 'complete.json', {'schema': SCHEMA, 'coverage': coverage, 'questions': 64,
            'registration_sha256': sha(registered), 'bank_sha256': sha(bank), 'raw_sha256': sha(raw_path),
            'transport_errors_sha256': sha(errors) if errors.exists() else None,
            'transport_error_records': errors_count, 'acquisition_compute_is_lower_bound': bool(errors_count),
            'training_only': True, 'scored': False, 'answers_replaced': 0, 'new_neural_calls': 0})
        print('All192 paid observations complete; no outcome scored', flush=True)


def validated_complete(args):
    complete = read(args.output / 'complete.json')
    if (complete.get('schema') != SCHEMA or complete.get('training_only') is not True
            or complete.get('scored') is not False or complete.get('answers_replaced') != 0
            or complete.get('new_neural_calls') != 0 or complete.get('questions') != 64
            or complete.get('coverage') != {
            'expected_calls': 192, 'physical_calls': 192, 'logical_observations': 192}):
        raise ValueError('Require sealed64question/192observation grid before training labels')
    for name, key in (('registration.json', 'registration_sha256'), ('train.jsonl', 'bank_sha256'), ('raw-generations.jsonl', 'raw_sha256')):
        if sha(args.output / name) != complete[key]:
            raise ValueError('Changed sealed observation bank')
    errors = args.output / 'transport-errors.jsonl'
    if (sha(errors) if errors.exists() else None) != complete['transport_errors_sha256']:
        raise ValueError('Changed transport accounting')
    count = len(protocol.strict_records(errors))
    if (complete['transport_error_records'] != count
            or complete['acquisition_compute_is_lower_bound'] != bool(count)):
        raise ValueError('Transport uncertainty must remain explicit in the completion receipt')
    if read(args.output / 'registration.json') != registration(args):
        raise ValueError('Changed scoring-observation registration')
    rows, trees, *_ = public_inputs(args)
    known, coverage = restore(rows, trees, protocol.strict_records(args.output / 'train.jsonl'),
                              protocol.strict_records(args.output / 'raw-generations.jsonl'), True)
    if coverage != complete['coverage']:
        raise ValueError('Changed complete coverage')
    return rows, trees, known, complete


def diagnose(args):
    import scoring_observation_diagnostic
    scoring_observation_diagnostic.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'collect', 'diagnose'))
    for name in ('data', 'mentor', 'base', 'legacy-examples', 'baseline-diagnostic', 'refined-completion', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    {'register': register, 'collect': collect, 'diagnose': diagnose}[args.phase](args)
