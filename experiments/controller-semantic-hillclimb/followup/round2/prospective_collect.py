"""Audit frozen development policies, then collect the whole prospective panel blind.

Test collection writes private response receipts and ungraded decisions only.
The separate grader requires complete, verified receipts for all eight methods.
"""

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary, canonical_review
from difficulty_pomdp import PLAIN, MissingObservation
from late_pairs import judge_prompt, sanitized, llm_policy
from review_pilot import inference
from review_fields import parse as parse_review_fields
from long_review_pilot import review
from reasoning_pilot import short_repair
from transfer_variants import actor
from study import fixed
from study_common import BASELINES, statistics
from semantic_data import read
from transport import resilient
import extended_acquire as old
import mentor_pairs
from prospective_freeze import sha, verify, load_models
from prospective_audit import replay_paid_inputs, audit_costs, prefix

APP = 'srr-followup-solver-20261120'
METHODS = (*BASELINES, 'A', 'B', 'C')
STORES = ('plain', 'ordinary', 'strong')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def strict_records(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Partial receipt tail requires explicit recovery; never regenerate a completed answer: ' + str(path))
    return [json.loads(line) for line in raw.splitlines()]


def public_row(row):
    return {**{key: row[key] for key in ('id', 'question', 'domain', 'subject')}, 'answer': ''}


def parse_public_line(line):
    """Decode only whitelisted fields; lexically skip the reference value."""
    decoder = json.JSONDecoder()
    required = {'id', 'question', 'domain', 'subject'}
    result, seen = {}, set()
    text = line.strip()
    if not text.startswith('{'):
        raise ValueError('Question record is not a JSON object')
    i = 1
    while True:
        while i < len(text) and text[i].isspace(): i += 1
        if i >= len(text): raise ValueError('Incomplete public question record')
        if text[i] == '}':
            i += 1
            break
        key, i = decoder.raw_decode(text, i)
        if not isinstance(key, str) or key in seen:
            raise ValueError('Duplicate or nonstring question field')
        seen.add(key)
        while i < len(text) and text[i].isspace(): i += 1
        if i >= len(text) or text[i] != ':': raise ValueError('Missing question field separator')
        i += 1
        while i < len(text) and text[i].isspace(): i += 1
        start, depth, quoted, escaped = i, 0, False, False
        while i < len(text):
            char = text[i]
            if quoted:
                if escaped: escaped = False
                elif char == '\\': escaped = True
                elif char == '"': quoted = False
            elif char == '"': quoted = True
            elif char in '[{': depth += 1
            elif char in ']}':
                if depth == 0: break
                depth -= 1
            elif char == ',' and depth == 0: break
            i += 1
        if quoted or depth or i == start:
            raise ValueError('Incomplete JSON question value')
        if key in required:
            result[key] = json.loads(text[start:i])
            if not isinstance(result[key], str):
                raise ValueError('Public question fields must be strings')
        # In particular, no JSON decoder sees the reference-answer value.
        if i >= len(text): raise ValueError('Incomplete question object')
        if text[i] == '}':
            i += 1
            break
        if text[i] != ',': raise ValueError('Unexpected question separator')
        i += 1
    if text[i:].strip() or set(result) != required:
        raise ValueError('Missing public fields or trailing question content')
    return {**result, 'answer': ''}


def read_public_rows(path):
    return [parse_public_line(line) for line in path.read_text().splitlines()]


def source_group(row):
    source = 'GSM8K' if row['id'].startswith('gsm8k-') else 'SuperGPQA' if row['id'].startswith('supergpqa-') else 'unknown'
    if source == 'unknown':
        raise ValueError('Unexpected source in the reserved Round2 panel')
    return source


def method_store(frozen, method):
    if method in BASELINES or method == 'A':
        return 'plain'
    protocol = frozen['selected'][method]['protocol']
    if method == 'B' and protocol == 'ordinary-late':
        return 'ordinary'
    if protocol in ('mixed-strong-root', 'matched-retained-1024-512-128'):
        return 'strong'
    raise ValueError('Unsupported frozen method protocol')


def dispatch(frozen, models, method):
    if method in BASELINES:
        return lambda row, tree, judges: fixed(row, tree, *BASELINES[method])
    config = frozen['selected'][method]
    if method == 'B':
        function = llm_policy if config['protocol'] == 'ordinary-late' else importlib.import_module('matched_review_b').policy
        return lambda row, tree, judges: function(row, tree, judges, config['settings'])
    module = importlib.import_module(config['module'])
    return lambda row, tree, judges: module.evaluate(row, tree, judges, models[method], config['settings'], method)


class ResponseStore:
    def __init__(self, streams=None, records=None):
        self.streams = streams or {}
        self.known = {name: {} for name in STORES}
        for name, values in (records or {}).items():
            for item in values:
                key = item['id'], item['kind'], item['path']
                if key in self.known[name]:
                    raise ValueError('Duplicate response receipt')
                self.known[name][key] = item['response']

    def append(self, name, qid, kind, path, node, reused=None):
        key = qid, kind, path
        if key in self.known[name]:
            raise ValueError('Attempt to overwrite a completed response')
        if name not in self.streams:
            raise ValueError('This response bank is read-only')
        item = {'id': qid, 'kind': kind, 'path': path, 'response': node, 'exact_reuse': reused}
        stream = self.streams[name]
        stream.write(json.dumps(item) + '\n'); stream.flush(); os.fsync(stream.fileno())
        self.known[name][key] = node

    def view(self, qid, name):
        tree = {p: n for (i, k, p), n in self.known['plain'].items() if i == qid and k == 'solver'}
        reviews = {}
        if name != 'plain':
            for (i, kind, path), node in self.known[name].items():
                if i != qid:
                    continue
                if kind == 'solver':
                    tree[path] = node
                else:
                    reviews[qid, path] = canonical_review(node)
        return tree, reviews


def seed(qid, logical):
    return int(hashlib.sha256(f'{old.SEED}:{qid}:{logical}'.encode()).hexdigest()[:8], 16) % 2**31


def specification(row, name, kind, path, tree, reviews):
    """Reconstruct the exact frozen prompt and decoding contract from paid inputs."""
    if row.get('answer'):
        raise ValueError('Reference reached acquisition')
    if kind == 'judge':
        if name == 'strong':
            if path:
                raise ValueError('Strong protocol has only a root review')
            return None  # Both bounded phases are checked by verify_review().
        if name != 'ordinary' or path not in ('', 'repair', 'retry', 'guided'):
            raise ValueError('Forbidden ordinary review path')
        prompt = old.judge_prompt(row, path, prefix(tree, path)) if not path else judge_prompt(row, path, prefix(tree, path))
        return prompt, seed(row['id'], path), old.JUDGE_DIGEST, 0., 512, 4096
    if kind != 'solver':
        raise ValueError('Unknown response kind')
    parent = path.rsplit('/', 1)[0] if '/' in path else ''
    if name == 'plain':
        if path not in PLAIN:
            raise ValueError('Nonplain response cannot enter the shared plain bank')
        prompt = old.solver_prompt(row, path, prefix(tree, parent) if path else {}, {})
        return prompt, seed(row['id'], path), old.MODEL_DIGEST, .65 if path else 0., 256 if path else 128, 4096
    if name == 'strong':
        if path not in ('guided', 'guided/retry', 'guided/reuse', 'retry/reuse'):
            raise ValueError('Forbidden retained-review solver path')
        prompt = old.retry_prompt(row) if path == 'guided/retry' else short_repair(row, tree[parent], reviews[row['id'], ''])
        return prompt, mentor_pairs.seed_for(row['id'], path), old.MODEL_DIGEST, .65, 256 if path == 'guided/retry' else 128, 4096
    if name != 'ordinary':
        raise ValueError('Unknown response bank')
    if path.endswith('/reviewed') and parent in ('repair', 'retry', 'guided'):
        prompt = old.repair_prompt(row, tree[parent], old.feedback(reviews[row['id'], parent]))
        logical = parent + '/repair'
    elif path in ('guided', 'guided/repair', 'guided/retry'):
        prompt = old.solver_prompt(row, path, prefix(tree, parent), reviews)
        logical = path.replace('guided', 'repair')
    else:
        raise ValueError('Forbidden ordinary guided path')
    return prompt, seed(row['id'], logical), old.MODEL_DIGEST, .65, 256, 4096


def validate_node(row, name, kind, path, node, tree, reviews):
    spec = specification(row, name, kind, path, tree, reviews)
    if spec is None:
        if node['model_digest'] != old.JUDGE_DIGEST:
            raise ValueError('Strong reviewer weights changed')
        mentor_pairs.verify_review(row, tree[''], node, 1024, 512)
        if any(not 0 <= call['output_tokens'] <= call['output_cap'] or call['input_tokens'] <= 0 for call in node['calls']):
            raise ValueError('Invalid bounded-review physical token receipt')
        return
    prompt, expected_seed, digest, temperature, cap, context = spec
    if (node['prompt_sha256'], node['seed'], node['model_digest'], node['temperature']) != (
            hashlib.sha256(prompt.encode()).hexdigest(), expected_seed, digest, temperature):
        raise ValueError('Receipt differs from the frozen prompt, seed, weights, or temperature')
    if node.get('output_cap', cap) != cap or node.get('context_length', context) != context:
        raise ValueError('Receipt decoding limits changed')
    if not 0 <= node['output_tokens'] <= cap or node['input_tokens'] <= 0:
        raise ValueError('Invalid inference token receipt')
    if kind == 'solver' and node['answer'] != old.parse_answer(row['domain'], node['text']):
        raise ValueError('Stored solver answer differs from the frozen parser')
    if kind == 'judge':
        if path:
            verdict, valid, complete = parse_review_fields(node['text'])
            if node.get('json_complete') != complete:
                raise ValueError('Stored ordinary review completion flag changed')
            expected = sanitized({'p_correct': verdict.get('p_correct', .5), 'action': verdict.get('action', 'repair')})
        else:
            verdict, valid = old.parse_verdict(node['text'])
            expected = {'p_correct': verdict['p_correct'], 'action': verdict['action']}
        if (node['verdict'], node['parse_valid'], node['p_correct'], node['action'], node['answer']) != (
                verdict, valid, expected['p_correct'], expected['action'],
                old.parse_answer(row['domain'], 'Final: ' + str(verdict.get('answer', '')))):
            raise ValueError('Stored ordinary review differs from its frozen parser')


class Acquisition:
    def __init__(self, store, request=None):
        self.store, self.request = store, request

    def ensure(self, row, name, kind, path):
        if row.get('answer'):
            raise ValueError('Acquisition requires a blank reference')
        if kind == 'solver' and path in PLAIN:
            name = 'plain'
        tree, reviews = self.store.view(row['id'], name)
        existing = self.store.known[name].get((row['id'], kind, path))
        if existing is not None:
            validate_node(row, name, kind, path, existing, tree, reviews)
            return existing
        if self.request is None:
            raise MissingObservation(kind, path)
        if kind == 'judge' and path not in tree:
            self.ensure(row, name, 'solver', path)
        if kind == 'solver' and path:
            parent = path.rsplit('/', 1)[0] if '/' in path else ''
            self.ensure(row, name, 'solver', parent)
            if name == 'strong' or path == 'guided':
                self.ensure(row, name, 'judge', '')
            if path.endswith('/reviewed'):
                self.ensure(row, name, 'judge', parent)
        tree, reviews = self.store.view(row['id'], name)
        spec = specification(row, name, kind, path, tree, reviews)
        reused = None
        if kind == 'solver' and path == 'guided/retry':
            node = self.ensure(row, 'plain', 'solver', 'repair/retry')
            # Pure retries share a prompt and logical seed, not a repaired context.
            validate_node(row, name, kind, path, node, tree, reviews)
            reused = {'bank': 'plain', 'kind': 'solver', 'path': 'repair/retry', 'response_sha256': fingerprint(node)}
        elif name == 'strong' and kind == 'judge':
            node = canonical_review(review(self.request, row, tree[''], 1024, 512))
        elif name == 'strong':
            node = actor(self.request, row, spec[0], spec[1], spec[4])
        elif name == 'ordinary' and kind == 'judge' and path:
            node = sanitized(inference(self.request, row, spec[0], 'brief', True, path))
        elif name == 'ordinary' and kind == 'solver' and path.endswith('/reviewed'):
            node = inference(self.request, row, spec[0], 'brief', False, path.rsplit('/', 1)[0] + '/repair')
        else:
            old.request = self.request
            node = old.call(row, path, tree, reviews, kind == 'judge')
        validate_node(row, name, kind, path, node, tree, reviews)
        self.store.append(name, row['id'], kind, path, node, reused)
        return node


def validate_encoder_ledger(frozen, item, ledger):
    seen = set()
    decision_paths = {d['path'] for d in item.get('decisions', [])}
    review_paths = {d['path'] for d in item.get('decisions', []) if d['queried']}
    for receipt in ledger:
        key = receipt['recipe_sha256'], receipt['kind'], receipt['path'], receipt['input_sha256']
        if key in seen or receipt['recipe_sha256'] not in frozen['observer_assets']:
            raise ValueError('Duplicate or unregistered neural receipt')
        seen.add(key)
        recipe = frozen['observer_assets'][receipt['recipe_sha256']]['recipe']
        if receipt['parameters_billions'] != recipe['parameters'] / 1e9 or not 0 < receipt['slots'] <= recipe['max_length']:
            raise ValueError('Neural charge differs from the frozen physical observer')
        if receipt['kind'] not in ('solver', 'review') or receipt['path'] not in (decision_paths if receipt['kind'] == 'solver' else review_paths):
            raise ValueError('An unpaid or terminal response reached the observer')


def make_decision(frozen, models, method, row, store, acquisition=None):
    name = method_store(frozen, method)
    model = models.get(method)
    function = dispatch(frozen, models, method)
    if acquisition is not None:
        acquisition.ensure(row, 'plain', 'solver', '')
    for _ in range(8):
        tree, reviews = store.view(row['id'], name)
        try:
            item, ledger = replay_paid_inputs(function, row, tree, reviews,
                method if method in ('A', 'B', 'C') else 'baseline', model)
            break
        except MissingObservation as missing:
            if acquisition is None:
                raise
            acquisition.ensure(row, name, missing.kind, missing.path)
    else:
        raise ValueError('Policy exceeded finite acquisition horizon')
    paid_tree = prefix(tree, item['path'])
    paid_reviews = {(row['id'], d['path']): reviews[row['id'], d['path']]
                    for d in item.get('decisions', []) if d['queried']}
    for path, node in paid_tree.items():
        validate_node(row, 'plain' if path in PLAIN else name, 'solver', path, node, paid_tree, paid_reviews)
    for (_, path), node in paid_reviews.items():
        validate_node(row, name, 'judge', path, node, paid_tree, paid_reviews)
    validate_encoder_ledger(frozen, item, ledger)
    audited = audit_costs(item, paid_tree, paid_reviews, ledger, method if method in ('A', 'B', 'C') else 'baseline')
    item['judge_physical_calls'] = audited['physical_review_calls']
    receipts = [{'bank': 'plain' if path in PLAIN else name, 'kind': 'solver', 'path': path, 'sha256': fingerprint(node)}
                for path, node in paid_tree.items()]
    receipts += [{'bank': name, 'kind': 'judge', 'path': path, 'sha256': fingerprint(node)} for (_, path), node in paid_reviews.items()]
    return {'id': row['id'], 'method': method, 'bank': name, 'subject': row['subject'], 'source': source_group(row),
            'item': item, 'neural_receipts': ledger, 'paid_receipts': receipts,
            'final_response_sha256': fingerprint(paid_tree[item['path']]), 'blind_paid_prefix_audit': audited}


def verify_decision(frozen, record, store, row=None):
    if record['method'] not in METHODS or 'correct' in record['item']:
        raise ValueError('Decision is graded or has an unknown method')
    if record['bank'] != method_store(frozen, record['method']):
        raise ValueError('Decision used the wrong response protocol')
    tree, reviews = store.view(record['id'], record['bank'])
    paid_tree = prefix(tree, record['item']['path'])
    paid_reviews = {(record['id'], d['path']): reviews[record['id'], d['path']]
                    for d in record['item'].get('decisions', []) if d['queried']}
    if row is not None:
        if row['id'] != record['id'] or row.get('answer'):
            raise ValueError('Receipt verification requires the matching public question')
        for path, node in paid_tree.items():
            name = 'plain' if path in PLAIN else record['bank']
            validate_node(row, name, 'solver', path, node, paid_tree, paid_reviews)
        for (_, path), node in paid_reviews.items():
            validate_node(row, record['bank'], 'judge', path, node, paid_tree, paid_reviews)
    expected = [{'bank': 'plain' if path in PLAIN else record['bank'], 'kind': 'solver', 'path': path, 'sha256': fingerprint(node)}
                for path, node in paid_tree.items()]
    expected += [{'bank': record['bank'], 'kind': 'judge', 'path': path, 'sha256': fingerprint(node)} for (_, path), node in paid_reviews.items()]
    if expected != record['paid_receipts'] or fingerprint(paid_tree[record['item']['path']]) != record['final_response_sha256']:
        raise ValueError('Paid response changed after its decision')
    validate_encoder_ledger(frozen, record['item'], record['neural_receipts'])
    actual = audit_costs(record['item'], paid_tree, paid_reviews, record['neural_receipts'],
                        record['method'] if record['method'] in ('A', 'B', 'C') else 'baseline')
    if actual != record['blind_paid_prefix_audit']:
        raise ValueError('Saved cost audit disagrees with independent receipts')
    return paid_tree[record['item']['path']]


def development_store(data, frozen, method):
    rows, trees, original, late = ordinary(data, 'dev')
    records = {name: [] for name in STORES}
    for row in rows:
        qid = row['id']
        records['plain'] += [{'id': qid, 'kind': 'solver', 'path': path, 'response': node}
                             for path, node in trees[qid].items() if path in PLAIN]
        records['ordinary'] += [{'id': qid, 'kind': 'solver', 'path': path, 'response': node}
                                for path, node in trees[qid].items() if path not in PLAIN]
    merged = {**{k: v for k, v in original.items() if not k[1]}, **late}
    records['ordinary'] += [{'id': qid, 'kind': 'judge', 'path': path, 'response': node}
                            for (qid, path), node in merged.items()]
    if method_store(frozen, method) == 'strong':
        summary = Path(frozen['selected'][method]['source_summary'])
        bank = summary.parent / 'dev-observations.jsonl'
        if sha(bank) != json.loads(summary.read_bytes())['bank_sha256']:
            raise ValueError('Development strong-response bank changed')
        records['strong'] = strict_records(bank)
    return rows, ResponseStore(records=records)


def audit_development(args):
    frozen, models = load_models(args.study, args.device, args.freeze_sha256)
    data = Path(frozen['data_directory'])
    path = args.study / 'development-audit.json'
    if path.exists():
        raise ValueError('Development audit already exists; verify it instead of overwriting')
    reports, replayed, count = {}, [], 0
    from scoring import correct
    for method in METHODS:
        rows, store = development_store(data, frozen, method)
        if len(rows) != 256:
            raise ValueError('Incomplete development panel')
        values = []
        for full in rows:
            record = make_decision(frozen, models, method, public_row(full), store)
            final = verify_decision(frozen, record, store, public_row(full))
            replayed.append(record)
            values.append({**record['item'], 'correct': int(correct(final, full['answer']))})
            count += 1
        measured = statistics(values)
        measured['mean_judge_physical_calls'] = sum(r['judge_physical_calls'] for r in values) / len(values)
        expected = frozen['development_baselines'][method] if method in BASELINES else frozen['selected'][method]['dev']
        if any(not math.isclose(measured[k], v, rel_tol=1e-7, abs_tol=1e-7) for k, v in expected.items()):
            raise ValueError('Archived development replay differs from selected metrics: ' + method)
        reports[method] = measured
        print('Audited frozen development policy', method, flush=True)
    replay_path = args.study / 'development-replay.json'
    atomic_json(replay_path, replayed)
    atomic_json(path, {'passed': True, 'freeze_sha256': args.freeze_sha256, 'questions': 256,
                      'method_question_audits': count, 'methods': reports, 'test_used': False,
                      'decision_receipts_sha256': sha(replay_path),
                      'source_sha256': sha(__file__), 'observer_work_measured_from_actual_inputs': True})


def collect(args):
    import modal  # Fail on a missing dependency before claiming the test panel.
    frozen = verify(args.study, args.freeze_sha256)
    data = Path(frozen['data_directory'])
    audit = json.loads((args.study / 'development-audit.json').read_bytes())
    if not audit['passed'] or audit['freeze_sha256'] != args.freeze_sha256 or audit['method_question_audits'] != 2048:
        raise ValueError('Finish all development paid-prefix and independent-cost audits first')
    if audit['decision_receipts_sha256'] != sha(args.study / 'development-replay.json'):
        raise ValueError('Frozen development decision audit changed')
    target = args.study / 'collection'
    target.mkdir(exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((args.study / 'collection.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (target / 'complete.json').exists():
            raise ValueError('Collection already complete; do not recollect or regenerate')
        claim = {'freeze_sha256': args.freeze_sha256, 'question_sha256': frozen['reserved_question_sha256'],
                 'study': str(args.study.resolve()), 'rules': 'All1024questions; no reference-conditioned calls or policy changes.'}
        claim_path = data / 'test-collection-claim.json'
        if claim_path.exists() and json.loads(claim_path.read_bytes()) != claim:
            raise ValueError('Reserved panel is already claimed by another policy freeze')
        if not claim_path.exists():
            atomic_json(claim_path, claim)
        # No code above this point reads individual test question records.
        rows = read_public_rows(data / 'splits/test-questions.jsonl')
        ids = {r['id'] for r in rows}
        by_id = {r['id']: r for r in rows}
        if len(rows) != 1024 or len(ids) != 1024:
            raise ValueError('Preserve all1024unique reserved questions')
        frozen, models = load_models(args.study, args.device, args.freeze_sha256)
        records, streams = {}, {}
        for name in STORES:
            path = target / (name + '.jsonl')
            records[name] = strict_records(path)
            stream = stack.enter_context(path.open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            streams[name] = stream
        store = ResponseStore(streams, records)
        if any(qid not in ids for values in store.known.values() for qid, _, _ in values):
            raise ValueError('Response bank contains a nonreserved question')
        saved_path = target / 'ungraded-decisions.jsonl'
        saved = strict_records(saved_path)
        known = {(r['id'], r['method']): r for r in saved}
        if len(known) != len(saved) or any(qid not in ids or method not in METHODS for qid, method in known):
            raise ValueError('Duplicate or unexpected ungraded decision')
        for record in saved:
            if record['freeze_sha256'] != args.freeze_sha256:
                raise ValueError('Decision belongs to another freeze')
            verify_decision(frozen, record, store, by_id[record['id']])
        output = stack.enter_context(saved_path.open('a'))
        fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = modal.Cls.from_name(APP, 'Ollama')()
        worker.request.autoscaler.update(max_containers=2)
        request = resilient(worker, target / 'transport-errors.jsonl')
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        if tags.get(old.MODEL) != frozen['solver_digest'] or tags.get(old.JUDGE) != frozen['reviewer_digest']:
            raise ValueError('Inference model weights changed')
        acquisition = Acquisition(store, request)
        for index, row in enumerate(rows):
            for method in METHODS:
                key = row['id'], method
                if key in known:
                    continue
                record = {'freeze_sha256': args.freeze_sha256,
                          **make_decision(frozen, models, method, row, store, acquisition)}
                output.write(json.dumps(record) + '\n'); output.flush(); os.fsync(output.fileno())
                known[key] = record
            if (index + 1) % 16 == 0:
                print(f'Blind collection: {index + 1}/1024questions; {len(known)}/8192policy decisions', flush=True)
        if set(known) != {(qid, method) for qid in ids for method in METHODS}:
            raise ValueError('Incomplete prospective policy grid')
        for record in known.values():
            verify_decision(frozen, record, store, by_id[record['id']])
        verify(args.study, args.freeze_sha256)
        digests = {name + '.jsonl': sha(target / (name + '.jsonl')) for name in STORES}
        digests['ungraded-decisions.jsonl'] = sha(saved_path)
        audit_receipt = {'passed': True, 'questions': 1024, 'method_question_audits': len(known),
                         'freeze_sha256': args.freeze_sha256, 'digests': digests,
                         'test_references_supplied_to_policies': False, 'grades_opened': False,
                         'source_sha256': sha(__file__)}
        atomic_json(target / 'collection-audit.json', audit_receipt)
        atomic_json(target / 'complete.json', {**audit_receipt, 'methods': list(METHODS),
            'question_ids_sha256': fingerprint(sorted(ids)), 'audit_sha256': sha(target / 'collection-audit.json'),
            'development_audit_sha256': sha(args.study / 'development-audit.json')})
        print('All8192prospective decisions collected and audited; grades remain closed.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('audit-development', 'collect'))
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--freeze-sha256', required=True, help='Externally recorded frozen manifest digest.')
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = p.parse_args()
    (audit_development if args.phase == 'audit-development' else collect)(args)
