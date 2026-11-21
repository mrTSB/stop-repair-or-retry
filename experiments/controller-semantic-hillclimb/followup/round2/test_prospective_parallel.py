"""Synthetic only: no datasets, real models, Modal, or network requests."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prospective_parallel as parallel

serial = parallel.serial


def row(index=0):
    return {'id': f'gsm8k-synthetic-{index}', 'question': f'What is {index} plus two?',
            'subject': 'math', 'domain': 'math', 'answer': ''}


class OwnedLock:
    def __init__(self):
        self.lock, self.owner = threading.Lock(), None
    def __enter__(self):
        self.lock.acquire(); self.owner = threading.get_ident()
    def __exit__(self, *args):
        self.owner = None; self.lock.release()


class ParallelTests(unittest.TestCase):
    def test_serial_parallel_all_methods_receipts_charges_and_nn_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recipe = {'parameters': 400000000, 'max_length': 512}
            (directory / 'recipe.json').write_text(json.dumps(recipe))
            recipe_sha = parallel.sha(directory / 'recipe.json')
            owner = self
            class Observer:
                def __init__(self):
                    self.directory = str(directory)
                    self._recipe = recipe
                    self.active = self.maximum = 0
                def load(self):
                    pass
                def entry(self, safe, path, tree, judge=None):
                    owner.assertEqual(safe['answer'], '')
                    owner.assertEqual(set(tree), {''})
                    return {'id': safe['id'], 'text': tree[path]['text']}
                def values(self, safe, path, tree, judge=None):
                    self.entry(safe, path, tree, judge)
                    self.active += 1; self.maximum = max(self.maximum, self.active)
                    time.sleep(.0002)
                    self.active -= 1
                    return [0.] * 4, [0.] * 8, [0.] * 8, 20
            observer = Observer()
            models = {name: SimpleNamespace(classifier=observer) for name in ('A', 'C')}
            frozen = {'observer_assets': {recipe_sha: {'recipe': recipe}},
                      'selected': {'B': {'protocol': 'ordinary-late'},
                                   'C': {'protocol': 'mixed-strong-root'}}}
            original_dispatch = serial.dispatch
            def dispatch(frozen, models, method):
                if method in serial.BASELINES:
                    return original_dispatch(frozen, models, method)
                def policy(safe, tree, reviews):
                    action, depth = ('retry', 2) if method == 'C' else ('repair', 2)
                    if method != 'B':
                        models[method].classifier.values(safe, '', tree)
                    item = serial.fixed(safe, tree, action, depth)
                    item['decisions'] = [{'path': '', 'queried': False}]
                    if method != 'B':
                        item.update(verifier_calls=1, verifier_tokens=20, verifier_token_slots=20)
                        item['compute'] += 20 * .4 / 3.09
                    item['total_parameter_compute'] = item['compute']
                    return item
                return policy
            all_rows = [row(i) for i in range(16)]
            outcomes = []
            maximum = []
            for workers in (1, 4):
                folder = directory / str(workers); folder.mkdir()
                policy_lock = OwnedLock()
                guard = threading.Lock()
                active = peak = calls = 0
                order = {r['id']: [] for r in all_rows}
                def request(endpoint, payload):
                    nonlocal active, peak, calls
                    self.assertNotEqual(policy_lock.owner, threading.get_ident())
                    with guard:
                        active += 1; peak = max(peak, active); calls += 1
                    time.sleep(.003)
                    with guard:
                        active -= 1
                    return {'response': 'Final: 2', 'done': True}
                def fake_call(safe, path, tree, reviews, judge):
                    self.assertFalse(judge)
                    spec = serial.specification(safe, 'plain', 'solver', path, tree, reviews)
                    result = serial.old.request('/api/generate', {'prompt': spec[0], 'seed': spec[1]})
                    return {'text': result['response'], 'answer': '2',
                            'prompt_sha256': serial.hashlib.sha256(spec[0].encode()).hexdigest(),
                            'seed': spec[1], 'model_digest': spec[2], 'temperature': spec[3],
                            'input_tokens': 10, 'output_tokens': 5, 'confidence': None}
                with ExitStack() as stack:
                    streams = {name: stack.enter_context((folder / (name + '.jsonl')).open('a')) for name in parallel.STORES}
                    raw = stack.enter_context((folder / 'raw.jsonl').open('a'))
                    output = stack.enter_context((folder / 'decisions.jsonl').open('a'))
                    store = parallel.ResponseStore(streams)
                    journal = parallel.RequestJournal(request, raw)
                    acquisition = parallel.Acquisition(store, journal)
                    ledger = parallel.DecisionLedger(output)
                    def make(safe, method):
                        order[safe['id']].append(method)
                        if workers == 1:
                            return serial.make_decision(frozen, models, method, safe, store, acquisition)
                        return parallel.make_decision(frozen, models, method, safe, store, policy_lock, acquisition)
                    with patch.object(serial, 'dispatch', side_effect=dispatch), patch.object(serial.old, 'call', side_effect=fake_call):
                        records = parallel.run_questions(all_rows, make, ledger, workers)
                    self.assertEqual(calls, 16 * 5)
                    self.assertEqual(len(journal.known), calls)
                    self.assertEqual(len(records), 128)
                    for safe in all_rows:
                        self.assertEqual(order[safe['id']], list(parallel.METHODS))
                    for record in records.values():
                        serial.verify_decision(frozen, record, store, all_rows[int(record['id'].rsplit('-', 1)[1])])
                    outcomes.append(records); maximum.append(peak)
                    self.assertNotIn('values', vars(observer))
                self.assertEqual(len(serial.strict_records(folder / 'decisions.jsonl')), 128)
            self.assertEqual(outcomes[0], outcomes[1])
            self.assertEqual(maximum, [1, 4])
            self.assertEqual(observer.maximum, 1)
            self.assertEqual(outcomes[1][all_rows[0]['id'], 'A']['blind_paid_prefix_audit']['encoder_calls'], 1)

    def test_logical_singleflight_and_store_snapshot_under_concurrent_appends(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'plain.jsonl'
            with path.open('a') as stream:
                store = parallel.ResponseStore({'plain': stream})
                acquisition = parallel.Acquisition(store, lambda *args: None)
                count = 0
                def call(safe, path, tree, reviews, judge):
                    nonlocal count
                    count += 1; time.sleep(.005)
                    spec = serial.specification(safe, 'plain', 'solver', path, tree, reviews)
                    return {'text': 'Invalid completed answer', 'answer': '',
                            'prompt_sha256': serial.hashlib.sha256(spec[0].encode()).hexdigest(),
                            'seed': spec[1], 'model_digest': spec[2], 'temperature': spec[3],
                            'input_tokens': 10, 'output_tokens': 5}
                with patch.object(serial.old, 'call', side_effect=call):
                    with ThreadPoolExecutor(max_workers=12) as pool:
                        futures = [pool.submit(acquisition.ensure, row(), 'strong', 'solver', '') for _ in range(12)]
                        self.assertTrue(all(f.result()['answer'] == '' for f in futures))
                    self.assertEqual(count, 1)
                    def append(index):
                        acquisition.ensure(row(index), 'plain', 'solver', '')
                        for _ in range(4):
                            store.view(row()['id'], 'strong')
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        list(pool.map(append, range(1, 33)))
                self.assertEqual(len(store.known['plain']), 33)
            self.assertEqual(len(serial.strict_records(path)), 33)

    def test_raw_response_survives_parser_failure_and_resume_without_double_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'raw.jsonl'
            count = 0
            def request(*args):
                nonlocal count
                count += 1; time.sleep(.005)
                return {'response': 'A completed wrong or invalid answer', 'done': True}
            with path.open('a') as stream:
                journal = parallel.RequestJournal(request, stream)
                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(lambda _: journal('/api/generate', {'prompt': 'same', 'seed': 1}), range(8)))
                self.assertEqual(count, 1)
                results[0]['response'] = 'mutated parser view'
                self.assertEqual(journal('/api/generate', {'prompt': 'same', 'seed': 1})['response'], results[1]['response'])
                with self.assertRaises(ValueError):
                    journal('/api/generate', {'prompt': 'same', 'seed': 1})
                    raise ValueError('synthetic parse failure AFTER raw commit')
            saved = serial.strict_records(path)
            with path.open('a') as stream:
                resumed = parallel.RequestJournal(lambda *args: self.fail('Completed generation retried'), stream, saved)
                self.assertEqual(resumed('/api/generate', {'prompt': 'same', 'seed': 1}), results[1])
            broken = copy.deepcopy(saved[0]); broken['payload']['seed'] = 2
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                parallel.RequestJournal.validate(broken)
            with path.open('ab') as stream:
                stream.write(b'{"partial":')
            with self.assertRaisesRegex(ValueError, 'explicit recovery'):
                serial.strict_records(path)

    def test_request_cap_and_failed_raw_write_do_not_start_duplicate_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'raw.jsonl'
            guard = threading.Lock()
            active = maximum = calls = 0
            def request(endpoint, payload):
                nonlocal active, maximum, calls
                with guard:
                    active += 1; maximum = max(maximum, active); calls += 1
                time.sleep(.01)
                with guard:
                    active -= 1
                return {'response': 'completed', 'done': True}
            with path.open('a') as stream:
                journal = parallel.RequestJournal(request, stream)
                with ThreadPoolExecutor(max_workers=12) as pool:
                    list(pool.map(lambda i: journal('/api/generate', {'prompt': str(i)}), range(12)))
                self.assertEqual(maximum, 4)
                self.assertEqual(calls, 12)
                with patch.object(parallel, 'durable_line', side_effect=OSError('synthetic fsync failure')):
                    with self.assertRaisesRegex(OSError, 'fsync failure'):
                        journal('/api/generate', {'prompt': 'fails-to-persist'})
                    with self.assertRaisesRegex(RuntimeError, 'inspect/resume'):
                        journal('/api/generate', {'prompt': 'fails-to-persist'})
                self.assertEqual(calls, 13)

    def test_every_logical_node_and_review_phase_requires_its_original_raw_receipt(self):
        def raw_node(model, prompt, seed, cap, context):
            payload = {'model': model, 'prompt': prompt,
                       'options': {'seed': seed, 'temperature': 0., 'num_predict': cap, 'num_ctx': context}}
            response = {'done': True, 'model': model, 'response': 'completed ' + prompt,
                        'prompt_eval_count': 20, 'eval_count': 8, 'done_reason': 'stop'}
            raw = {'endpoint': '/api/generate', 'payload': payload, 'response': response,
                   'request_sha256': serial.fingerprint({'endpoint': '/api/generate', 'payload': payload}),
                   'response_sha256': serial.fingerprint(response)}
            node = {'model': model, 'text': response['response'], 'seed': seed, 'temperature': 0.,
                    'prompt_sha256': serial.hashlib.sha256(prompt.encode()).hexdigest(),
                    'output_cap': cap, 'context_length': context, 'input_tokens': 20, 'output_tokens': 8,
                    'finish': 'stop'}
            return raw, node
        root_raw, root = raw_node(serial.old.MODEL, 'root', 1, 128, 4096)
        first_raw, first = raw_node(serial.old.JUDGE, 'thought', 2, 1024, 8192)
        second_raw, second = raw_node(serial.old.JUDGE, 'answer', 3, 512, 8192)
        banks = {'plain': [{'kind': 'solver', 'path': '', 'response': root}],
                 'strong': [{'kind': 'judge', 'path': '', 'response': {'model': serial.old.JUDGE, 'calls': [first, second]}}]}
        raw = [root_raw, first_raw, second_raw]
        coverage = parallel.raw_coverage(banks, raw)
        self.assertEqual(coverage['logical_response_nodes'], 2)
        self.assertEqual(coverage['unique_raw_generations_covered'], 3)
        self.assertEqual(coverage['physical_response_nodes_including_exact_reuse'], 3)
        for missing in range(3):
            with self.assertRaisesRegex(ValueError, 'no matching durable'):
                parallel.raw_coverage(banks, raw[:missing] + raw[missing + 1:])
        changed = copy.deepcopy(banks)
        changed['strong'][0]['response']['calls'][0]['input_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'no matching durable'):
            parallel.raw_coverage(changed, raw)
        for field, value in (('thinking', 'a changed paid rationale'), ('finish', 'length')):
            changed = copy.deepcopy(banks)
            changed['strong'][0]['response']['calls'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'no matching durable'):
                parallel.raw_coverage(changed, raw)

    def test_failure_joins_running_workers_and_resume_keeps_completed_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'decisions.jsonl'
            failures = {'remaining': 1}
            calls = []
            def make(safe, method):
                calls.append((safe['id'], method))
                if safe['id'] == row()['id'] and method == 'A' and failures['remaining']:
                    failures['remaining'] -= 1
                    raise RuntimeError('synthetic interruption')
                time.sleep(.001)
                return {'id': safe['id'], 'method': method, 'item': {'compute': 10}}
            with path.open('a') as stream:
                ledger = parallel.DecisionLedger(stream)
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    parallel.run_questions([row(i) for i in range(8)], make, ledger)
                snapshot = ledger.snapshot()
            saved = serial.strict_records(path)
            self.assertEqual(len(saved), len(snapshot))
            calls.clear()
            with path.open('a') as stream:
                result = parallel.run_questions([row(i) for i in range(8)], make, parallel.DecisionLedger(stream, saved))
            self.assertEqual(len(result), 64)
            self.assertFalse(set(calls) & set(snapshot))
            self.assertEqual(len(serial.strict_records(path)), 64)
            with self.assertRaisesRegex(ValueError, 'unique public-only'):
                parallel.run_questions([row(), row()], make, parallel.DecisionLedger())
            with self.assertRaisesRegex(ValueError, 'unique public-only'):
                parallel.run_questions([{**row(), 'answer': 'secret'}], make, parallel.DecisionLedger())
            with self.assertRaisesRegex(ValueError, 'unique public-only'):
                parallel.run_questions([{**row(), 'answer': 0}], make, parallel.DecisionLedger())

    def test_matched_review_one_or_two_phases_preserves_paid_cost(self):
        for phases in (1, 2):
            with self.subTest(phases=phases), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                calls = []
                def request(endpoint, payload):
                    calls.append(payload)
                    cap = payload['options']['num_predict']
                    if phases == 2 and cap == 1024:
                        text = 'bounded thought without a complete answer'
                    elif phases == 2:
                        text = '2", "p_correct": 0.1, "action": "repair", "check": "Check addition"}'
                    else:
                        text = json.dumps({'answer': '2', 'p_correct': .1, 'action': 'repair', 'check': 'Check addition'})
                    return {'done': True, 'model': serial.old.JUDGE, 'response': text,
                            'prompt_eval_count': 10, 'eval_count': 20, 'done_reason': 'stop'}
                def call(safe, path, tree, reviews, judge):
                    spec = serial.specification(safe, 'plain', 'solver', path, tree, reviews)
                    return {'text': 'Final: 1', 'answer': '1', 'input_tokens': 10, 'output_tokens': 5,
                            'prompt_sha256': serial.hashlib.sha256(spec[0].encode()).hexdigest(),
                            'seed': spec[1], 'temperature': spec[3], 'model_digest': spec[2]}
                def actor(request, safe, prompt, seed, cap):
                    return {'text': 'Final: 2', 'answer': '2', 'input_tokens': 15, 'output_tokens': 5,
                            'prompt_sha256': serial.hashlib.sha256(prompt.encode()).hexdigest(),
                            'seed': seed, 'temperature': .65, 'model_digest': serial.old.MODEL_DIGEST, 'output_cap': cap}
                frozen = {'observer_assets': {}, 'selected': {'B': {'protocol': 'matched-retained-1024-512-128',
                    'settings': {'root_stop': 'agreement', 'second_stop': 'agreement', 'second_action': 'reuse'}}}}
                with ExitStack() as stack:
                    streams = {name: stack.enter_context((folder / name).open('a')) for name in parallel.STORES}
                    raw = stack.enter_context((folder / 'raw').open('a'))
                    store = parallel.ResponseStore(streams)
                    acquisition = parallel.Acquisition(store, parallel.RequestJournal(request, raw))
                    with patch.object(serial.old, 'call', side_effect=call), patch.object(serial, 'actor', side_effect=actor):
                        result = parallel.make_decision(frozen, {}, 'B', row(), store, threading.Lock(), acquisition)
                        reference = serial.make_decision(frozen, {}, 'B', row(), store)
                    self.assertEqual(result, reference)
                    self.assertEqual(len(calls), phases)
                    self.assertEqual(result['item']['judge_physical_calls'], phases)
                    self.assertEqual(result['item']['judge_tokens'], 30 * phases)
                    self.assertEqual(result['item']['solver_calls'], 2)

    def test_registration_requires_frozen_archive_and_exact_runtime_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            project = parallel.HERE.parents[3]
            paths = [Path(parallel.__file__), parallel.HERE / 'prospective_collect.py',
                     parallel.HERE / 'prospective_audit.py', parallel.HERE / 'prospective_grade.py',
                     parallel.HERE / 'prospective_freeze.py']
            sources = {}
            for path in paths:
                relative = str(path.relative_to(project))
                sources[relative] = parallel.sha(path)
                destination = study / 'runtime' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
            frozen = {'project_directory': str(project), 'runtime_sources': sources}
            payload = parallel.protocol_payload(frozen, study, 'freeze')
            self.assertEqual(payload['question_workers'], 4)
            self.assertEqual(payload['max_cloud_containers'], 2)
            self.assertEqual(payload['max_cloud_inputs'], 4)
            self.assertEqual(payload['methods_in_question_order'], list(serial.METHODS))
            self.assertIn('audit-development', payload['development_route'])
            registration = study / 'parallel-protocol'; registration.mkdir()
            (registration / 'prospective_parallel.py').write_bytes(Path(parallel.__file__).read_bytes())
            serial.atomic_json(registration / 'registration.json', payload)
            args = SimpleNamespace(study=study, freeze_sha256='freeze', protocol_sha256=parallel.sha(registration / 'registration.json'))
            with patch.object(parallel, 'verify', return_value=frozen):
                self.assertEqual(parallel.verify_protocol(args), frozen)
                payload['question_workers'] = 8
                serial.atomic_json(registration / 'registration.json', payload)
                with self.assertRaisesRegex(ValueError, 'externally recorded'):
                    parallel.verify_protocol(args)
                args.protocol_sha256 = parallel.sha(registration / 'registration.json')
                with self.assertRaisesRegex(ValueError, 'contract changed'):
                    parallel.verify_protocol(args)
            del sources[str(Path(parallel.__file__).relative_to(project))]
            with self.assertRaisesRegex(ValueError, 'original policy freeze'):
                parallel.protocol_payload(frozen, study, 'freeze')

    def test_full_synthetic_grid_and_no_early_grading_route(self):
        rows = [row(i) for i in range(1024)]
        with patch('builtins.print'):
            records = parallel.run_questions(rows, lambda safe, method: {'id': safe['id'], 'method': method},
                                             parallel.DecisionLedger())
        self.assertEqual(len(records), 8192)
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            target = study / 'collection'; target.mkdir()
            serial.atomic_json(target / 'complete.json', {'parallel_protocol_sha256': 'wrong'})
            serial.atomic_json(target / 'collection-audit.json', {})
            args = SimpleNamespace(study=study, protocol_sha256='registered', freeze_sha256='freeze')
            with patch.object(parallel, 'verify_protocol'), patch.object(parallel, 'verify_development'):
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    parallel.grade(args)


if __name__ == '__main__':
    unittest.main()
