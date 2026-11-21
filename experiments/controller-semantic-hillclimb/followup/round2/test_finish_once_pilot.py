import builtins
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import finish_once_pilot as run
from test_science_mentor_expansion import row, cohort, plain, Remote


class FinishTests(unittest.TestCase):
    def test_sealed_prefix_retains_hash_but_selects_only_seven_plain_controls(self):
        with TemporaryDirectory() as folder:
            data = Path(folder); safe = row(); tree = plain(safe)
            records = [{'id': safe['id'], 'kind': 'judge', 'path': '', 'response': {'text': 'Unrelated paid review'}},
                       {'id': safe['id'], 'kind': 'solver', 'path': 'guided', 'response': {'text': 'Unrelated guided'}}]
            records += [{'id': safe['id'], 'kind': 'solver', 'path': p, 'response': n} for p, n in tree.items()]
            bank = data/'train-observations.jsonl'; bank.write_text(''.join(json.dumps(r)+'\n' for r in records))
            run.write(data/'train-plain-complete.json', {'base_records': len(records), 'base_bank_sha256': run.sha(bank)})
            result, digest = run.plain_controls(data, {safe['id']})
            self.assertEqual(result[safe['id']], tree); self.assertEqual(digest, run.sha(bank))
            with bank.open('a') as stream: stream.write(json.dumps({'later': 'Unsealed ordinary append'})+'\n')
            self.assertEqual(run.plain_controls(data, {safe['id']})[0], result)
            records[0]['response']['text'] = 'Changed sealed review'
            bank.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaisesRegex(ValueError, 'sealed'): run.plain_controls(data, {safe['id']})

    def test_fixed_membership_and_question_only_public_loader(self):
        rows, contract = cohort()
        first = run.fixed.selection(rows, contract)
        changed = [{**r, 'answer': 'SECRET', 'correct': i % 2} for i, r in enumerate(rows)]
        self.assertEqual(first, run.fixed.selection(changed, contract))
        self.assertEqual(len(first), 128)
        with TemporaryDirectory() as folder:
            data = Path(folder); (data/'splits').mkdir()
            raw = [json.dumps({k: v for k, v in r.items() if k != 'answer'})[:-1]+',"answer": PRIVATE_SECRET}' for r in rows]
            (data/'splits/train-questions.jsonl').write_text('\n'.join(raw)+'\n')
            self.assertEqual(run.gates.public_train(data), rows)

    def test_prompt_keeps_reasoning_before_final_and_only_paid_root(self):
        safe = row(); root = {'text': 'Paid reasoning ending in an unfinished calculation', 'answer': 'A', 'unpaid': 'SECRET'}
        text = run.prompt(safe, root)
        self.assertIn(root['text'], text); self.assertNotIn('SECRET', text)
        self.assertIn('Do not restart', text)
        self.assertIn('90 words of additional scratch work, then exactly one line', text)
        self.assertNotIn('FIRST', text)
        self.assertEqual(text, run.prompt({**safe, 'unpaid': 'SECRET'}, {**root, 'unpaid': 'CHANGED'}))
        with self.assertRaises(ValueError): run.prompt({**safe, 'answer': 'SECRET'}, root)

    def test_exact_seed_caps_invalid_preservation_and_duplicate_singleflight(self):
        safe = row(); original = {safe['id']: plain(safe)}
        with TemporaryDirectory() as folder:
            directory = Path(folder); remote = Remote(); known = {}
            with (directory/'bank').open('a') as bank, (directory/'raw').open('a') as raw:
                request = run.RequestJournal(remote, raw, slots=2)
                collector = run.Collector(original, known, bank, request)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(collector.one, [safe, safe]))
            self.assertEqual(len(remote.calls), 1)
            self.assertEqual(remote.calls[0]['options'], {'temperature': .65, 'seed': run.fixed.logical_seed(safe['id'], 'repair'),
                                                        'num_predict': 256, 'num_ctx': 4096})
            records = run.protocol.strict_records(directory/'bank'); raw = run.protocol.strict_records(directory/'raw')
            restored, coverage = run.restore([safe], original, records, raw)
            self.assertEqual(restored[safe['id']]['response']['answer'], '')
            self.assertEqual(coverage['unique_raw_generations_covered'], 1)
            tampered = copy.deepcopy(records); tampered[0]['root_sha256'] = 'changed'
            with self.assertRaises(ValueError): run.restore([safe], original, tampered, raw)
            with self.assertRaises(ValueError): run.restore([safe], original, records, [])

    def test_returned_receipt_survives_failure_before_logical_append(self):
        safe = row(); original = {safe['id']: plain(safe)}
        with TemporaryDirectory() as folder:
            directory = Path(folder); remote = Remote()
            with (directory/'bank').open('a') as bank, (directory/'raw').open('a') as raw:
                request = run.RequestJournal(remote, raw)
                collector = run.Collector(original, {}, bank, request)
                with patch.object(run, 'durable_line', side_effect=RuntimeError('Simulated logical append failure')):
                    with self.assertRaises(RuntimeError): collector.one(safe)
            returned = run.protocol.strict_records(directory/'raw')
            self.assertEqual(len(returned), 1)
            with (directory/'bank').open('a') as bank, (directory/'raw').open('a') as raw:
                request = run.RequestJournal(lambda *a: self.fail('Do not regenerate'), raw, returned)
                collector = run.Collector(original, {}, bank, request); collector.one(safe)
            restored, _ = run.restore([safe], original, run.protocol.strict_records(directory/'bank'), returned)
            self.assertEqual(len(restored), 1); self.assertEqual(len(remote.calls), 1)

    def test_mandatory_gate_and_no_cloud_import_before_completion(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); args = SimpleNamespace(data=root/'data', priority_completion=[])
            with patch.object(run.gates, 'public_train', side_effect=AssertionError('Do not read inputs')):
                with self.assertRaisesRegex(ValueError, 'exact final'): run.inputs(args)
            output = root/'output'; output.mkdir()
            run.write(output/'registration.json', {'source_files': {}})
            args = SimpleNamespace(output=output, registration_sha256=run.sha(output/'registration.json'), wait=False)
            original_import = builtins.__import__
            def forbidden(name, *a, **kw):
                if name == 'modal': self.fail('Cloud client loaded before gates')
                return original_import(name, *a, **kw)
            with patch.object(run, 'registration', return_value={'source_files': {}}), \
                    patch.object(run.gates, 'prerequisite_receipts', return_value=({}, ['unfinished'])), \
                    patch('builtins.__import__', side_effect=forbidden):
                with self.assertRaisesRegex(ValueError, 'finish first'): run.collect(args)

    def test_source_manifest_excludes_unrelated_tests_and_captures_frozen_runtime(self):
        manifest = run.sources()
        self.assertIn(str(Path(run.__file__).resolve()), manifest)
        self.assertIn(str(Path(run.fixed.__file__).resolve()), manifest)
        self.assertIn(str(Path(run.gates.__file__).resolve()), manifest)
        self.assertNotIn(str(Path(__file__).resolve()), manifest)
        self.assertFalse(any('/.venv/' in path or '/test_' in path for path in manifest))

    def test_no_reference_read_before_complete_grid_and_seal(self):
        with TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder))
            with patch.object(run.fixed, 'lines', side_effect=AssertionError('No references before seal')):
                with self.assertRaises(FileNotFoundError): run.grade(args)
                run.write(args.output/'complete.json', {'questions': 127, 'physical_solver_calls': 127})
                with self.assertRaisesRegex(ValueError, 'partial'): run.grade(args)

    def test_full_grade_counts_fixed_controls_heuristics_novelty_and_costs(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder); (directory/'splits').mkdir()
            pool, contract = cohort(); chosen = run.fixed.selection(pool, contract)
            by_id = {r['id']: r for r in pool}; rows = [by_id[r['id']] for r in chosen]
            original, refs = {}, []
            for i, safe in enumerate(rows):
                answer = '1' if safe['domain'] == 'math' else 'A'
                tree = plain(safe)
                for node in tree.values():
                    node['answer'] = ('2' if safe['domain'] == 'math' else 'B') if i < 8 else answer
                    node['text'] = 'Final: '+node['answer']
                if i < 4: tree['']['answer'] = ''; tree['']['text'] = 'Unfinished work'
                original[safe['id']] = tree; refs.append({**safe, 'answer': answer})
            (directory/'splits/train-questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs))
            def remote(endpoint, payload):
                answer = '1' if 'where X is a number' in payload['prompt'] else 'A'
                text = 'Completed calculation.\nFinal: '+answer
                return {'done': True, 'model': run.protocol.old.MODEL, 'response': text, 'prompt_eval_count': 8,
                        'eval_count': 8, 'done_reason': 'stop', 'logprobs': [{'token': text, 'logprob': -.1}]}
            with (directory/'train.jsonl').open('a') as bank, (directory/'raw-generations.jsonl').open('a') as raw:
                journal = run.RequestJournal(remote, raw)
                collector = run.Collector(original, {}, bank, journal)
                for safe in rows: collector.one(safe)
            registration = {'scope': 'NEW protocol', 'promotion_requires': 'Matched controls and fresh validation'}
            run.write(directory/'registration.json', registration); run.write(directory/'prerequisites.json', {})
            saved, raw = run.protocol.strict_records(directory/'train.jsonl'), run.protocol.strict_records(directory/'raw-generations.jsonl')
            _, coverage = run.restore(rows, original, saved, raw)
            complete = {'questions': 128, 'physical_solver_calls': 128, 'coverage': coverage,
                'transport_errors_sha256': None,
                'registration_sha256': run.sha(directory/'registration.json'), 'bank_sha256': run.sha(directory/'train.jsonl'),
                'raw_sha256': run.sha(directory/'raw-generations.jsonl'), 'prerequisites_sha256': run.sha(directory/'prerequisites.json')}
            run.write(directory/'complete.json', complete)
            args = SimpleNamespace(output=directory, data=directory)
            with patch.object(run, 'registration', return_value=registration), patch.object(run, 'inputs', return_value=(rows, original)), patch('builtins.print'):
                run.grade(args)
            result = run.read(directory/'summary.json'); metrics = result['panels']['all']['methods']
            self.assertEqual(metrics['Finish once']['correct'], 128)
            self.assertEqual(metrics['Repair once']['correct'], 120)
            self.assertEqual(metrics['Valid-stop else Finish']['correct'], 124)
            self.assertEqual(metrics['Valid-stop else Repair']['correct'], 120)
            self.assertEqual(result['panels']['all']['new_correct_beyond_original_seven_path_oracle'], 8)
            self.assertAlmostEqual(metrics['Finish once']['mean_compute'], 4*(23+16)/3.09)
            self.assertTrue(result['scaling_threshold_met'])
            self.assertTrue(result['test_used'] is False)


if __name__ == '__main__':
    unittest.main()
