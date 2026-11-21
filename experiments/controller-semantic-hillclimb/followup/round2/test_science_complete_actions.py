"""Synthetic-only terminal action grid/paired-cost/seal tests."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import science_complete_actions as run
from test_scoring_completion_availability import canonical
from test_scoring_observation import plain, response, Remote


def raw_for(safe, prefix, tree):
    payload = run.pilot.request_payload(safe, prefix, tree)[0]
    result = response()
    return {'endpoint': '/api/generate', 'payload': payload,
        'request_sha256': run.digest({'endpoint': '/api/generate', 'payload': payload}),
        'response': result, 'response_sha256': run.digest(result),
        'started_unix': 1., 'returned_unix': 1.25}


def fixture():
    rows = [canonical(i, subject) for subject in run.COUNTS for i in range(128)]
    trees = {r['id']: plain(r) for r in rows}
    chosen = [r for r in rows if int(r['id'].rsplit('-', 1)[1]) < 32]
    raw = [raw_for(r, prefix, trees[r['id']]) for r in chosen for prefix in run.PREFIXES]
    imported = [run.logical_record(r, prefix, trees[r['id']], item, True)
        for (r, prefix), item in zip([(r, p) for r in chosen for p in run.PREFIXES], raw)]
    old_ids = {r['id'] for r in chosen}
    missing = [{'id': r['id'], 'subject': r['subject'], 'prefix_path': p, 'path': run.action_path(p),
        'payload_sha256': run.digest(run.pilot.request_payload(r, p, trees[r['id']])[0])}
        for r in rows if r['id'] not in old_ids for p in run.PREFIXES]
    evidence = {'files': {}, 'sealed_plain_prefix_sha256': 'synthetic',
        'original_pilot_coverage': {'physical_calls': 192}, 'original_pilot_transport_errors': 0,
        'pilot_ids_sha256': run.digest(sorted(old_ids)), 'all_science_ids_sha256': run.digest(sorted(trees))}
    return rows, trees, imported, raw, missing, evidence


class CompleteActionTests(unittest.TestCase):
    def test_terminal_budget_exact_request_and_no_failed_score_features(self):
        safe = canonical(); tree = plain(safe)
        for prefix in run.PREFIXES:
            raw = raw_for(safe, prefix, tree)
            item = run.logical_record(safe, prefix, tree, raw, False)
            self.assertEqual(item['path'], run.action_path(prefix))
            self.assertEqual(item['response']['total_answer_changing_continuations'], 1 if not prefix else 2)
            self.assertTrue(item['response']['terminal'])
            self.assertEqual(item['response']['additional_answer_changing_calls'], 1)
            self.assertNotIn('features', item); self.assertNotIn('features', item['response'])
            self.assertEqual(item['response']['answer'], 'B')
            self.assertEqual(item['response']['input_tokens'], 100)
            self.assertEqual(raw['payload']['options']['num_predict'], 1)
        for prefix in ('complete', 'repair/repair', 'retry/complete'):
            with self.assertRaises(ValueError): run.action_path(prefix)
        bad = raw_for(safe, '', tree); bad['response']['response'] = 'Final: B'
        bad['response_sha256'] = run.digest(bad['response'])
        self.assertEqual(run.logical_record(safe, '', tree, bad, False)['response']['answer'], '')

    def test_request_uses_only_paid_prefix_and_math_is_rejected(self):
        safe = canonical(); tree = plain(safe)
        raw = raw_for(safe, 'repair', tree)
        before = run.logical_record(safe, 'repair', tree, raw, False)
        changed = copy.deepcopy(tree)
        for path in ('retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry'):
            changed[path]['text'] = 'FUTURE_PRIVATE'; changed[path]['answer'] = 'A'
        self.assertEqual(before, run.logical_record(safe, 'repair', changed, raw, False))
        with self.assertRaises(ValueError): run.logical_record({**safe, 'answer': 'PRIVATE_REFERENCE'}, 'repair', tree, raw, False)
        with self.assertRaises(ValueError): run.logical_record({**safe, 'domain': 'math'}, 'repair', tree, raw, False)

    def test_register_preserves_all192_raw_bytes_and_import_identity(self):
        values = fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory); parent = root/'pilot'; parent.mkdir()
            raw_bytes = ''.join(json.dumps(v)+'\n' for v in values[3]).encode()
            (parent/'raw-generations.jsonl').write_bytes(raw_bytes)
            args = SimpleNamespace(data=root/'data', pilot=parent, output=root/'grid')
            with patch.object(run, 'public_inputs', return_value=values):
                run.register(args)
                self.assertEqual((args.output/'raw-generations.jsonl').read_bytes(), raw_bytes)
                rows, trees, reused, missing = run.registered_view(args)
                self.assertEqual((len(rows), len(reused), len(missing)), (256, 192, 576))
                self.assertEqual(len(run.pilot.protocol.strict_records(args.output/'train.jsonl')), 192)
                raw = bytearray((args.output/'raw-generations.jsonl').read_bytes()); raw[2] ^= 1
                (args.output/'raw-generations.jsonl').write_bytes(raw)
                with self.assertRaises(ValueError): run.registered_view(args)

    def test_entire768_grid_reuses192_and_acquires_exact576_without_outcome_filter(self):
        rows, trees, imported, raw, missing, _ = fixture()
        reused = {v['request_sha256'] for v in raw}
        with TemporaryDirectory() as directory:
            output = Path(directory); remote = Remote()
            (output/'raw').write_text(''.join(json.dumps(v)+'\n' for v in raw))
            (output/'bank').write_text(''.join(json.dumps(v)+'\n' for v in imported))
            known, _ = run.restore(rows, trees, reused, imported, raw)
            with (output/'bank').open('a') as bank, (output/'raw').open('a') as stream:
                journal = run.pilot.RequestJournal(remote, stream, raw, slots=2)
                collector = run.Collector(trees, reused, known, bank, journal)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(collector.one, [(r, p) for r in rows for p in run.PREFIXES]))
            self.assertEqual(len(remote.calls), 576)
            records, physical = run.pilot.protocol.strict_records(output/'bank'), run.pilot.protocol.strict_records(output/'raw')
            known, coverage = run.restore(rows, trees, reused, records, physical, True)
            self.assertEqual(coverage, {'expected_cells': 768, 'physical_calls': 768, 'logical_actions': 768,
                'reused_physical_calls': 192, 'new_physical_calls': 576})
            self.assertEqual(physical[:192], raw); self.assertEqual(len(known), 768)
            with self.assertRaises(ValueError): run.restore(rows, trees, reused, records[:-1], physical, True)
            with self.assertRaises(ValueError): run.restore(rows, trees, reused, records, physical[:-1], True)
            with self.assertRaises(ValueError): run.restore(rows, trees, reused, records+records[:1], physical)
            bad = copy.deepcopy(records); bad[-1]['reused_pilot_physical_response'] = True
            with self.assertRaises(ValueError): run.restore(rows, trees, reused, bad, physical)

    def test_raw_before_logical_crash_resume_never_reforwards_completed_token(self):
        safe = canonical(); trees = {safe['id']: plain(safe)}
        with TemporaryDirectory() as directory:
            output = Path(directory); remote = Remote()
            with (output/'bank').open('a') as bank, (output/'raw').open('a') as raw:
                collector = run.Collector(trees, set(), {}, bank, run.pilot.RequestJournal(remote, raw))
                with patch.object(run.pilot, 'durable_line', side_effect=RuntimeError('After physical receipt')):
                    with self.assertRaises(RuntimeError): collector.one((safe, ''))
            physical = run.pilot.protocol.strict_records(output/'raw')
            self.assertEqual(len(physical), 1)
            with (output/'bank').open('a') as bank, (output/'raw').open('a') as raw:
                journal = run.pilot.RequestJournal(lambda *a: self.fail('No replacement'), raw, physical)
                run.Collector(trees, set(), {}, bank, journal).one((safe, ''))
            self.assertEqual(len(remote.calls), 1)
            run.restore([safe], trees, set(), run.pilot.protocol.strict_records(output/'bank'), physical)

    def test_training_api_blinds_labels_until_complete_and_uses_explicit_grid_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); data, output = root/'data', root/'grid'; output.mkdir()
            run.write(output/'registration.json', {'paths': {'data': str(data), 'pilot': str(root/'pilot'), 'output': str(output)}})
            with patch.object(run.pilot.fixed, 'lines', side_effect=AssertionError('No labels before full768 seal')):
                with self.assertRaises(FileNotFoundError): run.training_data(data, output)
                with self.assertRaises(ValueError): run.training_data(root/'wrong-data', output)

    def test_fixed_controls_full_cost_and_gate_only_on16novel_all256_questions(self):
        rows, trees, _, raw, _, _ = fixture()
        pilot_ids = {r['id'] for r in rows if int(r['id'].rsplit('-', 1)[1]) < 32}
        labeled = []
        for i, safe in enumerate(rows):
            qid = safe['id']; labeled.append({**safe, 'answer': 'A' if i < 16 else 'B'})
            for prefix in run.PREFIXES:
                node = run.logical_record(safe, prefix, trees[qid], raw_for(safe, prefix, trees[qid]), qid in pilot_ids)['response']
                if i < 16: node['answer'] = 'A'
                trees[qid][run.action_path(prefix)] = node
        panels, outcomes, screen = run.aggregate(labeled, trees, pilot_ids)
        self.assertTrue(screen['passed']); self.assertFalse(screen['controller_promoted'])
        values = panels['all_science']; self.assertEqual(len(values['methods']), 11)
        self.assertEqual(values['original_seven_oracle_correct'], 240)
        self.assertEqual(values['combined_ten_oracle_correct'], 256)
        self.assertEqual(values['novel_correct_beyond_original_seven'], 16)
        self.assertEqual(panels['original_pilot64']['questions'], 64)
        self.assertEqual(panels['remaining192']['questions'], 192)
        self.assertEqual(values['methods']['Complete']['mean_tokens'], 124.)
        self.assertEqual(values['methods']['Repair then Complete']['mean_tokens'], 147.)
        self.assertEqual(values['methods']['Stop if valid else Complete']['mean_solver_calls'], 1.)
        self.assertTrue(all(v['methods']['Complete']['answer_changing_continuations'] == 1 for v in outcomes))
        for prefix in run.PREFIXES: trees[rows[0]['id']][run.action_path(prefix)]['answer'] = 'B'
        self.assertFalse(run.aggregate(labeled, trees, pilot_ids)[2]['passed'])

    def test_dependency_manifest_keeps_all_frozen_parsers_and_no_ambient_tests(self):
        paths = run.sources()
        self.assertIn(str(Path(run.__file__).resolve()), paths)
        self.assertIn(str(Path(run.secondary.__file__).resolve()), paths)
        self.assertIn(str(Path(run.pilot.__file__).resolve()), paths)
        self.assertNotIn(str(Path(__file__).resolve()), paths)
        self.assertFalse(any('/test_' in name or '/.venv/' in name for name in paths))


if __name__ == '__main__':
    unittest.main()
