"""Synthetic paid-prefix/resume/cost tests; no cloud, neural or private data."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import drop_review_repair_pilot as run
from test_science_mentor_expansion import row, plain, cohort, Remote


def fixture(n=64):
    rows = [row(i, subject) for subject in ('chemistry', 'physics') for i in range(n//2)]
    trees, reviews = {}, {}
    for safe in rows:
        tree = plain(safe)
        for path in run.EXTRA:
            tree[path] = {**copy.deepcopy(tree['repair']), 'text': 'Paid strong current answer. Final: B'}
        tree['guided/retry'] = tree['repair/retry']
        trees[safe['id']] = tree
        reviews[safe['id'], ''] = {'text': 'PRIVATE_MENTOR_FEEDBACK', 'answer': 'B',
            'input_tokens': 100, 'output_tokens': 30, 'physical_api_calls': 2}
    return rows, trees, reviews


class DropReviewTests(unittest.TestCase):
    def test_exact_standard_prompt_seed_caps_and_no_feedback(self):
        rows, trees, reviews = fixture(2); safe = rows[0]; tree = trees[safe['id']]
        payload = run.request_payload(safe, tree)
        self.assertEqual(payload['prompt'], run.protocol.old.repair_prompt(safe, tree['guided'], ''))
        self.assertIn(tree['guided']['text'], payload['prompt'])
        self.assertNotIn('PRIVATE_MENTOR_FEEDBACK', payload['prompt'])
        changed = copy.deepcopy(tree); changed['guided/reuse']['text'] = 'UNPAID_FUTURE'
        self.assertEqual(payload, run.request_payload(safe, changed))
        self.assertEqual(payload['options'], {'temperature': .65,
            'seed': run.fixed.logical_seed(safe['id'], 'repair/repair'), 'num_predict': 256, 'num_ctx': 4096})
        self.assertIn('at most 90 words', payload['prompt'])
        with self.assertRaises(ValueError): run.request_payload({**safe, 'answer': 'PRIVATE_REFERENCE'}, tree)

    def test_fixed64_membership_never_depends_on_labels(self):
        rows, contract = cohort()
        chosen = [v for v in run.fixed.selection(rows, contract) if v['subject'] in run.COUNTS]
        altered = [{**r, 'answer': 'PRIVATE_REFERENCE', 'correct': i % 2} for i, r in enumerate(rows)]
        self.assertEqual(chosen, [v for v in run.fixed.selection(altered, contract) if v['subject'] in run.COUNTS])
        self.assertEqual(len(chosen), 64)

    def test_singleflight_invalid_answer_preserved_and_full_raw_verification(self):
        rows, trees, reviews = fixture(2); safe = rows[0]
        with TemporaryDirectory() as directory:
            folder = Path(directory); remote = Remote(); known = {}
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw:
                collector = run.Collector(trees, reviews, known, bank, run.RequestJournal(remote, raw, slots=2))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(collector.one, [safe, safe]))
            self.assertEqual(len(remote.calls), 1)
            records, raw = run.protocol.strict_records(folder/'bank'), run.protocol.strict_records(folder/'raw')
            known, _ = run.restore([safe], trees, reviews, records, raw, True)
            self.assertEqual(known[safe['id']]['response']['answer'], '')
            for field in ('paid_prefix', 'response', 'raw_response_sha256'):
                bad = copy.deepcopy(records); bad[0][field] = 'tampered'
                with self.assertRaises(ValueError): run.restore([safe], trees, reviews, bad, raw)
            bad = copy.deepcopy(raw); bad[0]['payload']['options']['num_predict'] = 128
            bad[0]['request_sha256'] = run.digest({'endpoint': bad[0]['endpoint'], 'payload': bad[0]['payload']})
            with self.assertRaises(ValueError): run.restore([safe], trees, reviews, records, bad)

    def test_raw_survives_logical_failure_without_regeneration(self):
        rows, trees, reviews = fixture(2); safe = rows[0]
        with TemporaryDirectory() as directory:
            folder = Path(directory); remote = Remote()
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw:
                collector = run.Collector(trees, reviews, {}, bank, run.RequestJournal(remote, raw))
                with patch.object(run, 'durable_line', side_effect=RuntimeError('Append failed')):
                    with self.assertRaises(RuntimeError): collector.one(safe)
            receipts = run.protocol.strict_records(folder/'raw'); self.assertEqual(len(receipts), 1)
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw:
                journal = run.RequestJournal(lambda *a: self.fail('Never repeat a completed generation'), raw, receipts)
                run.Collector(trees, reviews, {}, bank, journal).one(safe)
            run.restore([safe], trees, reviews, run.protocol.strict_records(folder/'bank'), receipts, True)
            self.assertEqual(len(remote.calls), 1)
            with (folder/'bank').open('a') as stream: stream.write('{partial')
            with self.assertRaises(ValueError): run.protocol.strict_records(folder/'bank')

    def test_bad_physical_receipt_is_durable_and_fails_before_logical(self):
        rows, trees, reviews = fixture(2); safe = rows[0]; remote = Remote()
        def bad(endpoint, payload):
            value = remote(endpoint, payload); value['eval_count'] = 257; return value
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw:
                with self.assertRaises(ValueError):
                    run.Collector(trees, reviews, {}, bank, run.RequestJournal(bad, raw)).one(safe)
            self.assertEqual(len(run.protocol.strict_records(folder/'raw')), 1)
            self.assertEqual(run.protocol.strict_records(folder/'bank'), [])
            with self.assertRaises(ValueError):
                run.restore([safe], trees, reviews, [], run.protocol.strict_records(folder/'raw'))

    def test_full64_grid_and_no_labels_before_valid_seal(self):
        rows, trees, reviews = fixture()
        with TemporaryDirectory() as directory:
            folder = Path(directory); remote = Remote()
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw:
                collector = run.Collector(trees, reviews, {}, bank, run.RequestJournal(remote, raw))
                for safe in rows: collector.one(safe)
            records, raw = run.protocol.strict_records(folder/'bank'), run.protocol.strict_records(folder/'raw')
            _, coverage = run.restore(rows, trees, reviews, records, raw, True)
            self.assertEqual(coverage, {'questions': 64, 'physical_calls': 64, 'logical_repairs': 64})
            with self.assertRaises(ValueError): run.restore(rows, trees, reviews, records[:-1], raw, True)
            with self.assertRaises(ValueError): run.restore(rows, trees, reviews, records, raw[:-1], True)
            with self.assertRaises(ValueError): run.restore(rows, trees, reviews, records+records[:1], raw)
            with patch.object(run.fixed, 'lines', side_effect=AssertionError('No labels before seal')):
                with self.assertRaises(FileNotFoundError): run.grade(SimpleNamespace(output=folder))

    def test_full_cost_and_novelty_screen_are_distinct_from_accuracy_gain(self):
        rows, trees, reviews = fixture(); known = {}; references = {r['id']: 'A' for r in rows}
        for i, safe in enumerate(rows):
            qid = safe['id']; tree = trees[qid]
            for node in tree.values(): node['answer'] = 'A' if i >= 8 else 'B'
            known[qid] = {'response': {**tree['guided/reuse'], 'answer': 'A' if i < 6 or i >= 8 else 'B'}}
        reports, outcomes, screen = run.aggregate(rows, trees, reviews, known, references)
        all_rows = reports['all_science']
        self.assertEqual(all_rows['existing_ten_path_oracle_correct'], 56)
        self.assertEqual(all_rows['novel_correct_beyond_existing_ten'], 6)
        self.assertEqual(all_rows['combined_eleven_path_oracle_correct'], 62)
        self.assertTrue(screen['passed']); self.assertFalse(screen['controller_promoted'])
        methods = all_rows['methods']; self.assertEqual(len(methods), 9)
        self.assertEqual(methods['Drop-review standard second repair']['correct'], 62)
        expected = (4*69 + 3*130)/3.09
        self.assertAlmostEqual(methods['Drop-review standard second repair']['mean_compute'], expected)
        self.assertEqual(methods['Drop-review standard second repair']['mean_physical_review_calls'], 2.)
        self.assertEqual(methods['Repair twice']['mean_physical_review_calls'], 0.)
        self.assertEqual(all_rows['drop_review_paired_vs']['Strong retained second repair'], {'wins': 6, 'losses': 0})
        # Matching the retained path exactly is cheap but offers no new headroom.
        for safe in rows: known[safe['id']]['response']['answer'] = trees[safe['id']]['guided/reuse']['answer']
        self.assertFalse(run.aggregate(rows, trees, reviews, known, references)[2]['passed'])
        # Six new successes can still fail the full-trajectory compute screen.
        for i, safe in enumerate(rows):
            known[safe['id']]['response']['answer'] = 'A'
            known[safe['id']]['response']['input_tokens'] = 1000
        result = run.aggregate(rows, trees, reviews, known, references)[2]
        self.assertTrue(result['checks']['novel_correct_at_least6'])
        self.assertFalse(result['checks']['mean_full_compute_at_most_1p10_retained'])

    def test_source_manifest_captures_runtime_but_not_ambient_tests(self):
        manifest = run.sources()
        self.assertIn(str(Path(run.__file__).resolve()), manifest)
        self.assertIn(str(Path(run.protocol.__file__).resolve()), manifest)
        self.assertIn(str(Path(run.fixed.__file__).resolve()), manifest)
        self.assertFalse(any('/test_' in path or '/.venv/' in path for path in manifest))


if __name__ == '__main__':
    unittest.main()
