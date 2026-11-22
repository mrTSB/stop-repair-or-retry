from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import example_review_runner as run
from test_strong_late_review import fixture
from test_science_mentor_expansion import Remote


class ExampleReviewTests(unittest.TestCase):
    def test_example_reaches_teacher_only_and_compact_prompt_stays_frozen(self):
        rows, trees, *_ = fixture(); row = rows[0]
        packet = 'OTHER_TRAIN_QUESTION: entropy. Correct answer content: TRAIN_ONLY_ANSWER'
        remote = Remote(two_phase=True)
        with TemporaryDirectory() as tmp, run.configured_runner(), \
                patch.object(run, 'context', return_value=packet):
            folder = Path(tmp)
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw, (folder/'attempts').open('a') as attempts:
                journal = run.original.Journal(remote, raw, attempts)
                run.original.Collector(trees, {}, bank, journal).one(row)
            records = run.protocol.strict_records(folder/'bank')
            self.assertIn('TRAIN_ONLY_ANSWER', remote.calls[0]['prompt'])
            self.assertNotIn('TRAIN_ONLY_ANSWER', remote.calls[-1]['prompt'])
            self.assertEqual(remote.calls[2]['prompt'], run.spec.previous.spec.short_repair(row, trees[row['id']][''], records[0]['response']))
            with self.assertRaises(ValueError): run.example_row({**row, 'answer': 'PRIVATE_CURRENT_REFERENCE'})

    def test_full64_physical_replay_with_example_context(self):
        rows, trees, *_ = fixture(); remote = Remote(two_phase=True)
        with TemporaryDirectory() as tmp, run.configured_runner(), \
                patch.object(run, 'context', return_value='A disjoint supervised TRAIN example'):
            folder = Path(tmp)
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw, (folder/'attempts').open('a') as attempts:
                journal = run.original.Journal(remote, raw, attempts)
                collector = run.original.Collector(trees, {}, bank, journal)
                with ThreadPoolExecutor(max_workers=2) as executor: list(executor.map(collector.one, rows))
            records = run.protocol.strict_records(folder/'bank'); raw = run.protocol.strict_records(folder/'raw')
            known, coverage = run.spec.restore(rows, trees, records, raw, True)
            self.assertEqual(len(known), 192); self.assertEqual(coverage['physical_calls'], 256)
            run.original.attempts_checked(run.protocol.strict_records(folder/'attempts'), raw)
            with self.assertRaises(ValueError): run.spec.restore(rows, trees, records[:-1], raw, True)

    def test_scoped_new_protocol_restores_all_registered_globals(self):
        before = (run.spec.SCHEMA, run.spec.PATHS, run.spec.EXTRA_BANKS, run.spec.context,
            run.spec.grounded_row, run.spec.SCREEN, run.original.NAMES, run.original.registration)
        with run.configured_runner():
            self.assertEqual(run.spec.SCHEMA, run.SCHEMA)
            self.assertEqual(run.spec.PATHS, run.PATHS)
            self.assertIs(run.spec.grounded_row, run.example_row)
            self.assertIn('arithmetic1024-root-review-pilot-v1', run.spec.EXTRA_BANKS)
        after = (run.spec.SCHEMA, run.spec.PATHS, run.spec.EXTRA_BANKS, run.spec.context,
            run.spec.grounded_row, run.spec.SCREEN, run.original.NAMES, run.original.registration)
        self.assertEqual(before, after)


if __name__ == '__main__': unittest.main()
