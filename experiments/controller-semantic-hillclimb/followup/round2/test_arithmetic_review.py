from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import arithmetic_review_runner as run
from test_strong_late_review import fixture
from test_science_mentor_expansion import Remote


class ArithmeticRemote(Remote):
    def __init__(self): super().__init__(two_phase=True)
    def __call__(self, endpoint, payload):
        result = super().__call__(endpoint, payload)
        if payload['options']['num_predict'] == 1024:
            result['response'] = 'Check numerical ratio <calc>1/3</calc> then <calc>sqrt(-1)</calc>'
            result['thinking'] = ''
        return result


class ArithmeticReviewTests(unittest.TestCase):
    def collect(self, folder, rows, trees):
        remote = ArithmeticRemote()
        with run.configured_runner(), (folder/'bank').open('a') as bank, \
                (folder/'raw').open('a') as raw, (folder/'attempts').open('a') as attempts:
            journal = run.original.Journal(remote, raw, attempts)
            collector = run.original.Collector(trees, {}, bank, journal)
            with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(collector.one, rows))
        return remote, run.protocol.strict_records(folder/'bank'), run.protocol.strict_records(folder/'raw')

    def test_paid_tool_results_reach_decision_and_compact_prompt_stays_frozen(self):
        rows, trees, *_ = fixture(); row = rows[0]; tree = trees[row['id']]
        tree['']['text'] = 'PAID_ROOT Final: B'; tree['repair']['text'] = 'UNPAID_BRANCH'
        with TemporaryDirectory() as tmp:
            remote, records, raw = self.collect(Path(tmp), [row], trees)
        self.assertEqual(len(remote.calls), 4)
        first, second, guided, reuse = remote.calls
        self.assertIn('PAID_ROOT', first['prompt']); self.assertIn('0.333333333333', second['prompt'])
        self.assertIn('Invalid, unsupported', second['prompt'])
        self.assertEqual(guided['prompt'], run.spec.previous.spec.short_repair(row, tree[''], records[0]['response']))
        self.assertFalse(any('UNPAID_BRANCH' in p['prompt'] for p in remote.calls))
        self.assertEqual([p['options']['num_predict'] for p in remote.calls], [1024, 512, 128, 128])
        self.assertEqual([p['options']['seed'] for p in (guided, reuse)],
            [run.protocol.seed(row['id'], p) for p in ('repair', 'repair/repair')])
        teacher = records[0]['response']
        self.assertEqual(teacher['physical_api_calls'], 2)
        self.assertEqual(teacher['calculator_expression_count'], 2)
        self.assertEqual(teacher['input_tokens'], 40); self.assertEqual(teacher['output_tokens'], 16)
        with self.assertRaises(ValueError): run.spec.reasoning_prompt({**row, 'answer': 'SECRET_REFERENCE'}, tree[''])

    def test_full_grid_durable_attempts_and_tamper_detection(self):
        rows, trees, *_ = fixture()
        with TemporaryDirectory() as tmp:
            folder = Path(tmp); remote, records, raw = self.collect(folder, rows, trees)
            known, coverage = run.spec.restore(rows, trees, records, raw, True)
            self.assertEqual(len(known), 192); self.assertEqual(coverage['physical_calls'], 256)
            self.assertEqual(len(remote.calls), 256)
            attempts = run.protocol.strict_records(folder/'attempts')
            with run.configured_runner():
                run.original.attempts_checked(attempts, raw)
                with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as stream, (folder/'attempts').open('a') as a:
                    journal = run.original.Journal(lambda *args: self.fail('Completed response regenerated'), stream, a, raw, attempts)
                    run.original.Collector(trees, known, bank, journal).one(rows[0])
                with self.assertRaises(ValueError): run.original.attempts_checked(attempts, raw[:-1])
            altered = copy.deepcopy(records); altered[0]['response']['calculator_results'][0]['value'] = '1'
            with self.assertRaises(ValueError): run.spec.restore(rows, trees, altered, raw)
            with self.assertRaises(ValueError): run.spec.restore(rows, trees, records[:-1], raw, True)

    def test_runner_scope_restores_registered_predecessor(self):
        before = run.original.spec, run.original.NAMES, run.original.registration, run.original.require_priority
        with run.configured_runner():
            self.assertIs(run.original.spec, run.spec)
            self.assertIs(run.original.registration, run.registration)
        self.assertEqual(before, (run.original.spec, run.original.NAMES, run.original.registration, run.original.require_priority))

    def test_registration_declares_new_tool_and_matched_b(self):
        record = {'teacher': {}, 'retrieval_neural_calls': 0, 'retrieval_CPU_not_in_parameter_token_proxy': True,
            'questions_with_textbook_context': 0, 'textbook_tokens_charged_in_teacher_input': True}
        with patch.object(run, 'require_priority'), patch.object(run, 'base_registration', return_value=record):
            result = run.registration(None)
        self.assertTrue(result['calculator']['matched_B_must_receive_identical_tool_availability'])
        self.assertTrue(result['teacher']['always_two_phases'])
        self.assertEqual(result['minimum_physical_teacher_calls'], 128)
        self.assertNotIn('retrieval_neural_calls', result)


if __name__ == '__main__': unittest.main()
