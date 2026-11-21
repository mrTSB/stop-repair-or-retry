import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import scoring_completion_availability as run
from test_scoring_observation import row, plain, response


def canonical(i=0, subject='chemistry'):
    safe = row(i, subject)
    safe['question'] = f'Public synthetic question {i}?\n\nOptions:\nA. One\nB. Two\nC. Three\nD. Four'
    return safe


class CompletionAvailabilityTests(unittest.TestCase):
    def test_exact_returned_letter_only_no_fallback_or_case_conversion(self):
        letters = run.offered_letters(canonical())
        self.assertEqual(letters, list('ABCD'))
        for text in ('A', ' B ', '\nC\t'):
            self.assertEqual(run.hypothetical_answer(text, letters), text.strip())
        for text in ('a', 'Final: A', 'A.', 'J', '', 'AB', '<think>A'):
            self.assertEqual(run.hypothetical_answer(text, letters), '')
        with self.assertRaises(ValueError): run.offered_letters({**canonical(), 'answer': 'PRIVATE_REFERENCE'})
        with self.assertRaises(ValueError): run.offered_letters({**canonical(), 'question': canonical()['question']+'\nB. nested label'})

    def test_secondary_registration_must_precede_any_scoring_call(self):
        with TemporaryDirectory() as directory:
            output = Path(directory); args = SimpleNamespace(output=output)
            (output/'raw-generations.jsonl').touch()
            with patch.object(run, 'specification', side_effect=AssertionError('Do not register late')):
                with self.assertRaisesRegex(ValueError, 'before the first'): run.register(args)
            (output/'raw-generations.jsonl').unlink()
            with patch.object(run, 'specification', return_value={'schema': run.SCHEMA, 'extra_calls': 0}):
                run.register(args)
            self.assertTrue((output/'secondary-availability/registration.json').is_file())
            with patch.object(run, 'specification', return_value={'schema': run.SCHEMA, 'extra_calls': 0}):
                with self.assertRaises(FileExistsError): run.register(args)

    def test_no_training_label_read_before_full_primary_seal(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            with patch.object(run.pilot.fixed, 'lines', side_effect=AssertionError('Labels opened too early')):
                with self.assertRaises(FileNotFoundError): run.grade(args)

    def test_all64_union_costs_and_invalid_responses_without_selection(self):
        rows = [canonical(i, subject) for subject in ('chemistry', 'physics') for i in range(32)]
        trees = {r['id']: plain(r) for r in rows}; known = {}; raw = []; references = {}
        for i, safe in enumerate(rows):
            qid = safe['id']; references[qid] = 'A' if i < 32 else 'B'
            for path in run.pilot.PATHS:
                payload, _ = run.pilot.request_payload(safe, path, trees[qid])
                selected = ' A' if i < 8 else ('Final: A' if i < 16 else 'B')
                result = response(selected)
                receipt = {'endpoint': '/api/generate', 'payload': payload,
                    'request_sha256': run.pilot.digest({'endpoint': '/api/generate', 'payload': payload}),
                    'response': result, 'response_sha256': run.pilot.digest(result),
                    'started_unix': 1., 'returned_unix': 1.25}
                raw.append(receipt)
                known[qid, path] = run.pilot.observation(safe, path, trees[qid], receipt)
        panels, outcomes = run.aggregate(rows, trees, references, known, raw)
        self.assertEqual(len(outcomes), 192)
        all_rows = panels['all_science']
        self.assertEqual(all_rows['questions'], 64)
        self.assertEqual(all_rows['original_seven_path_oracle_correct'], 32)
        self.assertEqual(all_rows['combined_ten_path_oracle_correct'], 40)
        self.assertEqual(all_rows['novel_correct_beyond_original_seven'], 8)
        self.assertEqual(all_rows['paths']['root']['valid'], 56)
        self.assertEqual(all_rows['paths']['root']['correct'], 40)
        self.assertAlmostEqual(all_rows['paths']['root']['mean_full_hypothetical_trajectory_tokens'], 124.)
        self.assertAlmostEqual(all_rows['paths']['repair']['mean_full_hypothetical_trajectory_tokens'], 147.)
        self.assertEqual({v['hypothetical_total_continuations'] for v in outcomes if v['path']}, {2})
        self.assertTrue(all(trees[r['id']]['']['answer'] == 'B' for r in rows))
        partial = dict(known); del partial[rows[0]['id'], '']
        with self.assertRaises(ValueError): run.aggregate(rows, trees, references, partial, raw)
        changed = copy.deepcopy(raw); changed[0]['response_sha256'] = 'changed'
        with self.assertRaises(ValueError): run.aggregate(rows, trees, references, known, changed)


if __name__ == '__main__':
    unittest.main()
