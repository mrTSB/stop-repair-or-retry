import copy
import unittest
from unittest.mock import patch

import numpy as np

import second_review_value_diagnostic as diagnostic


def records():
    return [{'id': f'synthetic-{i}', 'fold': i % 3, 'subject': 'chemistry' if i % 2 else 'physics',
        'features': [float(i % 5), float(i % 4), float(i) / 30.],
        'state': i % 8, 'labels': [(i % 8 >> bit) & 1 for bit in range(3)],
        'future_compute': [0., 500. + i, 4000. + i * 10.], 'prefix_compute': 3000.} for i in range(30)]


class SecondReviewValueTests(unittest.TestCase):
    def test_held_labels_and_future_costs_do_not_enter_held_predictions(self):
        values = records(); p, c, receipt = diagnostic.predict_excluded(values)
        changed = copy.deepcopy(values)
        for row in changed:
            if row['fold'] == 1:
                row['state'] = (row['state'] + 3) % 8
                row['future_compute'] = [0., 1e10, 2e10]
        p2, c2, _ = diagnostic.predict_excluded(changed)
        mask = np.asarray([r['fold'] == 1 for r in values])
        np.testing.assert_array_equal(p[mask], p2[mask])
        np.testing.assert_array_equal(c[mask], c2[mask])
        for fold in receipt:
            self.assertFalse(set(fold['training_ids']) & set(fold['excluded_ids']))

    def test_single_class_training_folds_and_costs_remain_finite(self):
        values = records()
        for row in values:
            row['state'] = 5
        probabilities, costs, _ = diagnostic.predict_excluded(values)
        np.testing.assert_array_equal(probabilities[:, 5], np.ones(len(values)))
        self.assertTrue(np.isfinite(costs).all())
        np.testing.assert_array_equal(costs[:, 0], np.zeros(len(values)))

    def test_one_step_belief_action_and_ties_use_predicted_incremental_cost(self):
        probabilities = np.zeros((2, 8)); probabilities[0, 6] = 1.; probabilities[1, 0] = 1.
        costs = np.asarray([[0., 500., 5000.], [0., 500., 5000.]])
        self.assertEqual(diagnostic.decisions(probabilities, costs, 0.).tolist(), [1, 0])
        self.assertEqual(diagnostic.decisions(probabilities, costs, 3.).tolist(), [0, 0])

    def test_actual_accounting_includes_prefix_and_both_physical_review_phases(self):
        row = records()[0]; row['labels'] = [1, 0, 1]
        value = diagnostic.aggregate([row, row], [0, 2])
        self.assertEqual(value['correct'], 2)
        self.assertEqual(value['mean_compute'], 5000.)
        self.assertEqual(diagnostic.cost({'input_tokens': 100, 'output_tokens': 30}, 3.), 130. * 3. / 3.09)

    def test_paid_feature_api_rejects_reference_and_never_accepts_future_nodes(self):
        root = {'answer': 'A', 'text': 'paid root'}
        review = {'answer': 'B', 'action': 'repair', 'parse_valid': True, 'physical_api_calls': 1,
            'input_tokens': 12, 'output_tokens': 6, 'verdict': {'solution': 'paid thought'},
            'calls': [{'input_tokens': 12, 'output_tokens': 6, 'output_cap': 20, 'finish': 'stop'}]}
        guided = {'answer': 'B', 'text': 'paid guided', 'input_tokens': 30, 'output_tokens': 10, 'output_cap': 128}
        with self.assertRaisesRegex(ValueError, 'Blank'):
            diagnostic.public_features({'subject': 'chemistry', 'answer': 'A'}, root, review, guided)
        x = diagnostic.public_features({'subject': 'chemistry', 'answer': ''}, root, review, guided)
        self.assertEqual(len(x), len(diagnostic.FEATURES))
        self.assertTrue(np.isfinite(x).all())
        guided['unpaid_second_review'] = 'should not enter features'
        guided['reference_answer'] = 'should not enter features'
        np.testing.assert_array_equal(x, diagnostic.public_features({'subject': 'chemistry', 'answer': ''}, root, review, guided))

    def test_missing_question_fold_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'fold'):
            diagnostic.predict_excluded([r for r in records() if r['fold'] != 2])

    def test_bit_encoding_matches_declared_action_order(self):
        for state in range(8):
            self.assertEqual(sum(int(v) << bit for bit, v in enumerate(diagnostic.REWARDS[state])), state)

    def test_physical_phase_duplicates_omissions_and_mutation_fail(self):
        calls = [{'phase': phase, 'text': 'synthetic'} for phase in (diagnostic.pilot.THOUGHT_PHASE, diagnostic.pilot.ANSWER_PHASE)]
        rows = [{'id': 'q'}]; independent = {('q', 'judge'): {'calls': calls}}
        phases = [{'id': 'q', 'phase': c['phase'], 'response': copy.deepcopy(c)} for c in calls]
        diagnostic.verify_physical(rows, independent, phases)
        for bad in (phases[:1], phases + phases[:1]):
            with self.assertRaisesRegex(ValueError, 'physical'):
                diagnostic.verify_physical(rows, independent, bad)
        phases[0]['response']['text'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'differs'):
            diagnostic.verify_physical(rows, independent, phases)

    def test_dependency_closure_binds_executed_control_and_scoring_sources(self):
        from pathlib import Path
        names = {Path(p).name for p in diagnostic.source_files()}
        self.assertTrue({'second_review_value_diagnostic.py', 'independent_review_pilot.py',
            'second_review_pilot.py', 'mentor_pairs.py', 'paid_review_metadata.py',
            'review_metadata_diagnostic.py', 'scoring.py', 'banks.py'} <= names)


if __name__ == '__main__':
    unittest.main()
