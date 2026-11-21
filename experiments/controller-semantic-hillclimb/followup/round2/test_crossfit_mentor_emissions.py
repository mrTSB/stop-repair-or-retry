import unittest
from unittest.mock import patch

import numpy as np

import crossfit_mentor_emissions as c


class CrossfitMentorTest(unittest.TestCase):
    def test_calibration_excludes_all_rows_of_each_question(self):
        qids = np.asarray([f'q{i}' for i in range(30) for _ in range(4)])
        x = np.repeat(np.arange(30), 4)[:, None]
        y = np.arange(len(qids)) % 4
        train = np.tile([True, True, False, False], 30)
        calls = []

        def mock_fit(fit_x, fit_y):
            fitted = set(fit_x[:, 0])
            class Calibration:
                classes_ = np.arange(4)
                def predict_proba(self, held_x):
                    held = set(held_x[:, 0])
                    if fitted & held:
                        raise AssertionError('A sibling node leaked into the calibration fit')
                    calls.append((fitted, held))
                    return np.full((len(held_x), 4), .25)
            return Calibration()

        with patch.object(c, 'fit_calibrator', side_effect=mock_fit):
            probabilities, receipts = c.calibration_crossfit(x, y, qids, train)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sum(r['held_rows'] for r in receipts), len(qids))
        self.assertTrue(all(r['question_overlap'] == 0 for r in receipts))
        np.testing.assert_allclose(probabilities.sum(1), 1.)

    def test_emissions_normalize_and_preserve_parent_soft_support(self):
        support = c.observation_support()
        self.assertEqual(support[2, 0], 1e-4)  # Correct answer cannot be invalid.
        self.assertEqual(support[2, 5], 1e-4)  # Correct current, wrong mentor cannot agree.
        self.assertEqual(support[3, 4], 1e-4)  # Both correct cannot disagree.
        self.assertEqual(support[3, 5], 1.)
        baseline = c.rebuild_emissions([], ['chemistry'], 8.)['chemistry']
        np.testing.assert_allclose(baseline.sum(-1), 1.)
        self.assertAlmostEqual(baseline[0, 3, 4] / baseline[0, 3, 5], 1e-4)
        records = [('chemistry', 0, 3, 5), ('physics', 1, 2, 4)]
        rebuilt = c.rebuild_emissions(records, ['chemistry', 'physics'], 8.)
        self.assertTrue(all(np.isfinite(v).all() and (v > 0.).all() for v in rebuilt.values()))
        for value in rebuilt.values():
            np.testing.assert_allclose(value.sum(-1), 1.)


if __name__ == '__main__':
    unittest.main()
