import hashlib
import unittest
from unittest.mock import patch

import numpy as np

import factorized_repairability as f


class FactorizedTest(unittest.TestCase):
    def test_ipf_matches_marginals_preserves_interaction_odds(self):
        prior = np.asarray([20., 1., 3., 10., 3., 4., 1., 30.])
        targets = np.asarray([.17, .71, .88])
        result = f.reconstruct(prior, targets)
        np.testing.assert_allclose(result @ f.BITS, targets, atol=1e-8)
        self.assertAlmostEqual(float(result.sum()), 1.)
        self.assertTrue(np.all(result > 0.))
        # C/repair conditional odds ratios for each fixed retry bit.
        for t in (0, 1):
            indexes = [t, 2 + t, 4 + t, 6 + t]
            a, b, c, d = prior[indexes]
            x, y, z, w = result[indexes]
            self.assertAlmostEqual(float(a * d / (b * c)), float(x * w / (y * z)), places=7)

    def test_ipf_stabilizes_boundary_predictions(self):
        p = f.reconstruct(np.ones(8), [0., 1., .5])
        np.testing.assert_allclose(p @ f.BITS, [1e-6, 1. - 1e-6, .5], atol=1e-8)
        with self.assertRaises(ValueError):
            f.reconstruct(np.zeros(8), [.1, .2, .3])

    def test_crossfit_excludes_whole_questions_from_forecaster_and_prior(self):
        qids = np.asarray([f'q{i}' for i in range(30) for _ in range(3)])
        x = np.zeros((len(qids), 6)); x[:, 0] = np.repeat(np.arange(30), 3)
        x[:, -4] = np.tile([0., 1., 1.], 30); x[:, -3] = 1.
        y = np.arange(len(qids)) % 8
        calls = []

        def mock_fit(train_x, train_y, regularization, prior_smoothing):
            training_questions = set(train_x[:, 0])
            class HeldoutOnly:
                def predict_proba(self, held_x):
                    self_questions = set(held_x[:, 0])
                    if training_questions & self_questions:
                        raise AssertionError('Same question entered fitting and observations')
                    calls.append((training_questions, self_questions))
                    return np.full((len(held_x), 8), 1. / 8)
            return HeldoutOnly()

        with patch.object(f, 'fit_forecaster', side_effect=mock_fit):
            p, receipts = f.crossfit(x, y, qids)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(r['question_overlap'] == 0 for r in receipts))
        np.testing.assert_allclose(p.sum(1), 1.)


if __name__ == '__main__':
    unittest.main()
