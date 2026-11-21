"""Check two-state beliefs, stochastic kernels, and optional-review Bellman values."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import semantic_data
from binary_core import Dynamics, REWARD, N_OBS
from context_pomdp import REVIEW_WIDTH


def dynamics():
    rng = np.random.default_rng(42)

    def stochastic(shape):
        x = rng.random(shape) + .1
        return x / x.sum(axis=-1, keepdims=True)

    return Dynamics(np.array([.4, .6]), stochastic((2, N_OBS)),
                    stochastic((2, 2, 2, 2)), stochastic((2, 2, 2, N_OBS)),
                    rng.random((2, 2, 2)) * 600,
                    stochastic((2, N_OBS, 2, REVIEW_WIDTH)),
                    stochastic((2, REVIEW_WIDTH, 2, 2, 2)),
                    stochastic((2, 2, 2, N_OBS)), rng.random((2, 2, 2)) * 800,
                    rng.random((2, 2)) * 900, stochastic((N_OBS, 2, 3)))


def one_call(d, mass, condition, penalty, threshold):
    plain = [mass @ REWARD] + [mass @ d.transition[1, a] @ REWARD
                               - penalty * (mass @ d.costs[1, a]) / 1000 for a in (0, 1)]
    reviewed = 0.
    for r in range(REVIEW_WIDTH):
        joint = mass * d.likelihood(1, condition)[:, r]
        values = [joint @ REWARD] + [joint @ d.reviewed_transition[1, r, a] @ REWARD
                                     - penalty * (joint @ d.reviewed_costs[1, a]) / 1000 for a in (0, 1)]
        reviewed += max(values)
    reviewed -= penalty * (mass @ d.review_costs[1]) / 1000
    reviewed -= threshold * mass.sum()
    return max(max(plain), reviewed)


class BinaryPOMDP(unittest.TestCase):
    def test_future_information_backup_matches_independent_enumeration(self):
        d = dynamics()
        belief = np.array([.37, .63])
        penalty = .07
        threshold = .02
        actual = d.values(belief, 2, penalty, future_reviews=True, threshold=threshold)
        expected = [belief @ REWARD]
        for a in (0, 1):
            predicted = belief @ d.transition[0, a]
            total = sum(one_call(d, predicted * d.emission[0, a, :, o], o, penalty, threshold)
                        for o in range(N_OBS))
            expected.append(total - penalty * (belief @ d.costs[0, a]) / 1000)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_reviewed_reuse_backup_matches_joint_observation_enumeration(self):
        d = dynamics()
        belief = np.array([.6, .4])
        transition = d.reviewed_transition[0, 7]
        penalty = .04
        actual = d.values(belief, 2, penalty, transition, True, True, True, .01)
        expected = [belief @ REWARD]
        for a in (0, 1):
            predicted = belief @ transition[a]
            total = sum(one_call(d, predicted * d.reviewed_emission[0, a, :, o]
                                 * d.mentor[o, :, comparison], o, penalty, .01)
                        for o in range(N_OBS) for comparison in range(3))
            expected.append(total - penalty * (belief @ d.reviewed_costs[0, a]) / 1000)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_unnormalized_review_branches_preserve_probability_mass(self):
        d = dynamics()
        b = np.array([.28, .72])
        for future in (False, True):
            np.testing.assert_allclose(d.values(.3 * b, 2, .08, future_reviews=future),
                                       .3 * d.values(b, 2, .08, future_reviews=future), atol=1e-12)
            self.assertAlmostEqual(d.information_value(.3 * b, 2, .08, 2, True, future),
                                   .3 * d.information_value(b, 2, .08, 2, True, future), places=12)


if __name__ == '__main__':
    unittest.main()
