"""Independent enumerations of joint current/mentor finite-horizon values."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mentor_pomdp import JointDynamics, N_REVIEW, N_OBS, state


def fixture():
    rng = np.random.default_rng(2026112073)
    def norm(a):
        return a / a.sum(-1, keepdims=True)
    support = np.array([[float(s // 4 == t // 4 and s % 2 == t % 2) for t in range(8)] for s in range(8)])
    reviewed = norm(rng.uniform(.1, 1., (N_REVIEW, 2, 8, 8)) * support)
    last = norm(rng.uniform(.1, 1., (2, 2, 8, 8)) * support)
    emission = norm(rng.uniform(.1, 1., (2, 8, N_OBS * 3)))
    prior = norm(rng.uniform(.1, 1., 8))
    review = norm(rng.uniform(.1, 1., (8, N_REVIEW)))
    return JointDynamics(prior, norm(rng.uniform(.1, 1., (8, N_OBS))),
                          norm(rng.uniform(.1, 1., (2, 2, 8, 8)) * support),
                          review, reviewed, emission, last,
                          rng.uniform(200, 800, (2, 8)), rng.uniform(200, 800, (2, 2, 8)),
                          rng.uniform(1800, 3200, 8), emission.reshape(2, 8, N_OBS, 3).sum(2))


def enumerate_values(d, belief, signal, penalty):
    reward = np.array([float((s // 2) % 2) for s in range(8)])
    values = [float(belief @ reward)]
    for first in range(2):
        predicted = belief @ d.reviewed_transition[signal, first]
        continuation = 0.
        for observation in range(N_OBS * 3):
            mass = predicted * d.emission[first, :, observation]
            choices = [float(mass @ reward)]
            for last in range(2):
                success = sum(mass[s] * sum(d.last_transition[first, last, s, t] * reward[t] for t in range(8))
                              for s in range(8))
                price = sum(mass[s] * d.last_costs[first, last, s] for s in range(8))
                choices.append(success - penalty * price / 1000)
            continuation += max(choices)
        values.append(continuation - penalty * sum(belief[s] * d.first_costs[first, s] for s in range(8)) / 1000)
    return np.array(values)


class MentorTests(unittest.TestCase):
    def test_exact_root_backup_includes_the_retained_review(self):
        d = fixture()
        for signal in (0, 11, 30, 31):
            np.testing.assert_allclose(d.values(d.prior, signal, .07), enumerate_values(d, d.prior, signal, .07),
                                       rtol=1e-12, atol=1e-12)

    def test_expected_information_uses_each_paid_signal_probability_once(self):
        d = fixture()
        penalty = .04
        expected = sum(enumerate_values(d, d.prior * d.review[:, r], r, penalty).max()
                       for r in range(N_REVIEW))
        expected -= penalty * (d.prior @ d.review_costs) / 1000
        self.assertAlmostEqual(d.information_value(d.prior, penalty), expected, places=12)
        np.testing.assert_allclose(d.values(.3*d.prior, 4, penalty), .3*d.values(d.prior, 4, penalty),
                                   rtol=1e-12, atol=1e-12)

    def test_mentor_truth_and_question_difficulty_persist(self):
        d = fixture()
        for h in (0, 1):
            for c in (0, 1):
                for m in (0, 1):
                    self.assertEqual(state(2*h+c, m), 4*h+2*c+m)
        for matrix in [*d.reviewed_transition.reshape(-1, 8, 8), *d.last_transition.reshape(-1, 8, 8)]:
            predicted = d.prior @ matrix
            self.assertAlmostEqual(predicted[1::2].sum(), d.prior[1::2].sum())
            self.assertAlmostEqual(predicted[4:].sum(), d.prior[4:].sum())


if __name__ == '__main__':
    unittest.main()
