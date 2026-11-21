import unittest
from types import SimpleNamespace
import numpy as np
import continuous_repairability as policy


class ContinuousBeliefTest(unittest.TestCase):
    def test_limits_preserve_filter_and_paid_history_posterior(self):
        rng = np.random.default_rng(39)
        transition = rng.uniform(size=(2, 8, 8))
        transition /= transition.sum(-1, keepdims=True)
        emission = rng.uniform(size=(2, 8, 31))
        emission /= emission.sum(-1, keepdims=True)
        prior = policy.core.normalize(rng.uniform(size=8))
        posterior = policy.core.normalize(rng.uniform(size=8))
        d = SimpleNamespace(transition=transition, emission=emission)
        model = SimpleNamespace(dynamics={'math': d}, predicted=lambda *a: posterior.copy())
        row, tree = {'subject': 'math'}, {'repair': {'answer': '12'}}
        obs = policy.core.observation(posterior, True)
        expected = policy.core.normalize((prior @ transition[0]) * emission[0, :, obs])
        np.testing.assert_allclose(policy.update(model, row, 'repair', tree, prior, 0, 0.), expected)
        np.testing.assert_allclose(policy.update(model, row, 'repair', tree, prior, 0, 1.), posterior)
        tree['repair']['answer'] = ''
        result = policy.update(model, row, 'repair', tree, prior, 0, .5)
        self.assertEqual(float(result @ policy.core.CURRENT), 0.)
        self.assertAlmostEqual(result.sum(), 1.)


if __name__ == '__main__':
    unittest.main()
