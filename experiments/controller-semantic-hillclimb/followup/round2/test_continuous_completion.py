import unittest
from types import SimpleNamespace

import numpy as np
import continuous_completion_pomdp as policy


class ContinuousCompletionTests(unittest.TestCase):
    def test_endpoints_and_invalid_support(self):
        rng = np.random.default_rng(17)
        d = SimpleNamespace(transition=rng.uniform(size=(2, 16, 16)), emission=rng.uniform(size=(2, 16, 30)))
        d.transition /= d.transition.sum(2, keepdims=True)
        d.emission /= d.emission.sum(2, keepdims=True)
        model = SimpleNamespace(dynamics={'physics': d}, kernel=SimpleNamespace(signal=lambda p, v: 4 if v else 26))
        prior, posterior = rng.uniform(size=(2, 16)); prior /= prior.sum(); posterior /= posterior.sum()
        for valid in (False, True):
            expected = policy.core.mask_invalid((prior @ d.transition[1])*d.emission[1, :, 4 if valid else 26], valid)
            np.testing.assert_allclose(policy.update(model, 'physics', prior, 1, posterior, valid, 0.), expected)
            np.testing.assert_allclose(policy.update(model, 'physics', prior, 1, posterior, valid, 1.), policy.core.mask_invalid(posterior, valid))
            blended = policy.update(model, 'physics', prior, 1, posterior, valid, .5)
            np.testing.assert_allclose(blended, .5*expected+.5*policy.core.mask_invalid(posterior, valid))
            self.assertAlmostEqual(float(blended.sum()), 1.)
            if not valid: np.testing.assert_array_equal(blended[8:], 0.)
        with self.assertRaises(ValueError): policy.update(model, 'physics', prior, 1, posterior, True, .7)


if __name__ == '__main__': unittest.main()
