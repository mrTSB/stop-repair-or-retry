"""Conditional forecasts enumerate hidden states instead of reading runtime labels."""

import unittest
from unittest.mock import patch

import numpy as np

from tree_context_pomdp import Observer


class Forecast:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        x = np.asarray(x)
        np.testing.assert_equal(x[:, -4:], np.eye(4))
        p = np.array([.1, .8, .3, .9])
        return np.column_stack((1 - p, p))


class PairedForecasts(unittest.TestCase):
    def test_each_possible_state_is_forecast_without_a_correctness_reference(self):
        observer = Observer('unused', 1.)
        observer.values = lambda *a: (np.array([1., 2., 3., 4.]), None, None, 12)
        observer._forecasters = {(reviewed, action): Forecast() for reviewed in (False, True) for action in (0, 1)}
        with patch('tree_context_pomdp.base.visible', return_value=[.2, .4]):
            for reviewed in (False, True):
                actual = observer.transitions({'answer': 'SECRET'}, '', {}, reviewed)
                other = observer.transitions({'answer': 'DIFFERENT'}, '', {}, reviewed)
                np.testing.assert_equal(actual, other)
                np.testing.assert_allclose(actual, [[.1, .8, .3, .9], [.1, .8, .3, .9]])


if __name__ == '__main__':
    unittest.main()
