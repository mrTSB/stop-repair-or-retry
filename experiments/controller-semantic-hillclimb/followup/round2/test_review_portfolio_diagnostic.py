import unittest

import numpy as np

import review_portfolio_diagnostic as run


class PortfolioTests(unittest.TestCase):
    def test_held_labels_and_costs_cannot_affect_route_predictions(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(36, 7)); y = rng.integers(0, 2, size=(36, 4))
        cost = rng.uniform(100, 5000, size=y.shape)
        held = np.arange(36)%3 == 0; train = ~held
        changed = y.copy(); changed[held] = 1-changed[held]
        altered_cost = cost.copy(); altered_cost[held] = 1e9
        for recipe in run.RECIPES:
            p, c = run.predict(x, y, cost, train, held, recipe)
            q, d = run.predict(x, changed, altered_cost, train, held, recipe)
            np.testing.assert_array_equal(p, q); np.testing.assert_array_equal(c, d)
            self.assertTrue(np.isfinite(p).all())

    def test_degenerate_training_outcome_and_overlap(self):
        x = np.zeros((12, 3)); y = np.zeros((12, 2), int); y[:, 1] = 1
        held = np.arange(12)%3 == 0; train = ~held; cost = np.ones((12, 2))
        p, c = run.predict(x, y, cost, train, held, 'linear')
        np.testing.assert_allclose(p, np.tile([.1, .9], (4, 1)))
        np.testing.assert_array_equal(c, np.ones((4, 2)))
        with self.assertRaises(ValueError): run.predict(x, y, cost, ~np.zeros(12, bool), held, 'linear')


if __name__ == '__main__': unittest.main()
