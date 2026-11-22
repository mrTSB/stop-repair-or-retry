from types import SimpleNamespace
import unittest

import numpy as np

import difficulty_completion_pomdp as model


class DifficultyTests(unittest.TestCase):
    def test_hidden_difficulty_is_stable_and_transition_support_is_valid(self):
        rows = [{'id': str(i), 'subject': 'chemistry' if i%2 else 'physics', 'answer': 'A'} for i in range(12)]
        trees = {r['id']: {p: {'answer': 'A' if int(r['id'])%3 == 0 else 'B', 'input_tokens': 10, 'output_tokens': 2}
            for p in model.PATHS} for r in rows}
        states = {r['id']: model.labels(r, trees[r['id']]) for r in rows}
        records = [{'id': r['id'], 'path': p} for r in rows for p in ('', 'repair', 'retry')]
        p = np.asarray([np.eye(32)[states[r['id']][r['path']]] for r in records])
        observer = SimpleNamespace(_recipe={'parameters': 400000000, 'max_length': 512})
        fitted = model.fit_dynamics(rows, trees, observer, records, p, None, 'raw20_control', {})
        for r in rows:
            self.assertEqual(len({s//16 for s in states[r['id']].values()}), 1)
        different = model.DIFFICULT[:, None] != model.DIFFICULT[None, :]
        for dynamics in fitted.dynamics.values():
            np.testing.assert_allclose(dynamics.transition.sum(2), 1.)
            np.testing.assert_array_equal(dynamics.transition[:, different], 0.)
            for a in (0, 1):
                mismatch = model.AFTER[a, :, None] != model.CURRENT[None, :]
                np.testing.assert_array_equal(dynamics.transition[a][mismatch], 0.)

    def test_two_step_bellman_matches_explicit_observation_enumeration(self):
        rng = np.random.default_rng(238); p = rng.dirichlet(np.ones(32))
        transition = rng.dirichlet(np.ones(32), size=(2, 32))
        emission = rng.dirichlet(np.ones(5), size=(2, 32)); cost = rng.uniform(10, 500, size=(2, 3, 32))
        d = model.Dynamics(p, transition, emission, cost); penalty = .02
        alpha = np.vstack([model.CURRENT, *[model.AFTER[a]-penalty*cost[1, a]/1000 for a in range(3)]])
        expected = [p@model.CURRENT]
        for a in range(3):
            future = p@model.AFTER[a] if a == 2 else sum(max(alpha@(p@transition[a]*emission[a, :, o])) for o in range(5))
            expected.append(future-penalty*(p@cost[0, a])/1000)
        np.testing.assert_allclose(d.values(p, 2, penalty), expected)
        np.testing.assert_allclose(d.values(p, 1, penalty), [p@model.CURRENT, *[p@(model.AFTER[a]-penalty*cost[1, a]/1000) for a in range(3)]])
        with self.assertRaises(ValueError): d.values(p, 3, penalty)

    def test_predicted_observation_cannot_see_unpaid_future_or_reference(self):
        class Observer:
            def features(self, row, path, paid):
                self.seen = row, path, paid
                return np.zeros(3)
        observer = Observer(); forecast = model.fit_forecaster(np.zeros((100, 3)), np.zeros(100, int))
        planner = model.Planner(observer, forecast, {}, 'raw20_plus_project256', {})
        row = {'id': 'q', 'answer': '', 'subject': 'physics'}
        tree = {'': {'answer': ''}, 'repair': {'answer': 'A'}, 'repair/repair': {'answer': 'SECRET'}}
        p = planner.predicted(row, 'repair', tree)
        self.assertEqual(set(observer.seen[2]), {'', 'repair'}); self.assertEqual(observer.seen[0]['answer'], '')
        self.assertAlmostEqual(p.sum(), 1.)
        with self.assertRaises(ValueError): planner.predicted({**row, 'answer': 'A'}, 'repair', tree)
        invalid = model.mask(np.ones(32), False); np.testing.assert_array_equal(invalid[model.CURRENT == 1], 0.)
        self.assertTrue(24 <= model.signal(invalid, False) < 30)


if __name__ == '__main__': unittest.main()
