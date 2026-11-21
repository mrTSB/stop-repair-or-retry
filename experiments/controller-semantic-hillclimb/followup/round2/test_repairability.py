import unittest
from unittest.mock import patch
import numpy as np

import repairability_pomdp as model


class RepairabilityTest(unittest.TestCase):
    def test_cost_penalty_uses_thousands_of_tokens(self):
        d = model.Dynamics(np.ones(8) / 8, np.zeros((2, 8, 8)),
                           np.zeros((2, 8, model.N_OBS)), np.full((2, 2, 8), 650.))
        belief = np.eye(8)[model.state(0, 1, 0)]
        self.assertAlmostEqual(d.values(belief, 1, .1)[1], .935)

    def test_two_step_backup_matches_explicit_bayesian_enumeration(self):
        rng = np.random.default_rng(23)
        transition = rng.uniform(size=(2, 8, 8))
        transition /= transition.sum(-1, keepdims=True)
        emission = rng.uniform(size=(2, 8, model.N_OBS))
        emission /= emission.sum(-1, keepdims=True)
        d = model.Dynamics(np.ones(8) / 8, transition, emission,
                           rng.uniform(100., 900., size=(2, 2, 8)))
        b = model.normalize(rng.uniform(size=8))
        expected = [b @ model.CURRENT]
        for action in (0, 1):
            predicted = b @ transition[action]
            value = -.24 * (b @ d.cost[0, action]) / 1000
            for obs in range(model.N_OBS):
                unnormalized = predicted * emission[action, :, obs]
                mass = unnormalized.sum()
                value += mass * d.values(unnormalized / mass, 1, .24).max()
            expected.append(value)
        np.testing.assert_allclose(d.values(b, 2, .24), expected, atol=1e-12)

    def test_inputs_exclude_references_and_unpaid_branches(self):
        row = {'id': 'synthetic', 'subject': 'math', 'answer': 'secret-reference'}
        tree = {'': {'text': '3*4=12', 'answer': '12'},
                'repair': {'text': '3*4=12', 'answer': '12'},
                'retry': {'text': 'UNPAID', 'answer': '999'},
                'repair/repair': {'text': 'UNPAID', 'answer': '999'}}
        def visible(safe, path, paid):
            self.assertEqual(safe['answer'], '')
            self.assertEqual(set(paid), {'', 'repair'})
            return [1., 2.]
        neural = (np.zeros(4), np.zeros(8), np.zeros(8), 12)
        with patch.object(model, 'visible', side_effect=visible):
            x = model.observables(row, 'repair', tree, neural)
        self.assertTrue(np.isfinite(x).all())


if __name__ == '__main__':
    unittest.main()
