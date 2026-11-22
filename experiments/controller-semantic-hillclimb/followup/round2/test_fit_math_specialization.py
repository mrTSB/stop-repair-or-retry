import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import fit_math_specialization as study


class MathFitTests(unittest.TestCase):
    def test_structural_dynamics_preserved_and_impossible_emissions_stay_zero(self):
        rng = np.random.default_rng(238); n = 54
        records = [{'id': str(i//3), 'path': ('', 'repair', 'retry')[i % 3], 'subject': 'math'} for i in range(n)]
        data = {'records': records, 'x': rng.normal(size=(n, 5)), 'y': np.arange(n) % 8,
                'valid': np.ones(n, dtype=bool)}
        dynamics = SimpleNamespace(prior=np.ones(8)/8, transition=np.tile(np.eye(8), (2, 1, 1)),
            emission=np.ones((2, 8, 31))/31, cost=np.ones((2, 2, 8)))
        base = study.core.Planner(object(), object(), {'math': dynamics}, {'state_count': 8})
        oof = np.ones((n, 8))/8
        with patch.object(study.diagnostic, 'select', return_value=({'kind': 'linear', 'C': .01}, [])):
            model = study.fit(data, base, np.ones(n, dtype=bool), oof, 'registered')
        for name in ('prior', 'transition', 'cost'):
            np.testing.assert_array_equal(getattr(model.dynamics['math'], name), getattr(dynamics, name))
        np.testing.assert_allclose(model.dynamics['math'].emission.sum(2), 1.)
        np.testing.assert_array_equal(model.dynamics['math'].emission[:, 4:, -1], 0.)
        self.assertIsNot(model.dynamics['math'], dynamics)
        self.assertTrue(model.fit_receipt['nested_emission_forecasts'])


if __name__ == '__main__': unittest.main()
