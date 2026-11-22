import unittest
from types import SimpleNamespace

import numpy as np
import ensemble_completion_pomdp as runtime


class EnsembleTests(unittest.TestCase):
    def test_invalid_support_is_conditioned_before_mixing(self):
        a = np.array([.02]*8+[.105]*8); b = np.array([.12]*8+[.005]*8)
        models = [SimpleNamespace(classes_=np.arange(16), predict_proba=lambda x, p=p: np.tile(p, (len(x), 1))) for p in (a, b)]
        for valid in (True, False):
            expected = np.mean([runtime.previous.core.mask_invalid(runtime.previous.core.probabilities(m, [0.]), valid) for m in models], axis=0)
            np.testing.assert_array_equal(runtime.combine(models, [0.], valid), expected)
        np.testing.assert_allclose(runtime.combine(models, [0.], False)[8:], 0.)

    def test_shared_features_only_one_paid_prefix_forward(self):
        seen = []
        def features(row, path, tree):
            seen.append((row, path, tree)); return np.zeros(386)
        forecaster = SimpleNamespace(classes_=np.arange(16), predict_proba=lambda x: np.ones((len(x), 16))/16)
        model = runtime.Planner(SimpleNamespace(features=features), (forecaster, forecaster), {}, {}, 'raw20_plus_project256')
        model.predicted({'subject': 'physics', 'answer': 'SECRET'}, 'repair', {'': {'answer': 'A'}, 'repair': {'answer': ''}, 'retry': {'answer': 'B'}})
        self.assertEqual(len(seen), 1); self.assertEqual(seen[0][0]['answer'], '')
        self.assertEqual(set(seen[0][2]), {'', 'repair'})


if __name__ == '__main__': unittest.main()
