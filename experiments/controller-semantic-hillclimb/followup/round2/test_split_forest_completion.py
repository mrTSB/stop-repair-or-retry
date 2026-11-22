import unittest
from unittest.mock import patch

import numpy as np
import split_forest_completion_forecast as forecast


class SplitForestCompletionTests(unittest.TestCase):
    def test_joint_distribution_includes_unseen_states(self):
        x = np.arange(80).reshape(40, 2); y = np.arange(40) % 3
        with patch.object(forecast, 'TREES', 3):
            model = forecast.fit(x, y, forecast.SETTINGS[0])
        p = model.predict_proba(x[:3])
        self.assertEqual(p.shape, (3, 16)); self.assertTrue((p > 0).all())
        np.testing.assert_allclose(p.sum(1), 1.)
        masked = forecast.previous.predict(model, x[:3], np.zeros(3, dtype=bool))
        np.testing.assert_allclose(masked[:, 8:], 0., atol=1e-8)

    def test_outer_held_labels_do_not_change_fit_or_selection(self):
        rng = np.random.default_rng(4); n = 54
        records = [{'id': str(i//3), 'fold': i//3 % 3} for i in range(n)]
        data = {'records': records, 'features': {'raw20_control': rng.normal(size=(n, 4))},
                'labels': np.arange(n) % 16, 'valid': np.ones(n, dtype=bool)}
        with patch.object(forecast, 'TREES', 2), patch.object(forecast, 'SETTINGS', (forecast.SETTINGS[0],)):
            held, p, _, receipt = forecast.outer_prediction(data, 'raw20_control', 1)
            changed = {**data, 'labels': data['labels'].copy()}; changed['labels'][held] = 15-changed['labels'][held]
            _, after, _, second = forecast.outer_prediction(changed, 'raw20_control', 1)
        np.testing.assert_array_equal(p, after); self.assertEqual(receipt, second)


if __name__ == '__main__': unittest.main()
