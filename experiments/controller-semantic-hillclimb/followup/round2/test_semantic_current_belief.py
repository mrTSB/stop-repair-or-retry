import unittest
from unittest.mock import patch

import numpy as np
import diagnose_semantic_current as diagnostic
import semantic_current_belief as correction


class SemanticCorrectionTests(unittest.TestCase):
    def test_historical_summary_projects_only_fit_provenance(self):
        payload = b'{\n  "metrics": invalid_unread_value,\n  "fit_receipt": {"test_used":false,"science_neural_sha256":"h"}\n}'
        self.assertEqual(diagnostic.fitting_provenance(payload)['science_neural_sha256'], 'h')
        with self.assertRaises(ValueError): diagnostic.fitting_provenance(b'{"fit_receipt":{}}')

    def test_current_marginal_changes_without_changing_future_conditionals(self):
        rng = np.random.default_rng(2); p = rng.random((2, 16)); p /= p.sum(1, keepdims=True)
        q = correction.update(p, np.array([.8, .1]), np.array([True, False]))
        self.assertAlmostEqual(q[0] @ correction.CURRENT, .8)
        np.testing.assert_allclose(q[0, :8]/q[0, :8].sum(), p[0, :8]/p[0, :8].sum())
        np.testing.assert_allclose(q[0, 8:]/q[0, 8:].sum(), p[0, 8:]/p[0, 8:].sum())
        np.testing.assert_array_equal(q[1], p[1])

    def test_outer_labels_never_fit_nested_parent_or_calibration(self):
        n = 270; rng = np.random.default_rng(5)
        records = [{'id': str(i//3), 'fold': i//3 % 3, 'subject': 'physics', 'path': ('', 'repair', 'retry')[i % 3]} for i in range(n)]
        data = {'records': records, 'features': {diagnostic.FAMILY: rng.normal(size=(n, 4))},
                'labels': np.arange(n) % 16, 'valid': np.ones(n, dtype=bool)}
        original = np.ones(n, dtype=bool); indices = np.arange(n); logits = rng.normal(size=n); baseline = np.full((n, 16), 1/16)
        with patch.object(diagnostic.forecast, 'TREES', 3), patch.object(diagnostic.forecast, 'SETTINGS', (diagnostic.forecast.SETTINGS[0],)):
            held, before, _, receipt = diagnostic.outer_prediction(data, original, indices, records, logits, baseline, 1)
            changed = {**data, 'labels': data['labels'].copy()}; changed['labels'][held] = 15-changed['labels'][held]
            _, after, _, second = diagnostic.outer_prediction(changed, original, indices, records, logits, baseline, 1)
        np.testing.assert_array_equal(before, after); self.assertEqual(receipt, second)


if __name__ == '__main__': unittest.main()
