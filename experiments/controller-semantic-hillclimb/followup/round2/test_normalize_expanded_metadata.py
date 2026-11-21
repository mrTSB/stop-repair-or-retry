import copy
import unittest

import numpy as np

import normalize_expanded_metadata as run
import expanded_review_metadata as metadata
import test_expanded_review_metadata as fixtures


class NormalizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = fixtures.fixture()
        for r in records:
            for i in (15, 19, 21): r['features'][i] = 1.
        fitted = metadata.train_model(records)
        cls.x = np.asarray([r['features'] for r in records])
        cls.adapter = run.paid.Adapter(fitted, 1.5, {})
        cls.adapter.fitted['model']['scale'].var_[[15, 19, 21]] = [-1e-45, -2e-45, -3e-45]
        cls.adapter.receipt = {'learned_state_sha256': run.paid.digest(cls.adapter.state())}

    def test_only_constant_tiny_variances_change_all_predictions_identical(self):
        before = copy.deepcopy(self.adapter.state())
        corrected, proof = run.normalize_adapter(self.adapter, self.x)
        self.assertEqual(proof['changed_indexes'], [15, 19, 21])
        self.assertEqual(self.adapter.state(), before)
        self.assertTrue(proof['transformed_features_identical'])
        self.assertTrue(proof['raw_predictions_identical'])
        self.assertTrue(proof['calibrated_predictions_identical'])
        self.assertEqual(corrected.state()['variance'][15], 0.)
        run.paid_audit.validate_adapter_state(corrected.state(), corrected.receipt)
        with self.assertRaises(ValueError): run.paid_audit.validate_adapter_state(before, self.adapter.receipt)

    def test_materially_negative_or_nonconstant_or_nonunit_scale_rejected(self):
        for kind in ('material', 'scale', 'variable'):
            with self.subTest(kind=kind):
                adapter = copy.deepcopy(self.adapter); x = self.x.copy()
                if kind == 'material': adapter.fitted['model']['scale'].var_[15] = -1e-8
                if kind == 'scale': adapter.fitted['model']['scale'].scale_[15] = 2.
                if kind == 'variable': x[0, 15] += .1
                adapter.receipt['learned_state_sha256'] = run.paid.digest(adapter.state())
                with self.assertRaisesRegex(ValueError, 'bounded roundoff'):
                    run.normalize_adapter(adapter, x)

    def test_positive_statistics_unchanged_and_normalization_is_idempotent(self):
        corrected, _ = run.normalize_adapter(self.adapter, self.x)
        twice, proof = run.normalize_adapter(corrected, self.x)
        self.assertEqual(proof['changed_indexes'], [])
        self.assertEqual(twice.state(), corrected.state())
        for i in set(range(32)) - {15, 19, 21}:
            self.assertEqual(corrected.state()['variance'][i], self.adapter.state()['variance'][i])


if __name__ == '__main__': unittest.main()
