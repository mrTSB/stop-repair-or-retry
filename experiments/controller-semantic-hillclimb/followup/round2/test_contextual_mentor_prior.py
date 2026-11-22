import copy
import unittest
from unittest.mock import patch

import numpy as np

import contextual_mentor_prior as prior
from test_completion_mentor import fixture


class ContextualPriorTests(unittest.TestCase):
    def test_current_mass_and_within_current_conditional_distribution_preserved(self):
        p = np.arange(1., 25.).reshape(3, 8); mass = np.asarray([0., .37, 1.])
        result = prior.condition(p, mass)
        np.testing.assert_allclose(result.sum(1), 1.)
        np.testing.assert_allclose(result @ prior.CURRENT, mass)
        for current in (0, 1):
            mask = prior.CURRENT == current
            np.testing.assert_allclose(result[1, mask]/result[1, mask].sum(), p[1, mask]/p[1, mask].sum())
        with self.assertRaises(ValueError): prior.condition(p, [0., np.nan, 1.])

    def test_generic_kernel_exactly_matches_original_full256_fit(self):
        rows, complete, strong, judges = fixture()
        records, _, receipt = prior.transfer.join(rows, complete, rows, strong, judges)
        expected, _ = prior.transfer.fit(records, receipt); actual = prior.kernels(records)
        for subject in expected: np.testing.assert_array_equal(expected[subject], actual[subject])

    def test_outer_held_labels_do_not_change_context_or_baseline_predictions(self):
        rng = np.random.default_rng(24); n = 54
        records = [{'id': str(i), 'fold': i % 3, 'subject': 'physics' if i % 2 else 'chemistry'} for i in range(n)]
        labels = np.arange(n) % 8
        transfer = [{**r, 'source': int(prior.CURRENT[y]*8), 'target': int(y)} for r, y in zip(records, labels)]
        data = {'records': records, 'features': {'dense': rng.normal(size=(n, 6))}, 'labels': labels,
                'A_oof': np.ones((n, 16))/16, 'current_mass': np.ones(n)*.5, 'transfer_records': transfer}
        with patch.object(prior, 'CS', (.01,)):
            held, p, baseline, _, receipt = prior.outer_prediction(data, 'dense', 1)
            changed = copy.deepcopy(data); changed['labels'][held] ^= 1
            for r, keep in zip(changed['transfer_records'], held):
                if keep: r['target'] ^= 1
            _, after, old_after, _, second = prior.outer_prediction(changed, 'dense', 1)
        np.testing.assert_array_equal(p, after); np.testing.assert_array_equal(baseline, old_after)
        self.assertEqual(receipt, second)


if __name__ == '__main__': unittest.main()
