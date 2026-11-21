from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import nonlinear_repairability as nonlinear
import development_nonlinear as development


class NonlinearTests(unittest.TestCase):
    def test_outer_question_never_enters_its_model_or_temperature_selection(self):
        qids = np.asarray([f'q{i}' for i in range(60) for _ in range(3)])
        x = np.column_stack([np.repeat(np.arange(60), 3), np.tile([0., 1., 2.], 60)])
        y = np.arange(len(qids)) % 8
        predictions = []
        def fitting(tx, ty, leaves):
            train = set(tx[:, 0])
            class Fake:
                classes_ = np.arange(8)
                def predict_proba(self, held):
                    if train & set(held[:, 0]):
                        raise AssertionError('Own question entered crossfit prediction')
                    predictions.append((train, set(held[:, 0])))
                    result = np.ones((len(held), 8)); result[:, 0] = 2.
                    return result / result.sum(1, keepdims=True)
            return Fake()
        with patch.object(nonlinear, 'raw_fit', side_effect=fitting):
            model, oof, receipt = nonlinear.crossfit(x, y, qids, 7)
        self.assertEqual(len(predictions), 12)
        self.assertEqual(len(receipt['outer_folds']), 3)
        self.assertTrue(receipt['emission_temperature_excludes_own_question'])
        self.assertTrue(all(r['outer_held_overlap'] == 0 for f in receipt['outer_folds'] for r in f['inner_folds']))
        np.testing.assert_allclose(oof.sum(1), 1.)

    def test_histogram_family_is_deterministic_and_does_not_use_early_stopping(self):
        random = np.random.default_rng(11)
        x = random.normal(size=(128, 5)); y = (x[:, 0] > 0).astype(int) + 2 * (x[:, 1] > 0)
        first = nonlinear.raw_fit(x, y, 7); second = nonlinear.raw_fit(x, y, 7)
        self.assertFalse(first.early_stopping)
        self.assertEqual(first.max_iter, 100)
        self.assertEqual(first.min_samples_leaf, 30)
        self.assertEqual(first.l2_regularization, 5.)
        np.testing.assert_array_equal(nonlinear.aligned_probability(first, x), nonlinear.aligned_probability(second, x))
        self.assertEqual(nonlinear.aligned_probability(first, x).shape, (128, 8))
        with self.assertRaises(ValueError):
            nonlinear.raw_fit(x, y, 31)

    def test_replacement_preserves_transition_support_costs_and_state_semantics(self):
        rows = [{'id': f'q{i}', 'answer': '1', 'subject': ('math', 'chemistry', 'physics')[i % 3],
                 'domain': 'math', 'question': 'Synthetic question'} for i in range(48)]
        random = np.random.default_rng(41)
        trees, lookup = {}, {}
        paths = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
        for row in rows:
            trees[row['id']] = {path: {'answer': str(random.integers(2)), 'text': 'Final: 1',
                'confidence': None, 'input_tokens': 10, 'output_tokens': 8, 'finish': 'stop'} for path in paths}
            for path in ('', 'repair', 'retry'):
                lookup[row['id'], path] = (random.normal(size=4), random.normal(size=8), random.normal(size=8), 0)
        observer = SimpleNamespace(_recipe={'parameters': 400_000_000, 'max_length': 512})
        with nonlinear.threadpool_limits(limits=1):
            original = nonlinear.core.fit(rows, trees, observer, lookup, smoothing=8.)
        class FakeForecast:
            classes_ = np.arange(8)
            def predict_proba(self, x):
                return np.full((len(x), 8), .125)
        with patch.object(nonlinear, 'crossfit', return_value=(FakeForecast(), np.full((144, 8), .125), {})):
            changed = nonlinear.fit(rows, trees, observer, lookup, smoothing=8., max_leaf_nodes=7)
        for subject, dynamics in changed.dynamics.items():
            np.testing.assert_array_equal(dynamics.transition, original.dynamics[subject].transition)
            np.testing.assert_array_equal(dynamics.cost, original.dynamics[subject].cost)
            np.testing.assert_allclose(dynamics.emission.sum(-1), 1.)
            self.assertTrue(np.all(dynamics.emission[:, nonlinear.core.CURRENT == 1, -1] == 0.))
        self.assertEqual(changed.fit_receipt['state_count'], 8)
        self.assertEqual(changed.fit_receipt['fit_module'], 'nonlinear_repairability')

    def test_cached_features_require_exact_paid_input_and_never_use_reference(self):
        rows = [{'id': 'q', 'answer': 'SECRET'}]
        tree = {path: {'prompt_sha256': path or 'root', 'answer': 'A'} for path in development.PATHS}
        class Observer:
            _recipe = {'max_length': 512}
            def entry(self, row, path, paid):
                self_test.assertEqual(row['answer'], '')
                self_test.assertEqual(set(paid), {'', path})
                return {'id': row['id'], 'path': path, 'answer': paid[path]['answer']}
        self_test = self; observer = Observer()
        metadata = [{'id': 'q', 'path': p, 'token_slots': 20, 'prompt_sha256': tree[p]['prompt_sha256'],
                     'input_sha256': development.fingerprint({'id': 'q', 'path': p, 'answer': 'A'})}
                    for p in development.PATHS]
        outputs = {k: np.zeros((3, width)) for k, width in (('state', 4), ('plain', 8), ('guided', 8))}
        values, _ = development.verify_features(rows, {'q': tree}, observer, metadata, outputs, True)
        self.assertEqual(values['q', 'repair'][3], 20)
        metadata[1]['input_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'exact paid'):
            development.verify_features(rows, {'q': tree}, observer, metadata, outputs, True)


if __name__ == '__main__':
    unittest.main()
