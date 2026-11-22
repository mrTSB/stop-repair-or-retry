import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import regularized_completion_pomdp as new
import fit_regularized_completion as driver
import test_nested_dense_diagnostic as fixtures


class RegularizedCompletionTests(unittest.TestCase):
    def test_dynamics_identical_to_frozen_fit_given_same_forecasts(self):
        rows = [{'id': f'q{i}', 'subject': 'chemistry' if i % 2 else 'physics', 'answer': 'A'} for i in range(36)]
        paths = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry',
                 'complete', 'repair/complete', 'retry/complete')
        trees = {r['id']: {p: {'answer': 'A' if (i + j) % 3 else 'B', 'text': str(j),
            'input_tokens': 100 + j, 'output_tokens': 8} for j, p in enumerate(paths)} for i, r in enumerate(rows)}
        observer = SimpleNamespace(_recipe={'parameters': 100000000, 'max_length': 512})
        lookup = {(r['id'], p): None for r in rows for p in ('', 'repair', 'retry')}
        def features(row, path, tree, neural):
            return [int(row['id'][1:]), float(bool(path)), float(tree[path]['text'])]
        fitted = []; original = new.core.classifier_for
        def record(x, y, c):
            model = original(x, y, c); fitted.append(model); return model
        with threadpool_limits(limits=1), patch.object(new.core.plain, 'observables', side_effect=features), \
                patch.object(new.core, 'classifier_for', side_effect=record):
            old = new.core.fit(rows, trees, observer, lookup)
        records = [{'id': r['id'], 'path': p} for r in rows for p in ('', 'repair', 'retry')]
        probabilities = []
        for row in rows:
            fold = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16) % 3
            for p in ('', 'repair', 'retry'):
                probabilities.append(new.core.probabilities(fitted[fold], features(row, p, trees[row['id']], None)))
        model = new.fit_dynamics(rows, trees, observer, records, probabilities, old.forecaster, 'raw20_control', {})
        for subject in old.dynamics:
            for field in ('prior', 'transition', 'emission', 'cost'):
                np.testing.assert_array_equal(getattr(model.dynamics[subject], field), getattr(old.dynamics[subject], field))
        with self.assertRaises(ValueError):
            new.fit_dynamics(rows, trees, observer, records[::-1], probabilities, old.forecaster, 'raw20_control', {})

    def test_dense_forward_is_shared_trimmed_and_not_serialized(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.project = torch.nn.Identity(); self.calls = 0
            def forward(self, input_ids, attention_mask, numeric):
                self.calls += 1; self.width = input_ids.shape[1]
                z = self.project(torch.ones((1, 256)))
                return z[:, :4], z[:, :8], z[:, :8]
        observer = new.DenseObserver('synthetic', 1.)
        observer._model = Net(); observer._recipe = {'max_length': 512}
        tensors = {'input_ids': torch.ones((1, 512), dtype=torch.long),
            'attention_mask': torch.cat((torch.ones(1, 13), torch.zeros(1, 499)), 1), 'numeric': torch.zeros(1, 2)}
        row = {'id': 'q', 'answer': 'SECRET'}; tree = {'': {'prompt_sha256': 'p', 'text': 'paid', 'answer': 'A'},
            'repair': {'unpaid': True}}
        def entry(public, path, paid):
            self.assertEqual(public['answer'], ''); self.assertEqual(set(paid), {''}); return {}
        with patch.object(observer, 'load'), patch.object(observer, 'entry', side_effect=entry), \
                patch.object(new.perception, 'encode', return_value=tensors):
            a = observer.values(row, '', tree); b = observer.values(row, '', tree)
        self.assertEqual(observer._model.calls, 1); self.assertEqual(observer._model.width, 13)
        self.assertEqual(a[3], 13); self.assertEqual(b[3], 13)
        self.assertEqual(observer.__getstate__()['_dense_cache'], {})
        with self.assertRaises(ValueError): observer.values(row, '', tree, judge={})

    def test_dense_planner_reads_only_paid_prefix(self):
        class Observer:
            def features(inner, row, path, tree):
                self.assertEqual(row['answer'], ''); self.assertEqual(set(tree), {'', 'repair'}); return np.zeros(386)
        class Forecast:
            classes_ = np.arange(16)
            def predict_proba(inner, values): return np.ones((1, 16)) / 16
        model = new.Planner(Observer(), Forecast(), {}, {}, 'raw20_plus_project256')
        p = model.predicted({'id': 'q', 'subject': 'physics', 'answer': 'A'}, 'repair',
            {'': {'answer': 'B'}, 'repair': {'answer': ''}, 'retry': {'unpaid': True}})
        self.assertEqual(float(p @ new.core.CURRENT), 0.)

    def test_final_training_selection_is_separate_from_emission_forecasts(self):
        data = fixtures.fixture()
        with threadpool_limits(limits=1): model, receipt = driver.final_forecaster(data, 'raw20_control')
        self.assertEqual(len(receipt['search']), 4)
        self.assertTrue(receipt['emissions_use_separate_nested_OOF'])
        self.assertTrue(receipt['scores_are_training_selection_not_validation'])
        self.assertEqual(model[-1].n_features_in_, 130)


if __name__ == '__main__': unittest.main()
