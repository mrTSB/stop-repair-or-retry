import unittest
from types import SimpleNamespace

import numpy as np

import expanded_completion_variants as variants


class ExpandedVariantTests(unittest.TestCase):
    def test_original_partition_zero_weight_preserves_all_decisions_and_costs(self):
        core = variants.original.core
        rng = np.random.default_rng(15)
        transition = rng.random((2, 16, 16)); transition /= transition.sum(2, keepdims=True)
        emission = rng.random((2, 16, 30)); emission /= emission.sum(2, keepdims=True)
        dynamics = core.Dynamics(np.ones(16)/16, transition, emission, rng.uniform(50, 400, (2, 3, 16)))
        observer = SimpleNamespace(charges=lambda *a, **kw: (100, 1, 100, .435), features=lambda *a: np.zeros(386))
        class Forecast:
            classes_ = np.arange(16)
            def predict_proba(self, x): return np.asarray([rng.dirichlet(np.ones(16))])
        model = variants.original.Planner(observer, Forecast(), {'physics': dynamics}, {}, 'raw20_plus_project256')
        wrapped = variants.wrap(model)
        row = {'id': 'q', 'subject': 'physics', 'domain': 'science', 'answer': 'A'}
        paths = ('', 'repair', 'retry', 'complete', 'repair/repair', 'repair/retry', 'repair/complete', 'retry/repair', 'retry/retry', 'retry/complete')
        tree = {p: {'answer': 'A' if i % 2 else '', 'text': 'cached', 'input_tokens': 100+i, 'output_tokens': 2} for i, p in enumerate(paths)}
        for penalty in (0., .02, .1):
            state = rng.bit_generator.state
            before = variants.original.evaluate(row, tree, {}, model, {'penalty': penalty}, 'A')
            rng.bit_generator.state = state
            after = variants.evaluate_science(row, tree, {}, wrapped, {'penalty': penalty, 'posterior_weight': 0.}, 'A')
            self.assertEqual(before, after)
        self.assertIsNot(wrapped.dynamics, model.dynamics)

    def test_kernel_matches_original_partition_and_rejects_llm_route(self):
        core = variants.original.core
        for p in np.random.default_rng(8).dirichlet(np.ones(16), 80):
            for valid in (True, False):
                self.assertEqual(variants.FixedKernel().signal(p, valid), core.observation(p, valid))
        with self.assertRaises(ValueError): variants.evaluate({}, {}, {}, None, {}, 'B')


if __name__ == '__main__': unittest.main()
