import copy
import unittest
import numpy as np
from threadpoolctl import threadpool_limits

import clustered_completion_pomdp as policy


def fixture():
    rows = [{'id': str(i), 'subject': 'chemistry' if i < 20 else 'physics', 'answer': 'A'} for i in range(40)]
    paths = ('', 'repair', 'retry', 'complete', 'repair/repair', 'repair/retry', 'repair/complete', 'retry/repair', 'retry/retry', 'retry/complete')
    trees = {r['id']: {p: {'answer': '' if (i+len(p)) % 3 == 0 else 'A'} for p in paths} for i, r in enumerate(rows)}
    records = [{'id': r['id'], 'path': p} for r in rows for p in ('', 'repair', 'retry')]
    valid = np.asarray([bool(trees[r['id']][r['path']]['answer']) for r in records])
    probabilities = np.random.default_rng(5).dirichlet(np.ones(16), len(records))
    probabilities = np.asarray([policy.core.mask_invalid(p, v) for p, v in zip(probabilities, valid)])
    return rows, trees, records, valid, probabilities


class ClusteredObservationTests(unittest.TestCase):
    def test_partition_uses_child_forecasts_only_and_preserves_validity_symbols(self):
        _, _, records, valid, probabilities = fixture()
        with threadpool_limits(limits=1):
            kernel = policy.fit_kernel(records, probabilities, valid)
            changed = probabilities.copy()
            changed[np.asarray([r['path'] == '' for r in records])] = 1/16
            second = policy.fit_kernel([{**r, 'unused_reference': 'SECRET'} for r in records], changed, valid)
        np.testing.assert_array_equal(kernel.valid_centers, second.valid_centers)
        np.testing.assert_array_equal(kernel.invalid_centers, second.invalid_centers)
        for p, v in zip(probabilities, valid):
            signal = kernel.signal(p, v)
            self.assertTrue(0 <= signal < 24 if v else 24 <= signal < 30)

    def test_rebin_preserves_dynamics_costs_and_exact_invalid_support(self):
        rows, trees, records, valid, probabilities = fixture()
        core = policy.core
        dynamics = {s: core.Dynamics(np.ones(16)/16, np.ones((2, 16, 16))/16,
                    np.ones((2, 16, 30))/30, np.ones((2, 3, 16))*100) for s in ('chemistry', 'physics')}
        base = policy.previous.previous.Planner(None, None, dynamics, {'smoothing': 32.}, 'raw20_plus_project256')
        before = copy.deepcopy(base)
        with threadpool_limits(limits=1): model = policy.rebin(base, rows, trees, records, probabilities)
        for subject, d in model.dynamics.items():
            for name in ('prior', 'transition', 'cost'):
                np.testing.assert_array_equal(getattr(d, name), getattr(before.dynamics[subject], name))
            np.testing.assert_allclose(d.emission.sum(-1), 1.)
            np.testing.assert_array_equal(d.emission[:, core.CURRENT == 1, 24:], 0.)
            self.assertEqual(d.values(np.ones(16)/16, 2, .02).shape, (4,))


if __name__ == '__main__': unittest.main()
