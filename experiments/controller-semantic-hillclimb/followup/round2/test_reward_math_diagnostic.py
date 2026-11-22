import copy
import unittest

import numpy as np

import reward_math_diagnostic as run


class MathRewardTests(unittest.TestCase):
    def test_even_embedding_preserves_current_repair_retry_marginals(self):
        rng = np.random.default_rng(8); p = rng.dirichlet(np.ones(8), size=10)
        embedded = np.zeros((10, 16)); embedded[:, ::2] = p
        np.testing.assert_allclose(run.collapse(embedded), p)
        bits16 = np.stack([(np.arange(16) >> n)&1 for n in (3, 2, 1)], axis=1)
        bits8 = np.stack([(np.arange(8) >> n)&1 for n in (2, 1, 0)], axis=1)
        np.testing.assert_allclose(embedded@bits16, run.collapse(embedded)@bits8)
        with self.assertRaises(ValueError): run.collapse(p)

    def test_conditional_decisions_exclude_held_future_truth_and_costs(self):
        rows = [{'id': str(i), 'answer': '4', 'subject': 'math'} for i in range(12)]
        records = [{'id': r['id'], 'path': p, 'fold': int(r['id'])%3} for r in rows for p in ('', 'repair', 'retry')]
        paths = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
        trees = {r['id']: {p: {'answer': '4', 'input_tokens': 20, 'output_tokens': 5} for p in paths} for r in rows}
        slots = {(r['id'], r['path']): 10 for r in records}
        p = np.tile([.1, .05, .65, .05, .05, .025, .05, .025], (len(records), 1))
        before = run.conditional(rows, trees, {'records': records}, p, slots, 1000000000)
        changed = copy.deepcopy(trees)
        for r in rows:
            if int(r['id'])%3 == 0:
                for path in paths:
                    if '/' in path: changed[r['id']][path].update(answer='99', input_tokens=10**9)
        after = run.conditional(rows, changed, {'records': records}, p, slots, 1000000000)
        held = lambda results: [(r['id'], r['path'], r['action']) for r in results if int(r['id'])%3 == 0]
        self.assertEqual(held(before), held(after))
        self.assertTrue(all(r['action'] in (0, 1, 2) for r in before))
        self.assertEqual(len(before), 24)


if __name__ == '__main__': unittest.main()
