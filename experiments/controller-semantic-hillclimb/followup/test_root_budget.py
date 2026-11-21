"""A numerical POMDP fixture checks root-only information and forced submission."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medium_pomdp  # Initialize the frozen parent-module paths.
from difficulty_pomdp import Dynamics, N_OBS
from context_pomdp import REVIEW_WIDTH
from root_medium_pomdp import evaluate


class RootBudgetTests(unittest.TestCase):
    def test_repairs_failed_transfer_without_purchasing_unavailable_late_review(self):
        prior = np.array([.5, .5, 0., 0.])
        initial = np.zeros((4, N_OBS)); initial[:, 1] = 1.
        transition = np.zeros((2, 2, 4, 4))
        reviewed = np.zeros((2, REVIEW_WIDTH, 2, 4, 4))
        for h in (0, 1):
            for stage in (0, 1):
                for action in (0, 1):
                    p = (.6 if stage == 0 else .7) if action == 0 else .4
                    transition[stage, action, 2*h:2*h+2, 2*h:2*h+2] = [1-p, p]
                    q = (.8 if stage == 0 else .99) if action == 0 else .4
                    reviewed[stage, :, action, 2*h:2*h+2, 2*h:2*h+2] = [1-q, q]
        emission = np.zeros((2, 2, 4, N_OBS))
        emission[:, :, [0, 2], 2] = 1.
        emission[:, :, [1, 3], 3] = 1.
        review = np.zeros((2, N_OBS, 4, REVIEW_WIDTH))
        review[:, :, [0, 2], 0] = 1.
        review[:, :, [1, 3], 1] = 1.
        d = Dynamics(prior, initial, transition, emission, np.zeros((2, 2, 4)),
                     review, reviewed, emission.copy(), np.zeros((2, 2, 4)),
                     np.zeros((2, 4)), np.full((N_OBS, 4, 3), 1/3))
        observed_paths = []

        def observed(row, path, tree):
            # There is no decision left after the second solver continuation.
            self.assertIn(path, ('', 'guided'))
            observed_paths.append(path)
            return 1 if path == '' else 2

        observer = SimpleNamespace(charges=lambda *a, **k: (20, 2, 20, .184))
        model = SimpleNamespace(dynamics={'math': d}, classifier=observer,
                                review_observer=None, observed=observed,
                                review_signal=lambda *a: 0)
        row = {'id': 'fixture', 'domain': 'math', 'subject': 'math', 'answer': '1'}
        tree = {path: {'answer': answer, 'input_tokens': 1, 'output_tokens': 1}
                for path, answer in [('', '0'), ('guided', '0'), ('guided/repair', '1')]}
        judges = {('fixture', ''): {'answer': '1', 'parse_valid': True,
                                     'input_tokens': 1, 'output_tokens': 2}}
        item = evaluate(row, tree, judges, model,
                        {'penalty': 0., 'plan_reviews': False, 'reuse_review': True}, 'C')
        self.assertEqual(item['path'], 'guided/repair')
        self.assertEqual(item['correct'], 1)
        self.assertEqual([v['queried'] for v in item['decisions']], [True, False])
        self.assertEqual((item['solver_calls'], item['judge_calls']), (3, 1))
        self.assertNotIn('guided/repair', observed_paths)
        self.assertAlmostEqual(item['compute'], (4 * 6 + 3 * 3 + .184 * 20) / 3.09)
        with self.assertRaises(ValueError):
            evaluate(row, tree, judges, model, {'penalty': 0., 'plan_reviews': True}, 'C')


if __name__ == '__main__':
    unittest.main()
