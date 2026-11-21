"""A mathematically worthless review must not be purchased due to summation noise."""

import unittest
from types import SimpleNamespace

import numpy as np

import stable_pomdp
import grounded_pomdp
import tree_context_pomdp
import calibrated_review_pomdp
import stacked_pomdp
import efficient_stacked_pomdp


class ReviewRoundoff(unittest.TestCase):
    def test_zero_information_value_does_not_require_unpaid_review(self):
        def values(belief, remaining, penalty, transition, reviewed=False, reuse=False):
            # Perturb every branch by less than a meaningful utility unit,
            # as can happen when a finite observation expectation is summed.
            return np.array([.8, .7, .6]) * belief.sum() + (1e-14 if reviewed else 0.)

        dynamics = SimpleNamespace(
            prior=np.array([.2, .8, 0., 0.]), update=lambda p, l: p * l / (p @ l),
            values=values, likelihood=lambda *a: np.ones((4, 31)) / 31,
            review_costs=np.zeros((2, 4)))
        model = SimpleNamespace(
            dynamics={'math': dynamics}, context_weight=0., review_observer=None,
            observed=lambda *a: 0, likelihood=lambda *a: np.ones(4),
            context=lambda *a: np.zeros((31, 2, 4, 4)) if a[4] else np.zeros((2, 4, 4)),
            classifier=SimpleNamespace(charges=lambda *a, **kw: (0, 0, 0, 0.)))
        row = {'id': 'x', 'domain': 'math', 'subject': 'math', 'answer': '7'}
        tree = {'': {'answer': '7', 'input_tokens': 10, 'output_tokens': 1}}
        for module in (stable_pomdp, grounded_pomdp, tree_context_pomdp,
                       calibrated_review_pomdp, stacked_pomdp, efficient_stacked_pomdp):
            result = module.evaluate(row, tree, {}, model,
                                     {'penalty': 0., 'continuous_filter': True}, 'C')
            self.assertEqual(result['judge_calls'], 0)
            self.assertEqual(result['decisions'][0]['action'], 'stop')
            self.assertGreater(result['decisions'][0]['information_value'], 0.)
            self.assertLess(result['decisions'][0]['information_value'], 1e-9)


if __name__ == '__main__':
    unittest.main()
