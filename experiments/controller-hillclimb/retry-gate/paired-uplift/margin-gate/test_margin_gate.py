import unittest

import numpy as np

from arithmetic import correction
from margin_model import maximize_value


def math_row(text, answer, question='How many items are left?'):
    return {'domain': 'math', 'question': question, 'compact': {'text': text, 'answer': answer}}


class ExactArithmeticTests(unittest.TestCase):
    def test_consistent_chain_has_no_correction(self):
        row = math_row('Total = 6 + (2/3 × 6) + (3 × 8) = 6 + 4 + 24 = 34\nFinal: 34', '34')
        self.assertEqual(correction(row), '')

    def test_inconsistent_intermediate_is_not_used(self):
        row = math_row('Total = 5 * (10 + 20) = 100 + 75 = 160\nFinal: 160', '160')
        self.assertEqual(correction(row), '')

    def test_matching_full_expression_can_fix_a_reported_total(self):
        row = math_row('Amount = (30,000/8) * (500/750) * 4 = 7500\nFinal: 7500', '7500')
        self.assertEqual(correction(row), '10000')

    def test_rounding_and_foreign_result_are_preserved(self):
        row = math_row('5/3 = 2\nFinal: 2', '2', 'Round to the nearest integer.')
        self.assertEqual(correction(row), '')
        row = math_row('200 - (40 * 2) = 160\nFinal: 10', '10')
        self.assertEqual(correction(row), '')


class AllocationTests(unittest.TestCase):
    def test_uses_best_feasible_group_instead_of_best_ratio(self):
        result = maximize_value(np.array([10., 8., 8.]), np.array([7, 6, 6]), 12)
        self.assertEqual(result.tolist(), [False, True, True])

    def test_rejects_nonpositive_and_unpayable_requests(self):
        result = maximize_value(np.array([-1., 1.]), np.array([2, 7]), 6)
        self.assertEqual(result.tolist(), [False, False])


if __name__ == '__main__':
    unittest.main()
