import unittest
from numeric_consistency import number, observations


class ArithmeticObservationsTest(unittest.TestCase):
    def test_true_and_false_arithmetic(self):
        self.assertEqual(observations('3 × 4 = 12. Final: 12')[4], 0.)
        self.assertEqual(observations('3 × 4 = 13. Final: 13')[4], 1.)

    def test_chains_fractions_and_percentages(self):
        self.assertEqual(observations('20 - 5 = 15 / 3 = 5')[4], 1.)
        self.assertEqual(observations('20 - 5 = 15; 15 / 3 = 5')[4], 0.)
        self.assertEqual(observations('1 / 3 = 0.33')[4], 0.)
        self.assertEqual(observations('20% of 60 = 12')[4], 0.)
        self.assertEqual(observations('1,000 / 4 = 250')[4], 0.)

    def test_never_execute_text_or_unbounded_expressions(self):
        self.assertIsNone(number('__import__("os").getcwd()'))
        self.assertIsNone(number('10 ** 10000000'))
        self.assertIsNone(number('1/0'))
        self.assertIsNone(number('2 ** 2 ** 2 ** 2 ** 2'))
        self.assertEqual(number('-(3+2)/2'), -2.5)


if __name__ == '__main__':
    unittest.main()
