import math
import unittest

import bounded_arithmetic as tool


class ArithmeticTests(unittest.TestCase):
    def test_valid_numerical_work(self):
        self.assertAlmostEqual(tool.calculate('sqrt(2)*sin(pi/4)'), 1.)
        self.assertAlmostEqual(tool.calculate('log(exp(2.5))'), 2.5)
        self.assertAlmostEqual(tool.calculate('6.62607015e-34*299792458/(500e-9)'), 3.972891714297857e-19)
        self.assertAlmostEqual(tool.calculate('degrees(atan(1))'), 45.)
        self.assertEqual(tool.calculate('-(3+2)**2/5'), -5.)

    def test_only_bounded_scalar_ast(self):
        for expression in ('__import__("os")', 'open("file")', '(1).__class__', 'x+1',
                '[1,2][0]', '(lambda: 1)()', 'True+1', '"text"', '2**1000000',
                '10**100*10', 'exp(1000)', '1/0', 'sqrt(-1)', '(2 if 1 else 3)',
                'sum(i for i in range(10))', 'sin(x=2)', 'log(2,3)', '1e101'):
            with self.subTest(expression=expression), self.assertRaises((ValueError, TypeError, ArithmeticError)):
                tool.calculate(expression)
        with self.assertRaises(ValueError): tool.calculate('1+'*128+'1')

    def test_tool_result_limits_and_errors_are_preserved(self):
        text = '<calc>sqrt(4)</calc><calc>open("file")</calc>'+('<calc>pi</calc>'*20)
        values = tool.evaluate_text(text)
        self.assertEqual(len(values), 8); self.assertEqual(values[0]['value'], '2')
        self.assertFalse(values[1]['ok']); self.assertNotIn('value', values[1])
        self.assertEqual(tool.evaluate_text('No arithmetic needed'), [])


if __name__ == '__main__': unittest.main()
