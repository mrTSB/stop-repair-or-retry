import unittest

from c80_targets import BASELINES, assess


def table(n=100, baseline=55, a=65, b=70, c=81):
    return [{'method': name, 'n': n, 'correct': correct, 'accuracy': correct/n, 'mean_compute': cost}
            for name, correct, cost in [*((k,baseline,100) for k in BASELINES), ('A',a,90), ('B',b,200), ('C',c,190)]]


class TargetTests(unittest.TestCase):
    def test_exactly80_is_not_over80_and_development_is_not_fresh(self):
        result = assess(table(c=80))
        self.assertTrue(result['checks']['C_gap_at_least10pp'])
        self.assertFalse(result['checks']['C_accuracy_strictly_over80percent'])
        result = assess(table())
        self.assertTrue(result['all_metrics_passed'])
        self.assertFalse(result['fresh_targets_confirmed'])
        self.assertTrue(assess(table(), fresh_validation=True)['fresh_targets_confirmed'])

    def test_current_panel_needs28_more_correct_C_and5_A(self):
        result = assess(table(256,132,153,177,177))
        self.assertEqual(result['minimum_correct'], {'A':158,'B':158,'C':205})
        self.assertEqual(result['additional_correct_needed'], {'A':5,'B':0,'C':28})
        self.assertFalse(result['all_metrics_passed'])

    def test_fresh1024_threshold_and_stronger_B_raise_C_target(self):
        self.assertFalse(assess(table(1024,500,610,700,819))['checks']['C_accuracy_strictly_over80percent'])
        self.assertTrue(assess(table(1024,500,610,700,820))['checks']['C_accuracy_strictly_over80percent'])
        self.assertEqual(assess(table(100,60,80,91,100))['minimum_correct']['C'],101)
        self.assertFalse(assess(table(100,60,80,91,100))['accuracy_targets_feasible_on_this_panel'])

    def test_no_equal_cost_rounded_accuracy_or_missing_baseline_pass(self):
        rows = table()
        rows[-1]['mean_compute'] = rows[-2]['mean_compute']
        self.assertFalse(assess(rows)['checks']['C_compute_strictly_below_B'])
        rows[-1]['accuracy'] = .82
        with self.assertRaises(ValueError): assess(rows)
        with self.assertRaises(ValueError): assess(table()[1:])


if __name__ == '__main__': unittest.main()
