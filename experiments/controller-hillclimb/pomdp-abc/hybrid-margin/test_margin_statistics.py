import unittest

from study_common import BASELINES,paired_margins


class PairedMaximumComparison(unittest.TestCase):
    def test_highest_baseline_is_recomputed_in_each_resample(self):
        rows=[{'subject':'physics'},{'subject':'physics'}]
        outcomes={name:[0,0] for name in BASELINES}
        outcomes.update({'Stop':[1,0],'Repair once':[0,1],
                         'A':[1,0],'B':[1,0],'C':[1,1]})
        results={name:[{'correct':answer,'compute':10 if name=='C' else 12}
                       for answer in values] for name,values in outcomes.items()}
        report=paired_margins(rows,results,repetitions=1000)
        # Stop and Repair once tie on the original sample, but each can win a resample.
        self.assertEqual(report['A_above_highest_baseline']['point'],0.)
        self.assertEqual(report['A_above_highest_baseline']['ci95'],[-1.,0.])
        self.assertEqual(report['C_above_B']['point'],.5)
        self.assertEqual(report['C_compute_saved_vs_B'],{'point':2.,'ci95':[2.,2.]})


if __name__=='__main__':unittest.main()
