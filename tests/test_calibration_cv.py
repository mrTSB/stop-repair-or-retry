"""Offline invariants for unsupported-observation calibration and grouped CV."""
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/pomdp-controller'))
spec=importlib.util.spec_from_file_location('calibration_cv',ROOT/'experiments/pomdp-controller/calibration_cv.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class CalibrationTests(unittest.TestCase):
    def test_unseen_likelihood_preserves_prior_and_prediction(self):
        table=v.unknown_equal([[.8,.15,.05],[.2,.4,.4]],[[8,2,0],[2,8,0]])
        for row in table:self.assertAlmostEqual(sum(row),1)
        self.assertEqual(table[0][2],table[1][2])
        prior=[.7,.3]
        m={'prior':prior,'initial_observation':table,'transition':{'repair':[[.8,.2],[.1,.9]]},'observation':{'repair':table}}
        for a,b in zip(v.p.initial_belief(m,2),prior):self.assertAlmostEqual(a,b)
        for a,b in zip(v.p.update(m,prior,'repair',2),v.p.predict(m,prior,'repair')):self.assertAlmostEqual(a,b)
        self.assertAlmostEqual(table[0][0]/table[0][1],.8/.15)

    def test_all_unseen_is_uniform_and_supported_unchanged(self):
        self.assertEqual(v.unknown_equal([[.7,.3],[.2,.8]],[[0,0],[0,0]]),[[.5,.5],[.5,.5]])
        self.assertEqual(v.unknown_equal([[.7,.3],[.2,.8]],[[1,0],[0,1]]),[[.7,.3],[.2,.8]])

    def test_folds_are_question_grouped_balanced_and_order_independent(self):
        rows=[{'id':f'{domain}-{i}','domain':domain} for domain in ['math','science'] for i in range(50)]
        fold=v.folds(rows);self.assertEqual(fold,v.folds(list(reversed(rows))))
        for k in range(5):
            fitting={r['id'] for r in rows if fold[r['id']]!=k};held={r['id'] for r in rows if fold[r['id']]==k}
            self.assertEqual(len(fitting),80);self.assertEqual(len(held),20);self.assertFalse(fitting&held)
            self.assertEqual(sum(fold[r['id']]==k for r in rows if r['domain']=='math'),10)

    def test_fit_rejects_dev_and_duplicate_ids_before_accessing_trees(self):
        for rows in ([{'split':'dev','id':'a'}],[{'split':'train','id':'a'},{'split':'train','id':'a'}]):
            with self.assertRaises(ValueError):v.fit(rows,'pooled')

if __name__=='__main__':unittest.main()
