import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import development_regularized_completion as driver


class DevelopmentRegularizedTests(unittest.TestCase):
    def test_cached_features_hide_reference_and_unpaid_branches(self):
        observer=driver.CachedDense('synthetic',1.)
        row={'id':'q','answer':'SECRET'}
        tree={'':{'text':'root'},'repair':{'text':'paid'},'retry':{'text':'UNPAID'}}
        def entry(public,path,paid):
            self.assertEqual(public['answer'],'');self.assertEqual(set(paid),{'','repair'})
            return {'path':path,'text':paid[path]['text']}
        observer.expected={('q','repair'):driver.nested.fixed.fingerprint({'path':'repair','text':'paid'})}
        observer.saved={('q','repair'):(np.zeros(4),np.zeros(8),np.zeros(8),123)}
        observer.joint={('q','repair'):np.ones(256)}
        def numeric(public,path,paid,values):
            self.assertEqual(public['answer'],'');self.assertEqual(set(paid),{'','repair'})
            self.assertEqual(values[-1],123);return np.zeros(130)
        with patch.object(observer,'entry',side_effect=entry),patch.object(driver.runtime.core.plain,'observables',side_effect=numeric):
            self.assertEqual(observer.features(row,'repair',tree).shape,(386,))
            changed=copy.deepcopy(tree);changed['repair']['text']='altered'
            with self.assertRaises(ValueError):observer.values(row,'repair',changed)
            with self.assertRaises(ValueError):observer.values(row,'repair',tree,judge={})

    def test_partial_replay_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'registration.json').write_text('{}');(folder/'features').mkdir()
            driver.preserve_partial(folder)
            (folder/'raw20_control-0.pkl').write_bytes(b'partial')
            with self.assertRaises(ValueError):driver.preserve_partial(folder)

    def test_all_original_and_matched_controls_are_recomputed(self):
        rows=[{'id':s,'subject':s} for s in ('math','chemistry','physics')]
        trees={r['id']:{} for r in rows};math={'math':{'score':7}}
        statistics=lambda items:sum(v['score'] for v in items)
        baseline={n:6 for n in driver.original.BASELINES}
        matched={n:{'full256_with_retained_math_A':11,'science':{'all':4,'chemistry':2,'physics':2}}
            for n,*_ in driver.original.MATCHED}
        summary={'baselines':baseline,'matched_complete_controls':matched}
        with patch.object(driver.original,'statistics',side_effect=statistics), \
                patch.object(driver.original,'fixed',return_value={'score':2}) as fixed, \
                patch.object(driver.original,'matched_item',return_value={'score':2}) as complete:
            self.assertEqual(driver.controls(rows,trees,math,summary),(baseline,matched))
            self.assertEqual(fixed.call_count,15);self.assertEqual(complete.call_count,12)
            corrupt=copy.deepcopy(summary);corrupt['matched_complete_controls'][next(iter(matched))]['science']['all']=3
            with self.assertRaises(ValueError):driver.controls(rows,trees,math,corrupt)

    def test_test_question_access_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test-questions.jsonl';path.write_text('SECRET')
            with driver.parent_io.no_test_reads(),self.assertRaises((ValueError,RuntimeError,AssertionError)):
                path.read_text()


if __name__=='__main__':unittest.main()
