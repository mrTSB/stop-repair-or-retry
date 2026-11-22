import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import policy_crossfit_diagnostic as run
from test_clustered_completion import fixture


class WholePolicyTests(unittest.TestCase):
    def test_outer_questions_never_enter_any_forecaster_fit_and_costs_use_paid_slots(self):
        rows,trees,records,valid,_=fixture();core=run.runtime.core
        for row in rows:row['domain']='science'
        for tree in trees.values():
            for node in tree.values():node.update(text='synthetic paid answer',input_tokens=100,output_tokens=3)
        rng=np.random.default_rng(7);x=rng.normal(size=(len(records),386));labels=[]
        for i,r in enumerate(records):
            r['fold']=int(r['id'])%3;r['subject']=rows[int(r['id'])]['subject'];x[i,0]=int(r['id'])
            path=r['path'];prefix=path+'/' if path else ''
            labels.append(core.state(*[core.plain.correct(trees[r['id']][p],'A') for p in (path,*[prefix+a for a in ('repair','retry','complete')])]))
        keys=[(r['id'],r['path']) for r in records]
        cache=run.CachedObserver({'parameters':435000000,'max_length':512},
            {k:(np.zeros(4),np.zeros(8),np.zeros(8),17) for k in keys},dict(zip(keys,x)),
            {k:run.fixed.fingerprint(trees[k[0]][k[1]]) for k in keys})
        data={'records':records,'features':{run.nested.FAMILIES[1]:x},'labels':np.array(labels),'valid':valid}
        seen=[];original=run.forecast.fit
        def fit(values,*args):seen.append(set(values[:,0].astype(int)));return original(values,*args)
        with threadpool_limits(limits=1),patch.object(run.forecast,'fit',side_effect=fit):
            results,receipt=run.fit_fold(rows,trees,cache,data,40,run.nested.FAMILIES[1],0)
        held={i for i in range(40) if i%3==0}
        self.assertEqual(set(results),{str(i) for i in held});self.assertEqual(len(seen),4)
        self.assertTrue(all(not(ids&held) for ids in seen));self.assertEqual(len(receipt['emission_folds']),3)
        for item in results.values():
            self.assertEqual(item['verifier_token_slots'],17*item['verifier_calls'])
            self.assertLessEqual(item['solver_calls'],3);self.assertEqual(item['judge_calls'],0)
        public={**rows[0],'answer':''};paid={'':trees['0']['']}
        with self.assertRaises(ValueError):cache.values(rows[0],'',paid)
        with self.assertRaises(ValueError):cache.values(public,'',{**paid,'retry':trees['0']['retry']})


if __name__=='__main__':unittest.main()
