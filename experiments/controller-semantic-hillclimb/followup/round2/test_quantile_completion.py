import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import quantile_completion_pomdp as new


class QuantilePOMDPTests(unittest.TestCase):
    def test_fixed_kernel_exactly_matches_original_observation(self):
        kernel=new.Kernel(np.asarray([.3,.7,.97]),np.full(3,.5));rng=np.random.default_rng(23)
        for valid in (False,True):
            for _ in range(50):
                p=rng.uniform(size=16);p/=p.sum()
                self.assertEqual(kernel.signal(p,valid),new.core.observation(p,valid))

    def test_thresholds_ignore_root_and_reference_fields(self):
        rng=np.random.default_rng(24);records=[{'id':str(i),'path':p,'answer':'SECRET'} for i in range(30) for p in ('','repair','retry')]
        p=rng.uniform(size=(90,16));p/=p.sum(1,keepdims=True);valid=np.ones(90,bool)
        a=new.fit_kernel(records,p,valid);changed=copy.deepcopy(records)
        for r in changed:r['answer']='DIFFERENT'
        p2=p.copy();p2[::3]=0.;p2[::3,15]=1.
        b=new.fit_kernel(changed,p2,valid)
        np.testing.assert_array_equal(a.current_edges,b.current_edges);np.testing.assert_array_equal(a.gain_edges,b.gain_edges)
        self.assertEqual(len(a.current_edges),3);self.assertTrue(np.all(np.diff(a.current_edges)>=0))

    def test_fixed_rebin_matches_all_original_dynamics(self):
        rows=[{'id':str(i),'subject':'chemistry' if i%2 else 'physics','answer':'A'} for i in range(30)]
        paths=('','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry','complete','repair/complete','retry/complete')
        trees={r['id']:{p:{'answer':'A' if (i+j)%3 else 'B','input_tokens':100,'output_tokens':5}
            for j,p in enumerate(paths)} for i,r in enumerate(rows)}
        records=[{'id':r['id'],'path':p} for r in rows for p in ('','repair','retry')]
        rng=np.random.default_rng(25);oof=rng.uniform(size=(90,16));oof/=oof.sum(1,keepdims=True)
        observer=SimpleNamespace(_recipe={'parameters':435000000,'max_length':512})
        base=new.previous.fit_dynamics(rows,trees,observer,records,oof,SimpleNamespace(), 'raw20_control',{})
        fixed=new.rebin(base,rows,trees,records,oof,new.Kernel(np.array([.3,.7,.97]),np.full(3,.5)))
        adaptive=new.rebin(base,rows,trees,records,oof)
        for subject in base.dynamics:
            for field in ('prior','transition','cost','emission'):
                np.testing.assert_array_equal(getattr(base.dynamics[subject],field),getattr(fixed.dynamics[subject],field))
            for field in ('prior','transition','cost'):
                np.testing.assert_array_equal(getattr(base.dynamics[subject],field),getattr(adaptive.dynamics[subject],field))
            np.testing.assert_allclose(adaptive.dynamics[subject].emission.sum(-1),1.)
        self.assertFalse(np.array_equal(base.dynamics['physics'].emission,adaptive.dynamics['physics'].emission))

    def test_fixed_kernel_runtime_matches_decisions_and_every_charge(self):
        row={'id':'q','domain':'science','subject':'physics','answer':'A'}
        tree={p:{'answer':'A' if p=='repair/complete' else '', 'text':p,'input_tokens':100+i,'output_tokens':1}
            for i,p in enumerate(('','repair','repair/complete'))}
        class Dynamics:
            transition=np.tile(np.eye(16),(2,1,1));emission=np.ones((2,16,30))/30
            def values(self,belief,remaining,penalty):return np.eye(4)[1 if remaining==2 else 3]
        class Observer:
            def charges(self,public,paid,judges,decisions,include_reviews):
                self_outer.assertEqual(public['answer'],'');self_outer.assertEqual(set(paid),{'','repair'})
                return 256,2,256,.435
        self_outer=self
        model=SimpleNamespace(dynamics={'physics':Dynamics()},classifier=Observer(),
            predicted=lambda *args:np.eye(16)[1],kernel=new.Kernel(np.asarray([.3,.7,.97]),np.full(3,.5)))
        old=new.core.evaluate(row,tree,{},model,{'penalty':.1},'A')
        actual=new.evaluate_science(row,tree,{},model,{'penalty':.1},'A')
        self.assertEqual(actual,old);self.assertEqual(actual['solver_calls'],3)


if __name__=='__main__':unittest.main()
