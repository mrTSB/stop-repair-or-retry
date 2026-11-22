import copy
import unittest
import numpy as np
from threadpoolctl import threadpool_limits

import factorized_completion_forecast as forecast
import test_nested_dense_diagnostic as fixtures


class FactorizedBeliefTests(unittest.TestCase):
    def test_sixteen_state_distribution_preserves_four_marginals(self):
        rng=np.random.default_rng(23);x=rng.normal(size=(96,5));y=np.arange(96)%16
        with threadpool_limits(limits=1):model=forecast.fit(x,y,.1)
        p=model.predict_proba(x);z=model.scaler.transform(x)
        q=np.stack([head.predict_proba(z)[:,list(head.classes_).index(1)] for head in model.heads],axis=1)
        np.testing.assert_allclose(p.sum(1),1.);np.testing.assert_allclose(p@forecast.BITS,q)
        self.assertEqual(p.shape,(96,16));self.assertTrue(np.isfinite(p).all())
        with threadpool_limits(limits=1):constant=forecast.fit(x,np.zeros(96,dtype=int),.1)
        self.assertTrue(np.all(constant.predict_proba(x)[:,0]>.999999))

    def test_outer_held_labels_and_features_cannot_change_selected_model(self):
        data=fixtures.fixture();family='raw20_control'
        held=np.asarray([r['fold']==0 for r in data['records']]);changed=copy.deepcopy(data)
        changed['labels'][held]=(changed['labels'][held]+7)%16;changed['features'][family][held]+=100
        with threadpool_limits(limits=1):
            mask,p,model,receipt=forecast.outer_prediction(data,family,0)
            mask2,_,other,receipt2=forecast.outer_prediction(changed,family,0)
        np.testing.assert_array_equal(mask,mask2);self.assertEqual(receipt['selected_C'],receipt2['selected_C'])
        self.assertEqual(receipt['inner_search'],receipt2['inner_search'])
        np.testing.assert_array_equal(model.scaler.mean_,other.scaler.mean_)
        for a,b in zip(model.heads,other.heads):
            if hasattr(a,'coef_'):np.testing.assert_array_equal(a.coef_,b.coef_)
        self.assertTrue(np.isfinite(p).all());self.assertEqual(len(p),int(held.sum()))


if __name__=='__main__':unittest.main()
