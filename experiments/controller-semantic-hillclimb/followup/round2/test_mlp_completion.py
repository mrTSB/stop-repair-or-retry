import unittest
import numpy as np
from threadpoolctl import threadpool_limits

import mlp_completion_forecast as head
import mlp_policy_crossfit as driver


class SmallHeadTests(unittest.TestCase):
    def test_missing_joint_states_keep_positive_normalized_probabilities(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(48,8));y=np.array([0,7,15]*16)
        with threadpool_limits(limits=1):model=head.fit(x,y)
        p=model.predict_proba(x[:5]);self.assertEqual(p.shape,(5,16))
        np.testing.assert_allclose(p.sum(1),1.);self.assertTrue((p>0).all())
        self.assertFalse(model.estimator[-1].early_stopping)
        constant=head.fit(x,np.zeros(48,dtype=int));np.testing.assert_allclose(constant.predict_proba(x[:2]).sum(1),1.)

    def test_fitting_adapter_restores_forest_after_error(self):
        old=driver.source.forecast.fit
        with self.assertRaises(RuntimeError):
            with driver.head_fitter():
                self.assertIs(driver.source.forecast.fit,head.fit);raise RuntimeError('synthetic')
        self.assertIs(driver.source.forecast.fit,old)


if __name__=='__main__':unittest.main()
