import unittest
from types import SimpleNamespace
import numpy as np
import bayesian_model_hybrid as policy


class BayesianModelTests(unittest.TestCase):
    def test_observation_updates_model_mass_not_each_component_separately(self):
        prior = np.zeros((2,8));prior[:,0] = [.5,.5]
        likelihood = np.ones((2,8));likelihood[:,0] = [.9,.1]
        result = policy.posterior(prior,likelihood)
        np.testing.assert_allclose(result.sum(1),[.9,.1])
        self.assertEqual(result.sum(),1.)

    def test_impossible_and_negative_observations_fail(self):
        for values in (np.zeros((2,8)),-np.ones((2,8)),np.full((2,8),np.nan)):
            with self.assertRaises(ValueError): policy.normalized(values)

    def test_bellman_action_is_shared_across_unobserved_models(self):
        class Dynamics:
            reviewed_transition = np.broadcast_to(np.eye(8),(1,2,8,8))
            emission = np.ones((2,8,1))
            first_costs = np.zeros((2,8))
            def __init__(self,value):self.value = value
            def alpha(self,first,penalty):
                return np.repeat((self.value if first == 0 else np.array([.4,.1,.2]))[:,None],8,axis=1)
        models = [Dynamics(np.array([.1,.9,.2])),Dynamics(np.array([.8,.2,.3]))]
        belief = np.zeros((2,8));belief[:,0] = [.5,.5]
        np.testing.assert_allclose(policy.values(models,belief,0,0),[0.,.55,.4])
        self.assertAlmostEqual(policy.values(models,belief,0,0).max(),.55)

    def test_query_value_preserves_observation_mass_and_price(self):
        class Dynamics:
            review = np.full((8,2),.5)
            review_costs = np.full(8,100.)
            reviewed_transition = np.broadcast_to(np.eye(8),(2,2,8,8))
            emission = np.ones((2,8,1))
            first_costs = np.zeros((2,8))
            def alpha(self,first,penalty):return np.repeat(np.array([.8,.1,.2])[:,None],8,axis=1)
        belief = np.full((2,8),1/16)
        self.assertAlmostEqual(policy.query_value([Dynamics(),Dynamics()],belief,1.),.7)


if __name__ == '__main__': unittest.main()
