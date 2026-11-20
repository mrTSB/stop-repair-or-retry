"""Numerical checks for the Bayesian filter and information acquisition decision."""

import unittest

import numpy as np

from belief import BeliefPlanner, Dynamics, GRID, OBSERVATIONS, features
from gate import Logistic


def dynamics():
    initial = np.full((2, OBSERVATIONS), 1/OBSERVATIONS)
    transition = np.array([[.6,.9],[.45,.8]])
    emission = np.full((2,2,OBSERVATIONS), .01)
    emission[:,0,:3] = .97/3
    emission[:,1,3:] = .97/3
    costs = np.full((2,2), 100.)
    judge = np.zeros((OBSERVATIONS,2,9))
    judge[:,0,0] = .8
    judge[:,0,8] = .2
    judge[:,1,0] = .1
    judge[:,1,8] = .9
    return Dynamics(.6, initial, transition, emission, costs, judge, 50.)


class BeliefTests(unittest.TestCase):
    def test_update_matches_explicit_bayes_rule(self):
        model = dynamics()
        prior = .4
        predicted = .6*.6+.4*.9
        likelihood = model.emission[0,:,5]
        expected = predicted*likelihood[1]/((1-predicted)*likelihood[0]+predicted*likelihood[1])
        self.assertAlmostEqual(model.update(prior,0,5),expected)

    def test_one_step_values_equal_expected_correctness_minus_cost(self):
        model = dynamics()
        planned = model.bellman(.2)
        for belief in (0,.2,.5,.8,1.):
            expected = [.6*(1-belief)+.9*belief-.02,.45*(1-belief)+.8*belief-.02]
            np.testing.assert_allclose(model.values(belief,1,planned)[1:],expected,atol=1e-12)

    def test_zero_remaining_calls_forces_submission_value(self):
        planned = dynamics().bellman(.2)
        np.testing.assert_array_equal(planned[0,0],GRID)
        np.testing.assert_array_equal(planned[0,1:],np.zeros((2,len(GRID))))

    def test_uninformative_judge_has_only_its_acquisition_cost(self):
        model = dynamics()
        model.judge_emission[:] = 1/9
        self.assertAlmostEqual(model.information_value(.5,2,model.bellman(.2),.2),-.01)

    def test_serialization_preserves_filter_and_plan(self):
        original = dynamics()
        expected = original.bellman(.07)
        restored = Dynamics.restore(original.as_dict())
        np.testing.assert_array_equal(restored.bellman(.07),expected)
        self.assertEqual(restored.judge_update(.4,8), original.judge_update(.4,8))

    def test_acquired_context_changes_transition_and_action(self):
        model = dynamics()
        model.judge_transition = np.broadcast_to(model.transition,(9,2,2)).copy()
        model.judge_transition[0] = [[0.,0.],[1.,1.]]
        values = model.judge_values(.2,1,0,model.bellman(.1),.1)
        self.assertEqual(int(values.argmax()),2)
        self.assertAlmostEqual(model.update(.2,1,5,0),1.)
        restored = Dynamics.restore(model.as_dict())
        np.testing.assert_array_equal(restored.judge_transition,model.judge_transition)

    def test_perfect_judge_can_have_positive_value_at_action_boundary(self):
        model = dynamics()
        model.judge_emission[:] = 0
        model.judge_emission[:,0,0] = 1
        model.judge_emission[:,1,8] = 1
        self.assertGreater(model.information_value(.7,1,model.bellman(.1),.1),0)

    def test_acquisition_price_uses_the_current_checkpoint(self):
        model = dynamics()
        model.judge_emission[:] = 1/9
        model.judge_stage_cost = np.array([20.,80.])
        planned = model.bellman(.2)
        self.assertAlmostEqual(model.information_value(.5,2,planned,.2),-.004)
        self.assertAlmostEqual(model.information_value(.5,1,planned,.2),-.016)

    def test_calibrated_initial_belief_survives_serialization(self):
        row = {'id':'unit','domain':'math','subject':'math','question':'What is 2 + 2?'}
        node = {'text':'2 + 2 = 4\nFinal: 4','answer':'4','finish':'stop','input_tokens':90,
                'output_tokens':15,'prompt_sha256':'unit'}
        tree = {'':node}
        width = len(features(row,'',tree))
        coef = np.zeros(width+1)
        coef[0] = np.log(.6/.4)
        classifier = Logistic(np.zeros(width),np.ones(width),coef)
        calibration = Logistic(np.zeros(1),np.ones(1),np.array([np.log(.713/.287),0.]))
        model = BeliefPlanner(classifier,{'math':dynamics()},'linear','calibrated',{'math':calibration})
        self.assertAlmostEqual(model.initial(row,tree),.713)
        restored = BeliefPlanner.restore(model.as_dict())
        self.assertAlmostEqual(restored.initial(row,tree),.713)
        self.assertAlmostEqual(restored.initial({**row,'answer':'hidden'},tree),.713)


if __name__ == '__main__':
    unittest.main()
