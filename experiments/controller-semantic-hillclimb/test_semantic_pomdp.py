import unittest
from types import SimpleNamespace

import numpy as np

from semantic_pomdp import Planner, Observer
from difficulty_pomdp import normalize


class ContinuousObservation(unittest.TestCase):
    def test_likelihood_recovers_calibrated_posterior_from_its_training_prior(self):
        posterior=np.array([.1,.3,.2,.4])
        valid_prior=np.array([.2,.2,.3,.3]);validity=np.array([.4,1.,.6,1.])
        classifier=SimpleNamespace(row_probabilities=lambda *a:posterior)
        model=Planner(classifier,{}, {},observation_cohorts={('physics',0):(valid_prior,validity)})
        prior=normalize(valid_prior/validity)
        likelihood=model.likelihood({'subject':'physics'},'',{'':{'answer':'A'}})
        np.testing.assert_allclose(normalize(prior*likelihood),posterior)

    def test_missing_answer_cannot_have_positive_correctness_belief(self):
        model=Planner(None,{}, {},observation_cohorts={('math',0):(
            np.array([.2,.2,.3,.3]),np.array([.4,1.,.6,1.]))})
        likelihood=model.likelihood({'subject':'math'},'',{'':{'answer':''}})
        posterior=normalize(np.ones(4)*likelihood)
        self.assertEqual(posterior[1]+posterior[3],0.)

    def test_semantic_transitions_preserve_hidden_difficulty_and_unit_mass(self):
        transitions=np.zeros((2,2,4,4))
        for h in (0,1):transitions[:,:,2*h:2*h+2,2*h:2*h+2]=.5
        d=SimpleNamespace(transition=transitions,reviewed_transition=np.repeat(transitions[:,None],31,axis=1))
        classifier=SimpleNamespace(transitions=lambda *a:np.array([[.1,.9,.3,.7],[.2,.8,.4,.6]]))
        model=Planner(classifier,{'math':d},{})
        row={'id':'x','subject':'math'};tree={'':{'prompt_sha256':'x','text':'text'}}
        for reviewed in (False,True):
            actual=model.context(row,'',tree,0,reviewed,.8)
            np.testing.assert_allclose(actual.sum(axis=-1),1.)
            if not reviewed:actual=actual[None]
            self.assertTrue((actual[:,:,:2,2:]==0).all())
            self.assertTrue((actual[:,:,2:,:2]==0).all())

    def test_every_checkpoint_and_paid_review_pass_is_charged(self):
        observer=Observer('unused',1.)
        observer.load=lambda:None
        observer._recipe={'max_length':384,'parameters':184_000_000}
        observer.values=lambda row,path,tree,judge=None:(None,None,None,90 if judge is None else 120)
        row={'id':'x'};decisions=[{'path':'','queried':False},{'path':'repair','queried':True}]
        tokens,calls,slots,parameters=observer.charges(row,{}, {('x','repair'):{}},decisions)
        self.assertEqual((tokens,calls,slots),(300,3,1152))
        self.assertEqual(parameters,.184)
        self.assertEqual(observer.charges(row,{}, {},decisions,include_reviews=False)[:3],(180,2,768))


if __name__=='__main__':unittest.main()
