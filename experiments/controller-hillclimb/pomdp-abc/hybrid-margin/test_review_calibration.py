import unittest

import numpy as np

import test_difficulty_pomdp as fixtures
from review_calibration import encode,features,Observer
from difficulty_pomdp import Planner,evaluate


class LearnedReviewChannel(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.RolesAndHiddenEvidence('test_a_cannot_read_an_llm_observation')
        self.f.setUp()
        self.judge={**self.f.judges['unit',''],'prompt_sha256':'review','text':'review proof',
                    'verdict':{'solution':'1+0=1'},'finish':'stop'}

    def test_channel_uses_calibrated_probability_instead_of_raw_confidence(self):
        current=self.f.tree['']
        self.assertNotEqual(encode(np.array([.95,.05,0.,0.]),self.judge,current),
                            encode(np.array([.01,.99,0.,0.]),self.judge,current))
        self.assertEqual(encode(np.array([.1,.9,0.,0.]),{**self.judge,'parse_valid':False},current),30)

    def test_review_features_ignore_reference_and_unpaid_continuations(self):
        a=features(self.f.row,'',self.f.tree,self.judge)
        b=features({**self.f.row,'answer':'unavailable'},'',{'':self.f.tree['']},self.judge)
        self.assertEqual(a,b)

    def test_a_never_acquires_the_learned_review_channel(self):
        class Forbidden:
            def signal(self,*a):raise AssertionError('A inspected a review')
        f=self.f;model=f.planner()
        model.review_signal=Forbidden().signal
        self.assertEqual(evaluate(f.row,f.tree,{},model,{'penalty':0.},'A')['judge_calls'],0)


if __name__=='__main__':unittest.main()
