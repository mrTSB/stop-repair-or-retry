import unittest
import json
from types import SimpleNamespace

import numpy as np

from difficulty_pomdp import (Dynamics, Planner, MissingObservation, N_OBS, REWARD, REVIEW_WIDTH,
                              arithmetic, evaluate, visible)


def dynamics():
    transition=np.zeros((2,2,4,4))
    for h in (0,1):
        for s in (0,1):
            for a in (0,1):
                p=((.8,.98) if h==0 else (.35,.85))[s]-.1*a
                transition[:,a,2*h+s,2*h]=1-p
                transition[:,a,2*h+s,2*h+1]=p
    emission=np.full((2,2,4,N_OBS),1/N_OBS)
    return Dynamics(np.array([.3,.2,.3,.2]),np.full((4,N_OBS),1/N_OBS),transition,
                    emission,np.full((2,2,4),300.),
                    np.full((N_OBS,4,REVIEW_WIDTH),1/REVIEW_WIDTH),
                    np.repeat(transition[:,None],REVIEW_WIDTH,axis=1),
                    emission.copy(),np.full((2,2,4),300.),np.full((2,4),200.),
                    np.full((N_OBS,4,3),1/3))


class ProbabilityAndPlanning(unittest.TestCase):
    def test_bayes_updates_persistent_difficulty_as_well_as_correctness(self):
        d=dynamics()
        actual=d.update(d.prior,np.array([.1,.2,.3,.4]))
        expected=np.array([.03,.04,.09,.08])/.24
        np.testing.assert_allclose(actual,expected)
        self.assertGreater(actual[2:].sum(),d.prior[2:].sum())

    def test_one_call_bellman_values_match_direct_expectation(self):
        d=dynamics();b=d.prior;penalty=.1
        actual=d.values(b,1,penalty)
        expected=[b@REWARD,*[b@d.transition[1,a]@REWARD-.03 for a in (0,1)]]
        np.testing.assert_allclose(actual,expected)

    def test_two_call_backup_matches_explicit_observation_enumeration(self):
        d=dynamics();b=d.prior;penalty=.1
        expected=[b@REWARD]
        for a in (0,1):
            p=b@d.transition[0,a];future=0.
            for o in range(N_OBS):
                mass=p*d.emission[0,a,:,o];prob=mass.sum();posterior=mass/prob
                future+=prob*max(posterior@REWARD,*[
                    posterior@d.transition[1,k]@REWARD-.03 for k in (0,1)])
            expected.append(future-.03)
        np.testing.assert_allclose(d.values(b,2,penalty),expected)

    def test_bellman_backup_is_homogeneous_for_review_branch_masses(self):
        d=dynamics()
        np.testing.assert_allclose(d.values(.17*d.prior,2,.1),.17*d.values(d.prior,2,.1))

    def test_uninformative_review_loses_its_full_cost(self):
        d=dynamics()
        value=d.information_value(d.prior,2,.1,2,False)-d.values(d.prior,2,.1).max()
        self.assertAlmostEqual(value,-.02)

    def test_future_query_option_never_reduces_zero_threshold_value(self):
        d=dynamics()
        ordinary=d.values(d.prior,2,.1)
        planned=d.values(d.prior,2,.1,future_reviews=True)
        self.assertTrue(np.all(planned>=ordinary-1e-12))
        # The uninformative reviewer is strictly costly and should be ignored.
        np.testing.assert_allclose(ordinary,planned)

    def test_nonmyopic_bellman_matches_explicit_future_review_enumeration(self):
        d=dynamics()
        d.review[:]=0
        for s in range(4):d.review[:,s,s]=1
        belief=d.prior;penalty=.1;expected=[belief@REWARD]
        for a in (0,1):
            predicted=belief@d.transition[0,a];future=0.
            for o in range(N_OBS):
                mass=predicted*d.emission[0,a,:,o];prob=mass.sum();posterior=mass/prob
                regular=d.values(posterior,1,penalty).max()
                queried=0.
                for signal in range(REVIEW_WIDTH):
                    branch=posterior*d.review[o,:,signal]
                    queried+=d.values(branch,1,penalty,d.reviewed_transition[1,signal],True).max()
                queried-=.02
                future+=prob*max(regular,queried)
            expected.append(future-.03)
        np.testing.assert_allclose(d.values(belief,2,penalty,future_reviews=True),expected)

    def test_arithmetic_checker_rejects_code_and_large_exponents(self):
        self.assertEqual(arithmetic('(7+2)*3/9'),3)
        for expression in ("__import__('os')",'2**100','x+1','1/0'):
            with self.assertRaises((ValueError,ZeroDivisionError)):
                arithmetic(expression)


class RolesAndHiddenEvidence(unittest.TestCase):
    def setUp(self):
        self.row={'id':'unit','domain':'math','subject':'math','question':'What is 1+0?','answer':'1'}
        self.tree={p:{'answer':'1' if p else '0','text':'1+0=1\nFinal: 1' if p else 'Final: 0',
                      'input_tokens':100,'output_tokens':20,'finish':'stop','confidence':None,
                      'prompt_sha256':p,'model_digest':'unit'} for p in
                   ('','repair','retry','guided','repair/repair','repair/retry',
                    'retry/repair','retry/retry','guided/repair','guided/retry')}
        self.judges={('unit',p):{'parse_valid':True,'action':'repair','p_correct':.8,
                               'answer':'1','input_tokens':200,'output_tokens':50}
                     for p in ('','repair','retry','guided')}

    def planner(self):
        def observed(row,path,tree):
            if path not in tree:raise MissingObservation('solver',path)
            return 3
        return SimpleNamespace(dynamics={'math':dynamics()},observed=observed)

    def test_a_cannot_read_an_llm_observation(self):
        class NoLLM(dict):
            def __getitem__(self,key):raise AssertionError('A read the LLM')
        result=evaluate(self.row,self.tree,NoLLM(),self.planner(),{'penalty':10.},'A')
        self.assertEqual(result['judge_calls'],0)

    def test_c_is_blind_to_reference_and_charges_all_paid_nodes(self):
        settings={'penalty':.1,'threshold':-100.}
        first=evaluate(self.row,self.tree,self.judges,self.planner(),settings,'C')
        second=evaluate({**self.row,'answer':'999'},self.tree,self.judges,self.planner(),settings,'C')
        first.pop('correct');second.pop('correct')
        self.assertEqual(first,second)
        self.assertGreater(first['judge_calls'],0)
        self.assertLessEqual(first['solver_calls'],3)
        self.assertAlmostEqual(first['compute'],(4*120*first['solver_calls']+3*250*first['judge_calls'])/3.09)

    def test_features_ignore_reference_and_unvisited_branches(self):
        a=visible(self.row,'repair',self.tree)
        pruned={p:self.tree[p] for p in ('','repair')}
        b=visible({**self.row,'answer':'999'},'repair',pruned)
        self.assertEqual(a,b)

    def test_full_hybrid_decision_trace_is_json_serializable(self):
        result=evaluate(self.row,self.tree,self.judges,self.planner(),
                        {'penalty':.1,'threshold':-100.,'plan_reviews':True},'C')
        self.assertEqual(json.loads(json.dumps(result)),result)

    def test_neural_observer_is_not_run_after_forced_terminal_submission(self):
        class Observer:
            def __init__(self):self.calls=[]
            def row_probabilities(self,row,path,tree):
                self.calls.append(path)
                if '/' in path:raise AssertionError('Unnecessary terminal verifier pass')
                return np.array([.3,.2,.3,.2])
        observer=Observer();model=Planner(observer,{'math':dynamics()},{})
        result=evaluate(self.row,self.tree,self.judges,model,{'penalty':0.},'A')
        self.assertEqual(result['solver_calls'],3)
        self.assertEqual(observer.calls,['','repair'])


if __name__=='__main__':
    unittest.main()
