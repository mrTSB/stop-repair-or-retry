"""Analytic tests of observed-context recursion; no real data or inference."""
import copy
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('context_diagnostic',ROOT/'experiments/pomdp-controller/context_diagnostic.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def model():
    base={'states':['wrong','right'],'observations':['low:invalid','low:valid'],'prior':[1,0],
        'initial_observation':[[1,0],[1,0]],'terminal_reward':[0,1],
        'transition':{a:[[1,0],[0,1]] for a in c.p.ACTIONS},
        'observation':{a:[[.8,.2],[.2,.8]] for a in c.p.ACTIONS},
        'cost_tokens':{a:[100,100] for a in c.p.ACTIONS}}
    contexts={key:copy.deepcopy(base) for key in c.CONTEXTS}
    contexts['initial']['transition']['repair']=[[.5,.5],[0,1]]
    contexts['after_repair']['transition']['retry']=[[0,1],[0,1]]
    contexts['after_repair']['cost_tokens']['retry']=[200,200]
    # Different child-context likelihood: using this before the context switch
    # would invert the posterior and make the saved belief wrong.
    contexts['after_repair']['observation']['repair']=[[.2,.8],[.8,.2]]
    return {'contexts':contexts,'pooled_prior_model':base}


class ContextTests(unittest.TestCase):
    def test_two_step_plan_uses_new_context(self):
        m=model();result=c.plan(m,[1,0],2,1,'initial')
        # After repair, z0 has posterior correctness .2 -> retry worth .8;
        # z1 has posterior .8 -> stop ties retry and wins. Total -.1+.8=.7.
        self.assertEqual(result['action'],'repair')
        self.assertAlmostEqual(result['action_values']['repair'],.7)
        self.assertAlmostEqual(c.plan(m,[1,0],1,1,'initial')['action_values']['repair'],.4)

    def test_replay_conditions_on_parent_likelihood_then_switches_context(self):
        paths=('', 'repair','retry','repair/repair','repair/retry','retry/repair','retry/retry')
        row={'id':'synthetic','domain':'math','tree':{path:{'confidence':'low','answer_parse_valid':False,'correct':False,'input_tokens':1,'output_tokens':1} for path in paths}}
        result=c.replay(model(),[row],1)[0]
        self.assertEqual(result['path'],'repair/retry')
        self.assertEqual(result['steps'][1]['context'],'after_repair')
        self.assertEqual(result['steps'][1]['belief'],[.8,.2])
        self.assertEqual(result['tokens'],6)

if __name__=='__main__':unittest.main()
