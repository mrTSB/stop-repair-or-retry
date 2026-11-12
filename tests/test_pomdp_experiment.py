"""Independent analytic checks of belief planning; no inference or test data."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest import mock
from stop_repair_retry import pomdp_experiment as p


def model():
    return {'states':['incorrect','correct'], 'observations':['bad','good'],
            'prior':[.5,.5], 'terminal_reward':[0,1],
            'initial_observation':[[.8,.2],[.2,.8]],
            'transition':{'repair':[[1,0],[0,1]],'retry':[[.25,.75],[.25,.75]]},
            'observation':{'repair':[[1,0],[0,1]],'retry':[[.5,.5],[.5,.5]]},
            'cost_tokens':{'repair':[50,50],'retry':[300,300]}}


class PlannerTests(unittest.TestCase):
    def test_initial_observation_and_action_update_are_distinct(self):
        m=model();p.validate_model(m)
        self.assertEqual(p.initial_belief(m,1),[.2,.8])
        self.assertEqual(p.update(m,[.9,.1],'repair',1),[0,1])
        self.assertEqual(p.update(m,[.9,.1],'retry',1),[.25,.75])

    def test_two_step_value_of_information(self):
        # One repair reveals correctness without changing it; after a bad answer
        # only, retry offers .75 correctness for .30 reward cost. Thus the
        # two-step repair value is -.05 + .5*1 + .5*(.75-.30) = .675.
        m=model()
        self.assertEqual(p.plan(m,[.5,.5],1,1)['action'],'stop')
        result=p.plan(m,[.5,.5],2,1)
        self.assertEqual(result['action'],'repair')
        self.assertAlmostEqual(result['value'],.675)
        self.assertAlmostEqual(result['action_values']['retry'],.45)

    def test_zero_horizon_and_tie_precedence(self):
        m=model()
        self.assertEqual(p.plan(m,[.2,.8],0,1),{'action':'stop','value':.8,'action_values':{'stop':.8}})
        m['transition']={a:[[1,0],[0,1]] for a in p.ACTIONS}
        m['cost_tokens']={a:[0,0] for a in p.ACTIONS}
        self.assertEqual(p.plan(m,[.5,.5],2,0)['action'],'stop')
        m['transition']={a:[[0,1],[0,1]] for a in p.ACTIONS}
        self.assertEqual(p.plan(m,[1,0],1,0)['action'],'repair')

    def test_cost_conditioning_and_units(self):
        m=model();m['cost_tokens']['retry']=[100,900]
        result=p.plan(m,[.8,.2],1,.1)
        self.assertAlmostEqual(result['action_values']['retry'],.75-.1*(.8*100+.2*900)/1000)

    def test_four_state_planning_interface(self):
        m={'states':list('abcd'),'observations':['only'],'prior':[.25]*4,
           'terminal_reward':[0,0,0,1],'initial_observation':[[1]]*4,
           'transition':{a:[[0,0,0,1]]*4 for a in p.ACTIONS},
           'observation':{a:[[1]]*4 for a in p.ACTIONS},
           'cost_tokens':{a:[10]*4 for a in p.ACTIONS}}
        p.validate_model(m)
        result=p.plan(m,[.25]*4,2,1)
        self.assertEqual(result['action'],'repair');self.assertAlmostEqual(result['value'],.99)

    def test_invalid_models_and_impossible_observation(self):
        m=model();m['transition']['repair'][0]=[.8,.8]
        with self.assertRaises(ValueError):p.validate_model(m)
        m=model();m['observation']['repair']=[[1,0],[1,0]]
        with self.assertRaises(ValueError):p.update(m,[.5,.5],'repair',1)
        # Planning skips impossible observations instead of conditioning on them.
        self.assertTrue(p.plan(m,[.5,.5],2,1)['value']>=.5)

    def test_policy_does_not_access_gold_or_unvisited_nodes(self):
        class Node(dict):
            def __getitem__(self,key):
                if key not in ('confidence','answer_parse_valid'):
                    raise AssertionError('Policy accessed forbidden field '+key)
                return super().__getitem__(key)
        self.assertEqual(p.observation(Node(confidence='high',answer_parse_valid=True)),5)
        self.assertEqual(p.observation(Node(confidence=None,answer_parse_valid=False)),6)

    def test_root_cost_is_sunk_but_evaluation_charges_it(self):
        m=model()
        paths=('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
        row={'id':'synthetic','domain':'math','tree':{path:{'confidence':'low',
             'answer_parse_valid':False,'correct':False,'input_tokens':10,'output_tokens':5} for path in paths}}
        first=p.replay(m,[row],1)[0]
        row['tree']['']['input_tokens']+=10000
        # Gold labels can change scores, but never actions or beliefs.
        for node in row['tree'].values():node['correct']=True
        second=p.replay(m,[row],1)[0]
        self.assertEqual(first['steps'],second['steps'])
        self.assertEqual(second['solver_tokens']-first['solver_tokens'],10000)
        self.assertEqual(first['controller_tokens'],0)

    def test_status_only_development_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);(run/'dev').mkdir()
            p.io.save(run/'dev/status.json',{'status':'complete','split':'dev','questions':60,'finished_utc':'now'})
            with mock.patch.object(p,'check_run',return_value=model()), mock.patch.object(p.io,'load_dataset',return_value=[{}]*60):
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    p.validate_phase(run,'dev')

    def test_fit_train_only_and_exact_smoothed_counts(self):
        tree={path:{'correct':path!='','confidence':'high','answer_parse_valid':True,
                    'input_tokens':7,'output_tokens':3}
              for path in ('','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry')}
        rows=[{'id':str(i),'split':'train','tree':copy.deepcopy(tree)} for i in range(100)]
        m=p.fit_model(rows)
        self.assertEqual(m['training_counts']['roots'],[100,0])
        self.assertAlmostEqual(m['prior'][1],1/102)
        for action in p.ACTIONS:
            self.assertEqual(m['training_counts']['transitions'][action],[[0,100],[0,200]])
            self.assertAlmostEqual(m['transition'][action][0][1],101/102)
            self.assertEqual(m['cost_tokens'][action],[10,10])
        rows[0]['split']='dev'
        with self.assertRaises(ValueError):p.fit_model(rows)


if __name__=='__main__':unittest.main()
