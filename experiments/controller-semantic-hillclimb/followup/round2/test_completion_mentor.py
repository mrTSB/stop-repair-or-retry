import copy
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import completion_mentor_hybrid as hybrid
import completion_mentor_transfer as transfer


def one(index,n=16):
    result=np.zeros(n); result[index]=1.; return result


def node(answer='B',tokens=10):
    return {'answer':answer,'text':answer,'input_tokens':tokens,'output_tokens':1,'prompt_sha256':str(tokens)}


def fixture():
    rows=[{'id':f'q{i}','subject':'chemistry' if i<128 else 'physics','domain':'science','answer':'A'} for i in range(256)]
    complete, strong, judges = {}, {}, {}
    for i,row in enumerate(rows):
        tree={p:node('A' if (i+len(p))%3==0 else 'B') for p in transfer.PLAIN_PATHS}
        complete[row['id']]={**copy.deepcopy(tree),**{p:node('A' if i%4==0 else 'B') for p in transfer.COMPLETE_PATHS}}
        strong[row['id']]={**copy.deepcopy(tree),**{p:node('A' if i%5==0 else 'B') for p in transfer.STRONG_PATHS}}
        strong[row['id']]['guided/retry']=copy.deepcopy(tree['repair/retry'])
        judges[row['id'],'']=node('A' if i%2 else 'B')
    return rows,complete,strong,judges


class FakeObserver:
    def __init__(self):
        self._recipe={'parameters':400_000_000}; self.temperature=1.; self.seen=[]
    def load(self): pass
    def entry(self,row,path,tree,judge=None):
        assert row['answer']==''
        assert set(tree)==set(['',path])
        return {'question':row['id'],'path':path,'text':tree[path]['text'],'review':judge is not None}
    def values(self,row,path,tree,judge=None):
        self.entry(row,path,tree,judge); self.seen.append((path,judge is not None))
        return np.zeros(4),np.zeros(8),np.zeros(8),10
    def charges(self,row,tree,judges,decisions,include_reviews):
        assert row['answer']=='' and not include_reviews
        return 10*len(decisions),len(decisions),10*len(decisions),.4


class OtherObserver(FakeObserver): pass


class TransferTests(unittest.TestCase):
    def test_builder_binds_parent_states_and_preserves_components_through_reload(self):
        rows,complete,strong,judges=fixture()
        records,_,receipt=transfer.join(rows,complete,rows,strong,judges)
        matrices,receipt=transfer.fit(records,receipt)
        observer=FakeObserver()
        science=hybrid.plain.Planner(observer,None,{s:SimpleNamespace() for s in hybrid.SCIENCE},
            {'state_count':16,'state_bits':hybrid.SOURCE_BITS,'complete_is_terminal':True})
        for count in (128,192,320):
            mentor_receipt={'state_count':8,'maximum_logical_reviews':1,'paired_root_questions':count}
            if count==320: mentor_receipt['fit_module']='expanded_mentor_core'
            else: mentor_receipt['hidden_states']=hybrid.TARGET_STATES
            mentor=hybrid.mentor_core.Planner(observer,{'science':np.arange(8)},mentor_receipt,
                SimpleNamespace(classifier=observer),review_observer=observer)
            parent=hybrid.retained.Planner(SimpleNamespace(classifier=observer),mentor,{'old':np.eye(8)},{})
            with patch.object(hybrid.assets,'asset_id',return_value='same-recipe'):
                model=hybrid.build(science,{'penalty':.08},parent,'hybrid_repairability',{'penalty':.04},
                    matrices,receipt,{'selection':'synthetic'})
            self.assertIs(model.mentor,mentor);self.assertIs(model.math_C,parent)
            restored=pickle.loads(pickle.dumps(model))
            np.testing.assert_array_equal(restored.mentor.dynamics['science'],mentor.dynamics['science'])
            self.assertEqual(restored.fit_receipt['states_per_mode'],[16,8])
            self.assertEqual(restored.fit_receipt['parent_semantics']['parent_fit_receipt'],mentor_receipt)
            self.assertIs(restored.mentor.classifier,restored.mentor.regular.classifier)
        changed=copy.deepcopy(receipt);changed['hardness']['supported_parent_functions']={}
        with self.assertRaises(ValueError): hybrid.parent_semantics(mentor,changed)

    def test_full_cohort_join_exact_parent_hardness_and_truth_support(self):
        rows,complete,strong,judges=fixture()
        records,joined,receipt=transfer.join(rows,complete,rows,strong,judges)
        matrices,fitted=transfer.fit(records,receipt)
        self.assertEqual(fitted['smoothing'],32.)
        for row,record in zip(rows,records):
            expected=int(np.mean([transfer.mentor.correct(strong[row['id']][p],row['answer'])
                                   for p in transfer.mentor.core.FIXED])<.6)
            self.assertEqual(record['hard'],expected)
            self.assertEqual(record['target']//4,expected)
        hybrid.validate_transfer(matrices)
        for matrix in matrices.values():
            np.testing.assert_allclose(matrix@transfer.mentor.REWARD,transfer.plain.CURRENT,atol=1e-14)
        self.assertFalse(fitted['mentor_dynamics_refitted'])
        self.assertEqual(len(joined),256)

    def test_all_questions_required_and_plain_or_alias_tamper_rejected(self):
        rows,complete,strong,judges=fixture()
        with self.assertRaises(ValueError): transfer.join(rows[:-1],complete,rows,strong,judges)
        changed=copy.deepcopy(strong);changed['q0']['']['text']='changed'
        with self.assertRaises(ValueError): transfer.join(rows,complete,rows,changed,judges)
        changed=copy.deepcopy(strong);changed['q0']['guided/retry']=node('changed')
        with self.assertRaises(ValueError): transfer.join(rows,complete,rows,changed,judges)
        changed=copy.deepcopy(rows);changed[0]['answer']='D'
        with self.assertRaises(ValueError): transfer.join(rows,complete,changed,strong,judges)

    def test_no_8x8_reshape_or_forbidden_current_mass(self):
        with self.assertRaises(ValueError): hybrid.validate_transfer({s:np.eye(8) for s in hybrid.SCIENCE})
        matrices={s:np.ones((16,8))/8 for s in hybrid.SCIENCE}
        with self.assertRaises(ValueError): hybrid.validate_transfer(matrices)
        rows,complete,strong,judges=fixture()
        records,_,receipt=transfer.join(rows,complete,rows,strong,judges)
        receipt['smoothing']=8.
        with self.assertRaises(ValueError): transfer.fit(records,receipt)

    def test_hindsight_counts_entire_panel_without_learnability_claim(self):
        rows,complete,strong,judges=fixture()
        _,joined,_=transfer.join(rows,complete,rows,strong,judges)
        result=transfer.availability(rows,joined)
        all_science=result['panels']['all_science']
        expected=sum(any(transfer.mentor.correct(n,row['answer']) for n in joined[row['id']].values()) for row in rows)
        self.assertEqual(all_science['hindsight_available_correct']['combined13_distinct'],expected)
        self.assertEqual(result['distinct_solver_responses'],13)
        self.assertIn('neither controller accuracy',result['interpretation'])


class RuntimeTests(unittest.TestCase):
    def model(self,query,route=(3,),semantic=False):
        observer=FakeObserver(); teacher=FakeObserver(); current=one(0)
        class PlainDynamics:
            transition=np.tile(np.eye(16),(2,1,1)); emission=np.ones((2,16,30))/30
            def values(self,belief,remaining,penalty): return np.eye(4)[route[2-remaining]]
        science=SimpleNamespace(classifier=observer,dynamics={'physics':PlainDynamics()},
            predicted=lambda row,path,tree: (observer.values(row,path,{p:tree[p] for p in ('',path)}),current)[1])
        class MentorDynamics:
            review=np.ones((8,32))/32
            reviewed_transition=np.tile(np.eye(8),(32,2,1,1))
            emission=np.ones((2,8,300))/300
            def information_value(self,belief,penalty): return 2. if query else 0.
            def values(self,belief,signal,penalty): return np.array([0.,1.,0.])
            def alpha(self,first,penalty): return np.array([np.zeros(8),np.ones(8),np.zeros(8)])
        regular=SimpleNamespace(observed=lambda row,path,tree:(teacher.values(row,path,tree),0)[1])
        mentor=SimpleNamespace(classifier=teacher,regular=regular,dynamics={'physics':MentorDynamics()},
                               review_observer=teacher if semantic else None)
        support=hybrid.plain.CURRENT[:,None]==hybrid.mentor_core.REWARD[None,:]
        matrix=support/support.sum(1,keepdims=True)
        model=hybrid.Planner(None,'hybrid_repairability',{},science,{},mentor,{'physics':matrix},{})
        model.prepare_observer=lambda _:None
        return model

    def run_route(self,query,route=(3,),semantic=False,answer='A'):
        row={'id':'q','subject':'physics','domain':'science','answer':answer}
        tree={p:node('A' if p.endswith('complete') or p.endswith('reuse') else 'B',20+i)
              for i,p in enumerate(['','complete','repair','repair/complete','guided','guided/reuse'])}
        calls=[{'input_tokens':30,'output_tokens':4},{'input_tokens':40,'output_tokens':5}]
        judge={**node('A'), 'input_tokens':70,'output_tokens':9,'calls':calls,'physical_api_calls':2,'parse_valid':True}
        model=self.model(query,route,semantic)
        with patch.object(hybrid.assets,'asset_id',return_value='recipe'),patch.object(hybrid.mentor_core,'signal',return_value=0):
            result=hybrid.evaluate(row,tree,{('q',''):judge},model,{'penalty':0.,'threshold':0.},'C')
        return result,model,tree

    def test_unqueried_complete_is_terminal_and_root_forward_charged_once(self):
        result,model,tree=self.run_route(False)
        self.assertEqual(result['path'],'complete');self.assertEqual(result['completion_calls'],1)
        self.assertEqual(result['judge_calls'],0);self.assertEqual(result['judge_physical_calls'],0)
        self.assertEqual(result['solver_calls'],2);self.assertEqual(result['verifier_calls'],1)
        expected=(4*sum(tree[p]['input_tokens']+1 for p in ('','complete'))+.4*10)/3.09
        self.assertAlmostEqual(result['compute'],expected)
        self.assertFalse(model.mentor.classifier.seen)

    def test_plain_repair_then_complete_never_observes_terminal(self):
        result,model,_=self.run_route(False,(1,3))
        self.assertEqual(result['path'],'repair/complete');self.assertEqual(result['solver_calls'],3)
        self.assertEqual(result['verifier_calls'],2)
        self.assertEqual({p for p,_ in model.science.classifier.seen},{'','repair'})

    def test_query_retains_old_paths_and_bills_both_review_phases(self):
        result,model,tree=self.run_route(True,semantic=True)
        self.assertEqual(result['path'],'guided/reuse');self.assertEqual(result['completion_calls'],0)
        self.assertEqual(result['judge_calls'],1);self.assertEqual(result['judge_physical_calls'],2)
        self.assertEqual(result['solver_calls'],3);self.assertEqual(result['verifier_calls'],3)
        self.assertEqual(result['judge_tokens'],79)
        expected=(4*sum(tree[p]['input_tokens']+1 for p in ('','guided','guided/reuse'))+3*79+.4*30)/3.09
        self.assertAlmostEqual(result['compute'],expected)
        self.assertEqual({p for p,_ in model.mentor.classifier.seen},{'','guided'})
        altered,_,_=self.run_route(True,semantic=True,answer='D')
        self.assertEqual(result['decisions'],altered['decisions']);self.assertEqual(result['compute'],altered['compute'])
        self.assertNotEqual(result['correct'],altered['correct'])

    def test_reference_blank_and_actual_distinct_adapter_billing(self):
        ledger={};row={'id':'q','answer':'hidden'};tree={'':node()}
        with patch.object(hybrid.assets,'asset_id',return_value='same-weights'):
            a,b=FakeObserver(),OtherObserver()
            hybrid.record_observation(ledger,a,row,'',tree)
            hybrid.record_observation(ledger,a,row,'',tree)
            self.assertEqual(len(ledger),1)
            hybrid.record_observation(ledger,b,row,'',tree)
            self.assertEqual(len(ledger),2)

    def test_review_phase_count_and_token_mismatch_fail_closed(self):
        with self.assertRaises(ValueError): hybrid.physical_calls({'calls':[{}], 'physical_api_calls':2})
        with self.assertRaises(ValueError): hybrid.physical_calls({'calls':[{'input_tokens':5,'output_tokens':1}],
            'input_tokens':4,'output_tokens':1,'physical_api_calls':1})

    def test_math_exact_old_route_and_settings(self):
        row={'id':'q','subject':'math','answer':'A'};tree={'':node('A')}
        model=SimpleNamespace(math_C=object(),math_policy_module='hybrid_repairability',math_settings={'penalty':.08})
        def unchanged(safe,paid,judges,old,settings,family):
            self.assertEqual(safe['answer'],'');self.assertIs(old,model.math_C)
            self.assertEqual(settings,model.math_settings);self.assertEqual(family,'C')
            return {'path':'','decisions':[{'path':'','queried':False}], 'compute':123.,'correct':0}
        with patch.object(hybrid.retained,'evaluate',side_effect=unchanged):
            result=hybrid.evaluate(row,tree,{},model,{'penalty':999},'C')
        self.assertEqual(result['compute'],123.);self.assertEqual(result['correct'],1)
        self.assertTrue(result['math_C_route_retained'])


if __name__=='__main__': unittest.main()
