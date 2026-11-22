import copy
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import completion_blend_pomdp as policy
import development_completion_blend as dev

core=policy.core


def node(answer,tokens=100):
    return {'answer':answer,'text':answer,'input_tokens':tokens,'output_tokens':1}


def row(qid='q',subject='physics'):
    return {'id':qid,'subject':subject,'domain':'math' if subject=='math' else 'science',
        'answer':'A','question':'Q\n\nOptions:\nA. yes\nB. no\nC. maybe\nD. none'}


class FixedDynamics:
    def __init__(self,route=(1,3)):
        self.route=route
        self.transition=np.tile(np.eye(16),(2,1,1))
        self.emission=np.ones((2,16,core.N_OBS))/core.N_OBS
    def values(self,belief,remaining,penalty):return np.eye(4)[self.route[2-remaining]]


class Observer:
    def values(self,public,path,paid):
        if public['answer'] or set(paid)!={'',path}:raise AssertionError('Unpaid or reference input')
        return (np.zeros(4),np.zeros(8),np.zeros(8),128)
    def charges(self,public,paid,judges,decisions,include_reviews):
        if public['answer'] or judges or include_reviews:raise AssertionError('Review/reference charged')
        if set(paid)!={d['path'] for d in decisions}:raise AssertionError('Unpaid observation charged')
        return 128*len(paid),len(paid),128*len(paid),.435


class Forecaster:
    classes_=np.arange(16)
    def predict_proba(self,x):return np.tile(np.arange(1,17)/136,(len(x),1))


def parent():
    science=core.Planner(Observer(),Forecaster(),{'physics':FixedDynamics(),'chemistry':FixedDynamics()},
        {'state_count':16,'state_bits':['current_correct','repair_correct','retry_correct','complete_correct']})
    return policy.parent_policy.Planner(SimpleNamespace(classifier=Observer()),science,
        {'penalty':.02,'posterior_weight':.5},{'penalty':.02},{'parent':'synthetic'})


def tree():
    return {'':node('B'),'repair':node('',200),'retry':node('B',300),
        'repair/repair':node('B',400),'repair/retry':node('B',500),
        'retry/repair':node('B',600),'retry/retry':node('B',700),
        'complete':node('A',800),'repair/complete':node('A',900),'retry/complete':node('A',1000)}


class BlendTests(unittest.TestCase):
    def test_fresh_test_guard_and_inference_block(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);protected=root/'test-questions.jsonl';protected.write_text('private')
            allowed=root/'dev-questions.jsonl';allowed.write_text('development')
            with dev.no_test_reads():
                self.assertEqual(allowed.read_text(),'development')
                with self.assertRaises(AssertionError):protected.read_bytes()
                with self.assertRaises(AssertionError):open(protected)
            with dev.original.guarded.no_model_calls():
                with self.assertRaises(AssertionError):
                    dev.original.guarded.perception.Observer('never-load',1.).load()

    def test_zero_exact_parent_and_same_path_half_cost_parity(self):
        model=parent().science
        with patch.object(core.plain,'observables',return_value=[0.]):
            original=core.evaluate(row(),tree(),{},model,{'penalty':.02},'A')
            zero=policy.science_evaluate(row(),tree(),model,.02,0.)
            half=policy.science_evaluate(row(),tree(),model,.02,.5)
        self.assertEqual(zero,original)
        self.assertEqual(half['path'],original['path'])
        for key in ('compute','total_parameter_compute','solver_tokens','solver_calls',
                    'verifier_tokens','verifier_token_slots','verifier_calls','judge_calls'):
            self.assertEqual(half[key],original[key])
        self.assertEqual(half['solver_calls'],3)
        self.assertEqual(half['verifier_calls'],2)

    def test_exact_half_formula_and_invalid_current_support(self):
        rng=np.random.default_rng(38)
        p=core.normalize(rng.uniform(size=16));prior=core.normalize(rng.uniform(size=16))
        dynamics=FixedDynamics();dynamics.emission=rng.uniform(size=dynamics.emission.shape)
        dynamics.emission/=dynamics.emission.sum(2,keepdims=True)
        inspected=[]
        def predicted(public,path,paid):
            self.assertFalse(public['answer']);self.assertEqual(set(paid),{'','repair'})
            inspected.append(path);return p
        model=SimpleNamespace(dynamics={'physics':dynamics},predicted=predicted)
        for valid in (False,True):
            paid=tree();paid['repair']['answer']='B' if valid else ''
            obs=core.observation(p,valid)
            finite=core.mask_invalid((prior@dynamics.transition[0])*dynamics.emission[0,:,obs],valid)
            expected=core.mask_invalid(.5*finite+.5*p,valid)
            actual=policy.update(model,row(),'repair',paid,prior,0,.5)
            np.testing.assert_array_equal(actual,expected)
            if not valid:self.assertEqual(float(actual@core.CURRENT),0.)
        self.assertEqual(inspected,['repair','repair'])
        with self.assertRaises(ValueError):policy.update(model,row(),'repair',tree(),prior,0,1.1)

    def test_only_paid_prefix_no_terminal_encoder_and_reference_independence(self):
        seen=[]
        class GuardTree(dict):
            def __getitem__(self,key):
                if key not in ('','repair','repair/complete'):raise AssertionError('Future branch read')
                return super().__getitem__(key)
        model=parent().science
        def features(public,path,paid,neural):
            self.assertFalse(public['answer']);self.assertEqual(set(paid),{'',path})
            seen.append(path);return [0.]
        with patch.object(core.plain,'observables',side_effect=features):
            a=policy.science_evaluate(row(),GuardTree(tree()),model,.02,.5)
            b=policy.science_evaluate({**row(),'answer':'B'},GuardTree(tree()),model,.02,.5)
        self.assertEqual(seen,['','repair','','repair'])
        self.assertEqual(a['decisions'],b['decisions'])
        self.assertNotEqual(a['correct'],b['correct'])
        self.assertEqual(a['verifier_calls'],2)

    def test_root_complete_terminal_and_missing_does_not_read_other_branch(self):
        model=parent().science;model.dynamics['physics']=FixedDynamics((3,))
        with patch.object(core.plain,'observables',return_value=[0.]):
            item=policy.science_evaluate(row(),tree(),model,.02,.5)
            self.assertEqual(item['solver_calls'],2);self.assertEqual(item['verifier_calls'],1)
            missing=tree();del missing['complete']
            with self.assertRaises(core.plain.MissingObservation):policy.science_evaluate(row(),missing,model,.02,.5)

    def test_build_preserves_learned_components_and_math_dispatch(self):
        original=parent();model=policy.build(original,{'source':'test'})
        for name in ('math','science'):
            self.assertEqual(dev.io.structural_digest(getattr(model,name)),dev.io.structural_digest(getattr(original,name)))
        self.assertIsNot(original.science,model.science)
        with patch.object(policy.parent_policy.continuous_repairability,'evaluate',return_value={'math':True}) as evaluate:
            self.assertEqual(policy.evaluate(row(subject='math'),{},{'unpaid':'review'},model,{},'A'),{'math':True})
            self.assertEqual(evaluate.call_args.args[2],{})
            self.assertEqual(evaluate.call_args.args[4],original.math_settings)
        for settings,family in (({'penalty':.03},'A'),({},'B')):
            with self.assertRaises(ValueError):policy.evaluate(row(),tree(),{},model,settings,family)
        original.science_settings={'penalty':.03}
        with self.assertRaises(ValueError):policy.build(original,{})

    def test_select_retains_stronger_previous_and_all_controls(self):
        metric=lambda correct,cost:{'n':256,'accuracy':correct/256,'correct':correct,'mean_compute':cost}
        controls={str(i):{'accuracy':(128+i)/256} for i in range(5)}
        matched={str(i):{'full256_with_retained_math_A':{'accuracy':(145+i)/256}} for i in range(6)}
        candidate={'dev':metric(145,100)};previous=[{'dev':metric(146,150)},{'dev':metric(144,90)}]
        result=dev.select(candidate,previous,controls,matched)
        self.assertEqual(result['selected_A'],previous[0]);self.assertTrue(result['retained_previous_A'])
        candidate={'dev':metric(147,200)}
        self.assertFalse(dev.select(candidate,previous,controls,matched)['retained_previous_A'])
        with self.assertRaises(ValueError):dev.select(candidate,previous,{},matched)

    def test_synthetic_full256_replay_seal_preserves_parent_and_controls(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'new';old=root/'old';output.mkdir();old.mkdir()
            (output/'registration.json').write_text('{}')
            rows=[row(str(i),'math' if i<128 else 'chemistry' if i<192 else 'physics') for i in range(256)]
            trees={r['id']:tree() for r in rows}
            for i,r in enumerate(rows):
                if r['subject']=='math':trees[r['id']]['']=node('A')
                elif i%2:trees[r['id']]['repair/complete']['answer']='B'
            original=parent();deploy=policy.build(original,{'synthetic':True})
            observer=Observer()
            def math_evaluate(r,t,*unused):
                out=core.plain.outcome(r,t,{},'',[''],[],[{'path':'','action':'stop','belief':[1.]}])
                out.update(verifier_tokens=128,verifier_calls=1,verifier_token_slots=128)
                return out
            with patch.object(core.plain,'observables',return_value=[0.]), \
                 patch.object(policy.parent_policy.continuous_repairability,'evaluate',side_effect=math_evaluate):
                values=[policy.parent_policy.evaluate(r,trees[r['id']],{},original,{},'A') for r in rows]
                math={r['id']:v for r,v in zip(rows,values) if r['subject']=='math'}
                baseline={name:dev.original.statistics([dev.original.fixed(r,trees[r['id']],*setting) for r in rows])
                    for name,setting in dev.original.BASELINES.items()}
                matched={}
                for name,prefix,conditional in dev.original.MATCHED:
                    items=[math[r['id']] if r['subject']=='math' else dev.original.matched_item(r,trees[r['id']],prefix,conditional) for r in rows]
                    matched[name]={'full256_with_retained_math_A':dev.original.statistics(items),
                        'science':{s:dev.original.statistics([v for r,v in zip(rows,items)
                            if (r['subject']!='math' if s=='all' else r['subject']==s)]) for s in ('all','chemistry','physics')}}
                summary={'completion_A':{'key':'parent','dev':dev.original.statistics(values)},
                    'math_metrics':dev.original.statistics(list(math.values())),'baselines':baseline,'matched_complete_controls':matched}
                dev.io.write(old/'private-candidate-items.json',{'parent':values})
                receipt={'science_structural_sha256':dev.io.structural_digest(deploy.science),
                    'math_structural_sha256':dev.io.structural_digest(deploy.math),'deploy_structural_sha256':dev.io.structural_digest(deploy)}
                registered={'previous_A_candidates':[{'dev':{'n':256,'correct':195,'accuracy':195/256,'mean_compute':10.}}],
                    'source_files':{},'input_evidence':{},'dev_questions_sha256':'synthetic-panel'}
                args=SimpleNamespace(parent=old,output=output,registration_sha256=dev.io.sha(output/'registration.json'))
                bundle=(rows,trees,original,deploy,observer,summary,receipt)
                dev.replay(args,bundle,registered)
            actual=dev.io.read(output/'summary.json');complete=dev.io.read(output/'complete.json')
            self.assertEqual(actual['blend_A']['dev']['n'],256)
            self.assertEqual(actual['blend_A']['dev']['correct'],192)
            self.assertEqual(actual['selected_A']['dev']['correct'],195)
            self.assertEqual(actual['baselines'],baseline);self.assertEqual(actual['matched_complete_controls'],matched)
            self.assertTrue(actual['parent_exactly_reproduced']);self.assertFalse(actual['fresh_validation'])
            self.assertEqual(actual['candidate_count'],1)
            self.assertEqual(actual['blend_A']['policy_module'],'completion_blend_pomdp')
            dev.io.verify({str(output/name):digest for name,digest in complete['artifacts'].items()})
            saved=pickle.loads((output/'completion-blend-A.pkl').read_bytes())
            self.assertEqual(saved.posterior_weight,.5)
            self.assertEqual(dev.io.structural_digest(saved),receipt['deploy_structural_sha256'])


if __name__=='__main__':unittest.main()
