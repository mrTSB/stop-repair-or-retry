import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

import completion_filter_diagnostic as diagnostic


def fixture(n=48):
    rng=np.random.default_rng(238);records=[];labels=[];valid=[];costs=[]
    for i in range(n):
        qid=f'q{i}';root=int(rng.integers(0,16))
        states=[root,*[diagnostic.core.state(diagnostic.core.AFTER[a,root],*rng.integers(0,2,3)) for a in (0,1)]]
        for path,label in zip(('','repair','retry'),states):
            records.append({'id':qid,'subject':'chemistry' if i<n//2 else 'physics',
                'path':path,'fold':diagnostic.fixed.refined.fold_for(qid)})
            labels.append(label);valid.append(True);costs.append([100+i,130+i,70+i])
    return {'records':records,'features':rng.normal(size=(3*n,9)),
        'labels':np.asarray(labels),'valid':np.asarray(valid),'costs':np.asarray(costs,dtype=float)}


class NestedFilterTests(unittest.TestCase):
    def test_outer_held_labels_costs_and_features_cannot_fit_any_component(self):
        data=fixture()
        with threadpool_limits(limits=1):
            held,p,forecaster,dynamics,receipt=diagnostic.fit_fold(data,0)
            changed=copy.deepcopy(data);changed['labels'][held]^=1;changed['costs'][held]+=1e6
            held2,p2,forecaster2,dynamics2,receipt2=diagnostic.fit_fold(changed,0)
            np.testing.assert_array_equal(held,held2);np.testing.assert_array_equal(p,p2)
            np.testing.assert_array_equal(forecaster[1].coef_,forecaster2[1].coef_)
            for subject in dynamics:
                for field in ('prior','transition','emission','cost'):
                    np.testing.assert_array_equal(getattr(dynamics[subject],field),getattr(dynamics2[subject],field))
            self.assertEqual(receipt['inner_probabilities_sha256'],receipt2['inner_probabilities_sha256'])
            changed['features'][held]+=1000
            _,_,forecaster3,dynamics3,_=diagnostic.fit_fold(changed,0)
            np.testing.assert_array_equal(forecaster[0].mean_,forecaster3[0].mean_)
            np.testing.assert_array_equal(forecaster[1].coef_,forecaster3[1].coef_)
            np.testing.assert_allclose(forecaster[0].mean_,data['features'][~held].mean(0))
            for subject in dynamics:np.testing.assert_array_equal(dynamics[subject].emission,dynamics3[subject].emission)

    def test_nested_inner_salt_has_three_folds_and_excludes_every_outer_question(self):
        data=fixture();outer=1
        with threadpool_limits(limits=1):held,_,_,_,receipt=diagnostic.fit_fold(data,outer)
        train={r['id'] for r,flag in zip(data['records'],held) if not flag}
        excluded={r['id'] for r,flag in zip(data['records'],held) if flag}
        self.assertFalse(train&excluded);self.assertEqual(len(train)+len(excluded),48)
        self.assertEqual(receipt['held_questions']*3,int(held.sum()))
        self.assertEqual({diagnostic.inner_fold(q,outer) for q in train},{0,1,2})
        for item in receipt['inner_folds']:
            fit={q for q in train if diagnostic.inner_fold(q,outer)!=item['fold']}
            valid=train-fit
            self.assertFalse((fit|valid)&excluded)
            self.assertEqual(item['fit_ids_sha256'],diagnostic.fixed.fingerprint(sorted(fit)))
            self.assertEqual(item['validation_ids_sha256'],diagnostic.fixed.fingerprint(sorted(valid)))

    def test_exact_state_support_normalized_dynamics_and_existing_bellman(self):
        data=fixture()
        with threadpool_limits(limits=1):_,_,_,dynamics,_=diagnostic.fit_fold(data,2)
        for d in dynamics.values():
            self.assertIs(type(d),diagnostic.core.Dynamics)
            np.testing.assert_allclose(d.prior.sum(),1.)
            np.testing.assert_allclose(d.transition.sum(2),1.)
            np.testing.assert_allclose(d.emission.sum(2),1.)
            np.testing.assert_array_equal(d.emission[:,diagnostic.core.CURRENT==1,24:],0.)
            for action in (0,1):
                support=diagnostic.core.CURRENT[None,:]==diagnostic.core.AFTER[action,:,None]
                np.testing.assert_array_equal(d.transition[action][~support],0.)
            self.assertTrue(np.isfinite(d.cost).all());self.assertTrue((d.cost>0).all())
            self.assertEqual(d.values(np.full(16,1/16),2,.02).shape,(4,))
        ids={r['id'] for r in data['records']};signals={(r['id'],r['path']):0 for r in data['records']}
        missing=signals.copy();missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):diagnostic.fit_dynamics(data,ids,missing)

    def test_half_is_exact_convex_mixture_not_likelihood_division(self):
        rng=np.random.default_rng(238)
        transition=rng.random((2,16,16));transition/=transition.sum(2,keepdims=True)
        emission=rng.random((2,16,30));emission/=emission.sum(2,keepdims=True)
        d=diagnostic.core.Dynamics(np.ones(16)/16,transition,emission,np.ones((2,3,16)))
        root=rng.random(16);root/=root.sum();child=rng.random(16);child/=child.sum()
        finite,blend=diagnostic.child_beliefs(d,root,child,1,True)
        obs=diagnostic.core.observation(child,True)
        expected=diagnostic.core.normalize((root@transition[1])*emission[1,:,obs])
        np.testing.assert_allclose(finite,expected);np.testing.assert_allclose(blend,(expected+child)/2)
        invalid=diagnostic.core.mask_invalid(child,False)
        finite,blend=diagnostic.child_beliefs(d,root,invalid,1,False)
        for p in (finite,blend):
            np.testing.assert_array_equal(p[diagnostic.core.CURRENT==1],0.)
            self.assertAlmostEqual(p.sum(),1.)

    def test_identical_root_forecasts_and_all_held_children_no_label_access(self):
        data=fixture()
        with threadpool_limits(limits=1):held,raw,_,dynamics,_=diagnostic.fit_fold(data,0)
        values=diagnostic.evaluate_fold(data,held,raw,dynamics)
        roots=np.asarray([not r['path'] for r,flag in zip(data['records'],held) if flag])
        for p in values.values():
            self.assertEqual(p.shape,raw.shape);np.testing.assert_array_equal(p[roots],raw[roots])
        changed=copy.deepcopy(data);changed['labels']^=15
        other=diagnostic.evaluate_fold(changed,held,raw,dynamics)
        for name in values:np.testing.assert_array_equal(values[name],other[name])

    def test_primary_screen_uses_two_children_per_question_and_excludes_root(self):
        data=fixture(256);uniform=np.full((768,16),1/16);perfect=np.eye(16)[data['labels']]
        roots=np.asarray([not r['path'] for r in data['records']]);perfect[roots]=uniform[roots]
        result=diagnostic.reports(data,dict(zip(diagnostic.METHODS,(uniform,perfect))))
        self.assertTrue(result['screen_passed']);self.assertFalse(result['controller_accuracy_claim'])
        self.assertEqual(result['panels']['all/children']['nodes'],512)
        self.assertEqual(result['panels']['all/root']['nodes'],256)
        self.assertIn('not the posterior',result['approximate_blend_limitation'])
        for value in result['child_comparisons'].values():self.assertLess(value['paired_question_CI95'][1],0)
        same=diagnostic.reports(data,dict.fromkeys(diagnostic.METHODS,uniform))
        self.assertFalse(same['screen_passed'])

    def test_protected_split_and_neural_guards_retained(self):
        with TemporaryDirectory() as temporary,diagnostic.fixed.training_only_reads():
            for filename in ('dev-questions.jsonl','test-questions.jsonl'):
                with self.assertRaises(AssertionError):Path(temporary,filename).read_bytes()
        with diagnostic.fixed.guarded.no_model_calls():
            with self.assertRaises(AssertionError):diagnostic.fixed.guarded.perception.Observer.load(None)


if __name__=='__main__':unittest.main()
