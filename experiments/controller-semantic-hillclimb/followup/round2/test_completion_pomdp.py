import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import completion_pomdp as core


def mass(index):
    out = np.zeros(16); out[index] = 1.
    return out


class CompletionTests(unittest.TestCase):
    def test_exact_bellman_matches_explicit_posterior_expectation(self):
        rng = np.random.default_rng(12)
        transition = rng.uniform(size=(2, 16, 16)); transition /= transition.sum(2, keepdims=True)
        emission = rng.uniform(size=(2, 16, core.N_OBS)); emission /= emission.sum(2, keepdims=True)
        cost = rng.uniform(100, 900, (2, 3, 16))
        dynamics = core.Dynamics(core.normalize(rng.uniform(size=16)), transition, emission, cost)
        belief, penalty = core.normalize(rng.uniform(size=16)), .13
        expected = [belief @ core.CURRENT]
        for action in range(3):
            value = 0.
            if action == 2:
                value = belief @ core.AFTER[2]
            else:
                predicted = belief @ transition[action]
                for obs in range(core.N_OBS):
                    joint = predicted * emission[action, :, obs]
                    weight = joint.sum(); posterior = joint / weight
                    terminal = [posterior @ core.CURRENT] + [
                        posterior @ (core.AFTER[a] - penalty * cost[1, a] / 1000)
                        for a in range(3)]
                    value += weight * max(terminal)
            expected.append(value - penalty * (belief @ cost[0, action]) / 1000)
        np.testing.assert_allclose(dynamics.values(belief, 2, penalty), expected, atol=1e-14)
        np.testing.assert_allclose(dynamics.values(belief, 1, penalty),
            [belief @ core.CURRENT, *[belief @ (core.AFTER[a] - penalty * cost[1, a] / 1000)
                                     for a in range(3)]], atol=1e-14)

    def test_invalid_observation_keeps_complete_vs_retry_fixability(self):
        complete = mass(core.state(0, 0, 0, 1))
        retry = mass(core.state(0, 0, 1, 0))
        self.assertNotEqual(core.observation(complete, False), core.observation(retry, False))
        self.assertTrue(24 <= core.observation(complete, False) < core.N_OBS)
        masked = core.mask_invalid(mass(15), False)
        self.assertEqual(masked @ core.CURRENT, 0)
        self.assertAlmostEqual(masked.sum(), 1)
        for s in core.STATES:
            self.assertTrue(0 <= core.observation(mass(s), True) < 24)

    def test_invalid_horizon_or_math_route_rejected(self):
        dynamics = core.Dynamics(None, None, None, None)
        with self.assertRaises(ValueError): dynamics.values(mass(0), 0, .1)
        with self.assertRaises(ValueError): dynamics.values(mass(0), 2, -.1)
        with self.assertRaises(ValueError):
            core.evaluate({'subject': 'math'}, {}, {}, None, {}, 'A')
        with self.assertRaises(ValueError):
            core.evaluate({'subject': 'physics'}, {}, {}, None, {}, 'C')

    def exercise_route(self, route, expect_path, observed):
        row = {'id': 'q', 'domain': 'science', 'subject': 'physics', 'answer': 'A'}
        node = lambda answer, tokens: {'answer': answer, 'text': answer, 'input_tokens': tokens, 'output_tokens': 1}
        tree = {'': node('', 100), 'repair': node('', 200), 'complete': node('A', 300),
                'repair/complete': node('A', 400), 'retry': {'do_not_read': True}}
        calls = []
        class Observer:
            def charges(self, public, paid, judges, decisions, include_reviews):
                self_outer.assertEqual(public['answer'], '')
                self_outer.assertEqual(set(paid), set(observed))
                self_outer.assertFalse(judges); self_outer.assertFalse(include_reviews)
                self_outer.assertEqual([d['path'] for d in decisions], observed)
                return 128 * len(observed), len(observed), 128 * len(observed), .435
        self_outer = self
        class Dynamics:
            transition = np.tile(np.eye(16), (2, 1, 1))
            emission = np.ones((2, 16, core.N_OBS)) / core.N_OBS
            def values(self, belief, remaining, penalty):
                action = route[2 - remaining]
                return np.eye(4)[action]
        class Model:
            classifier = Observer()
            dynamics = {'physics': Dynamics()}
            def predicted(self, public, path, paid):
                self_outer.assertEqual(public['answer'], '')
                calls.append(path)
                return mass(core.state(0, 0, 0, 1))
        answer = core.evaluate(row, tree, {}, Model(), {'penalty': .1}, 'A')
        self.assertEqual(answer['path'], expect_path)
        self.assertEqual(calls, observed)
        self.assertEqual(answer['completion_calls'], 1)
        self.assertEqual(answer['solver_calls'], len(observed) + 1)
        self.assertEqual(answer['judge_calls'], 0)
        self.assertEqual(answer['correct'], 1)
        expected_tokens = sum(tree[p]['input_tokens'] + tree[p]['output_tokens']
                              for p in observed + [expect_path])
        self.assertAlmostEqual(answer['compute'], (4*expected_tokens + .435*128*len(observed))/3.09)

    def test_root_complete_is_terminal_and_all_tokens_charged(self):
        self.exercise_route([3], 'complete', [''])

    def test_repair_then_complete_has_no_terminal_encoder(self):
        self.exercise_route([1, 3], 'repair/complete', ['', 'repair'])

    def test_missing_complete_does_not_select_unpaid_alternative(self):
        model = SimpleNamespace(dynamics={'physics': SimpleNamespace(values=lambda *args: np.array([0,0,0,1]))},
                                predicted=lambda *args: mass(1))
        with self.assertRaises(core.plain.MissingObservation) as error:
            core.evaluate({'subject':'physics','answer':'A'}, {'':{'answer':''}}, {}, model, {'penalty':0.}, 'A')
        self.assertEqual(error.exception.path, 'complete')

    def test_real_predictor_sees_only_paid_prefix_and_no_reference(self):
        inspected = []
        class Observer:
            def values(self, public, path, paid):
                self_outer.assertEqual(public['answer'], '')
                self_outer.assertEqual(set(paid), {'', 'repair'})
                inspected.append(path)
                return (np.zeros(4), np.zeros(8), np.zeros(8), 128)
        self_outer = self
        class Forecaster:
            classes_ = np.array([0, 1, 15])
            def predict_proba(self, features): return np.array([[.1,.2,.7]])
        def features(public, path, paid, neural):
            self.assertEqual(public['answer'], '')
            self.assertEqual(set(paid), {'','repair'})
            return [0.]
        class PaidTree(dict):
            def __getitem__(self, key):
                if key not in ('','repair'): raise AssertionError('Unpaid branch accessed')
                return super().__getitem__(key)
        model = core.Planner(Observer(), Forecaster(), {}, {})
        with patch.object(core.plain, 'observables', side_effect=features):
            p = model.predicted({'subject':'physics','answer':'private-reference'}, 'repair',
                PaidTree({'':{'answer':'A'},'repair':{'answer':''}}))
        self.assertEqual(inspected, ['repair'])
        self.assertEqual(float(p @ core.CURRENT), 0.)
        self.assertAlmostEqual(p.sum(), 1.)

    def test_forecasts_exclude_whole_question_and_fit_has_no_nn_calls(self):
        qids = [f'q{i}' for i in range(30)]
        rows = [{'id': qid, 'subject': ('physics' if i%2 else 'chemistry'), 'answer': 'A'}
                for i, qid in enumerate(qids)]
        paths = ['', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry',
                 'complete', 'repair/complete', 'retry/complete']
        trees = {row['id']: {p: {'answer': ('A' if (i + len(p))%3 == 0 else 'B'),
                    'text': 'private outcome', 'input_tokens':100,'output_tokens':1} for p in paths}
                 for i, row in enumerate(rows)}
        seen = []
        def features(public, path, paid, neural):
            self.assertEqual(public['answer'], '')
            return [qids.index(public['id']), len(path)]
        original = core.classifier_for
        def fit_record(x, y, regularization):
            seen.append(set(x[:, 0].astype(int)))
            return original(x, y, regularization)
        lookup = {(row['id'], p): None for row in rows for p in ('','repair','retry')}
        observer = SimpleNamespace(_recipe={'parameters':435000000,'max_length':512})
        with patch.object(core.plain, 'observables', side_effect=features), patch.object(core, 'classifier_for', side_effect=fit_record):
            fitted = core.fit(rows, trees, observer, lookup)
        for fold in range(3):
            held = {i for i,q in enumerate(qids) if int(hashlib.sha256(q.encode()).hexdigest()[:8],16)%3 == fold}
            self.assertEqual(seen[fold], set(range(30)) - held)
        self.assertEqual(seen[3], set(range(30)))
        self.assertEqual(fitted.fit_receipt['state_count'], 16)
        self.assertFalse(fitted.fit_receipt['paid_scoring_features_used'])
        for dynamics in fitted.dynamics.values():
            np.testing.assert_allclose(dynamics.transition.sum(2), 1)
            np.testing.assert_allclose(dynamics.emission.sum(2), 1)
            self.assertEqual(dynamics.emission[:, core.CURRENT == 1, 24:].sum(), 0.)
            for action in (0,1):
                for state in core.STATES:
                    self.assertEqual(dynamics.transition[action,state, core.CURRENT != core.AFTER[action,state]].sum(), 0.)


if __name__ == '__main__':
    unittest.main()
