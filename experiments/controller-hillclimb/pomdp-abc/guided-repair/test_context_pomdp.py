"""Check belief arithmetic, action costs, role separation, and hidden-label invariance."""

import json
import unittest
import numpy as np

from context_pomdp import Dynamics, Planner, REVIEW_WIDTH, evaluate
from gate import Logistic
from extended_acquire import parse_verdict


def constant(probability):
    return Logistic(np.zeros(1), np.ones(1), np.array([np.log(probability/(1-probability)), 0.]))


def dynamics():
    transition = np.tile(np.array([[.7, .98], [.4, .9]])[None], (2, 1, 1))
    emission = np.full((2, 2, 2, 6), 1/6)
    emission[:, :, 1, 0] = 0
    emission /= emission.sum(axis=-1, keepdims=True)
    costs = np.full((2, 2, 2), 400.)
    review = np.full((6, 2, REVIEW_WIDTH), 1/REVIEW_WIDTH)
    reviewed = np.repeat(transition[:, None], REVIEW_WIDTH, axis=1)
    calibration = Logistic(np.zeros(1), np.ones(1), np.array([0., 1.]))
    return Dynamics(.5, emission[0, 0], transition, emission, costs, review, reviewed,
                    emission.copy(), costs.copy(), np.array([250., 300.]),
                    [[constant(v) for v in a] for a in transition[0]], calibration)


class BeliefArithmetic(unittest.TestCase):
    def test_one_step_values_use_transition_and_full_cost(self):
        model = dynamics()
        planned = model.bellman(.1)
        values = model.action_values(.4, 1, model.transition[1], model.costs[1], planned[0].max(axis=0), .1)
        np.testing.assert_allclose(values, [.4, .6*.7+.4*.98-.04, .6*.4+.4*.9-.04])

    def test_bayesian_update_matches_manual_likelihood(self):
        model = dynamics()
        model.emission[0, 0, 0] = [0.5, 0., .5, 0., 0., 0.]
        model.emission[0, 0, 1] = [0., .2, .8, 0., 0., 0.]
        prior = .6*.7+.4*.98
        expected = prior*.8/((1-prior)*.5+prior*.8)
        self.assertAlmostEqual(model.posterior(.4, model.transition[0], 0, 2, 0), expected)

    def test_uninformative_review_loses_its_charged_cost(self):
        model = dynamics()
        planned = model.bellman(.1)
        current = model.action_values(.4, 1, model.transition[1], model.costs[1], planned[0].max(axis=0), .1)
        self.assertAlmostEqual(model.information_value(.4, 1, current, planned, .1), -.03, places=10)

    def test_perfect_free_review_has_nonnegative_value(self):
        model = dynamics()
        model.review_emission[:] = 0
        model.review_emission[:, 0, 0] = 1
        model.review_emission[:, 1, 1] = 1
        model.judge_costs[:] = 0
        planned = model.bellman(.2)
        current = model.action_values(.4, 1, model.transition[1], model.costs[1], planned[0].max(axis=0), .2)
        self.assertGreater(model.information_value(.4, 1, current, planned, .2), 0)

    def test_model_round_trip_preserves_plans(self):
        model = dynamics()
        model.agreement_emission = np.full((6, 2, 3), 1/3)
        restored = Dynamics.restore(json.loads(json.dumps(model.as_dict())))
        np.testing.assert_allclose(model.bellman(.08), restored.bellman(.08))
        np.testing.assert_allclose(model.predict_transition(np.zeros((1, 1)), 0, True),
                                   restored.predict_transition(np.zeros((1, 1)), 0, True))
        np.testing.assert_allclose(model.agreement_emission, restored.agreement_emission)

    def test_paid_mentor_agreement_is_a_bayesian_observation(self):
        model = dynamics()
        model.agreement_emission = np.tile(np.array([[.2, .7, .1], [.1, .1, .8]])[None], (6, 1, 1))
        p = .6*.7+.4*.98
        expected = p*.2*.8/((1-p)/6*.1+p*.2*.8)
        self.assertAlmostEqual(model.posterior(.4, model.transition[0], 0, 2, 0, True, 2), expected)


class ReviewParsing(unittest.TestCase):
    def test_latex_escape_recovery_preserves_explicit_verdict(self):
        text = r'{"solution":"Use \Delta E.","answer":"A","check":"Units agree.","p_correct":0.9,"action":"stop"}'
        obj, valid = parse_verdict(text)
        self.assertTrue(valid)
        self.assertEqual(obj['solution'], r'Use \Delta E.')
        self.assertEqual(obj['p_correct'], .9)
        self.assertEqual(obj['action'], 'stop')

    def test_truncation_does_not_invent_a_verdict(self):
        obj, valid = parse_verdict('{"solution":"An unfinished derivation')
        self.assertFalse(valid)
        self.assertEqual(obj['p_correct'], .5)
        self.assertEqual(obj['action'], 'repair')


class RoleAndAccounting(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'unit', 'domain': 'math', 'subject': 'math', 'question': '1+0?', 'answer': '1'}
        paths = ('', 'repair', 'retry', 'guided', 'repair/repair', 'repair/retry', 'retry/repair',
                 'retry/retry', 'guided/repair', 'guided/retry')
        self.tree = {p: {'answer': '1' if p else '0', 'text': 'Final: 1' if p else 'Final: 0',
                         'prompt_sha256': p, 'input_tokens': 100, 'output_tokens': 20} for p in paths}
        self.judges = {('unit', p): {'action': 'repair' if not p else 'stop', 'p_correct': .1 if not p else 1.,
                                    'answer': '1', 'parse_valid': True, 'input_tokens': 200, 'output_tokens': 50}
                       for p in ('', 'repair', 'retry', 'guided')}

    def test_b_uses_llm_without_any_pomdp(self):
        result = evaluate(self.row, self.tree, self.judges, None, {'judge_stop': 'action'}, 'B')
        self.assertEqual(result['path'], 'guided')
        self.assertEqual(result['solver_calls'], 2)
        self.assertEqual(result['judge_calls'], 2)
        self.assertEqual(result['correct'], 1)
        self.assertAlmostEqual(result['compute'], (240*4+500*3)/3.09)
        self.assertAlmostEqual(result['total_parameter_compute'], (240*4+500*35)/3.09)

    def test_b_is_invariant_to_hidden_reference(self):
        first = evaluate(self.row, self.tree, self.judges, None, {'judge_stop': 'action'}, 'B')
        changed = evaluate({**self.row, 'answer': '999'}, self.tree, self.judges, None, {'judge_stop': 'action'}, 'B')
        self.assertNotEqual(first.pop('correct'), changed.pop('correct'))
        self.assertEqual(first, changed)

    def test_b_can_reuse_a_paid_review_without_another_llm_call(self):
        result = evaluate(self.row, self.tree, self.judges, None,
                          {'judge_stop': 'action', 'reuse_review': True}, 'B')
        self.assertEqual(result['path'], 'guided')
        self.assertEqual(result['judge_calls'], 1)
        self.assertEqual(result['solver_calls'], 2)
        self.assertFalse(result['decisions'][-1]['queried'])
        self.assertAlmostEqual(result['compute'], (240*4+250*3)/3.09)

    def test_a_never_reads_llm_observations(self):
        class NoLLM(dict):
            def __getitem__(self, key):
                raise AssertionError('A read an LLM observation')

        class StubPlanner:
            def __init__(self):
                self.dynamics = {'math': dynamics()}

            def observed(self, row, path, tree):
                return .8, 3, np.zeros((1, 1))

        result = evaluate(self.row, self.tree, NoLLM(), StubPlanner(), {'penalty': 5.}, 'A')
        self.assertEqual(result['judge_calls'], 0)
        self.assertEqual(result['path'], '')

    def test_hybrid_decisions_ignore_hidden_reference_and_charge_reviews(self):
        class StubPlanner:
            def __init__(self):
                self.dynamics = {'math': dynamics()}

            def observed(self, row, path, tree):
                return .5, 3, np.zeros((1, 1))

        settings = {'penalty': .1, 'threshold': -10., 'contextual': False}
        first = evaluate(self.row, self.tree, self.judges, StubPlanner(), settings, 'C')
        changed = evaluate({**self.row, 'answer': '999'}, self.tree, self.judges, StubPlanner(), settings, 'C')
        self.assertGreater(first['judge_calls'], 0)
        self.assertTrue(first['path'].startswith('guided'))
        self.assertAlmostEqual(first['compute'], (first['solver_calls']*120*4+first['judge_calls']*250*3)/3.09)
        self.assertNotEqual(first.pop('correct'), changed.pop('correct'))
        self.assertEqual(first, changed)


if __name__ == '__main__':
    unittest.main()
