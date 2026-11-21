"""Numerical planning and independently priced blind-prefix tests; no model run."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import second_review_pomdp as p
import development_second_review as dev
from prospective_audit import encoder_ledger, audit_costs
from test_stronger_base_hybrid import FakeObserver, FakeBase, FakeRegular, parent


def stochastic(rng, shape, support=None):
    value = rng.uniform(.05, 1., shape)
    if support is not None:
        value *= support
    return value / value.sum(-1, keepdims=True)


def critical_dynamics(seed=7):
    rng = np.random.default_rng(seed)
    lift = np.zeros((8, 16))
    for s in range(8):
        lift[s, 2*s:2*s+2] = stochastic(rng, (2,))
    return p.CriticalDynamics(lift, stochastic(rng, (16, 8)),
                              stochastic(rng, (2, 16, 16), p.terminal_support()),
                              rng.uniform(10, 1000, (2, 16)), rng.uniform(100, 3000, 16))


def mentor_dynamics(seed=11):
    rng = np.random.default_rng(seed)
    return p.mentor_core.JointDynamics(
        np.ones(8) / 8, np.ones((8, 11)) / 11,
        stochastic(rng, (2, 2, 8, 8)), stochastic(rng, (8, 32)),
        stochastic(rng, (32, 2, 8, 8)), stochastic(rng, (2, 8, 33)),
        stochastic(rng, (2, 2, 8, 8)), rng.uniform(10, 1000, (2, 8)),
        rng.uniform(10, 1000, (2, 2, 8)), rng.uniform(100, 3000, 8), np.zeros((2, 8, 3)))


def enumerate_query(d, belief8, penalty):
    before = belief8 @ d.lift
    value = -penalty * float(before @ d.review_costs) / 1000
    for obs in range(d.review.shape[1]):
        joint = before * d.review[:, obs]
        mass = joint.sum()
        if mass:
            posterior = joint / mass
            candidates = [float(posterior @ p.REWARD)]
            for action in (0, 1):
                candidates.append(float(posterior @ d.transition[action] @ p.REWARD)
                                  - penalty * float(posterior @ d.costs[action]) / 1000)
            value += mass * max(candidates)
    return value


def node(answer):
    return {'answer': answer, 'text': 'Final: ' + answer, 'input_tokens': 10,
            'output_tokens': 5, 'prompt_sha256': answer}


def review(answer):
    return {'answer': answer, 'text': 'Review: ' + answer, 'p_correct': .1,
            'action': 'repair', 'parse_valid': True, 'verdict': {'solution': 'Review'},
            'input_tokens': 10, 'output_tokens': 6, 'physical_api_calls': 2,
            'calls': [{'input_tokens': 5, 'output_tokens': 3}] * 2}


def replay(function, row, available_tree, available_reviews, model=None):
    """Second-review-specific legality; independent generic physical-cost audit.

    The existing prospective C auditor intentionally remains limited to one
    review. Its B horizon allows two, so use only its cost routine with that
    limit after explicitly enforcing this new C's stricter availability here.
    """
    safe = {**row, 'answer': ''}; tree, judges, current = {'': available_tree['']}, {}, ''
    with encoder_ledger(model) as ledger:
        for _ in range(6):
            ledger.clear()
            try:
                item = function(safe, dict(tree), dict(judges))
                if item['path'] != current:
                    raise AssertionError('Unpaid output selected')
                audit_costs(item, tree, judges, list(ledger.values()), 'B')
                return item, list(ledger.values())
            except p.action_core.MissingObservation as missing:
                if missing.kind == 'judge':
                    if missing.path != current or (safe['id'], current) in judges:
                        raise AssertionError('Review must be new and concern the current paid answer')
                    if current and (current != 'guided' or safe['subject'] not in ('chemistry', 'physics') or len(tree) != 2):
                        raise AssertionError('Forbidden critical-review availability')
                    if len(judges) >= 2:
                        raise AssertionError('Too many reviews')
                    judges[safe['id'], current] = available_reviews[safe['id'], current]
                elif missing.kind == 'solver':
                    ancestor = missing.path.rsplit('/', 1)[0] if '/' in missing.path else ''
                    if ancestor != current or len(tree) >= 3 or missing.path in tree:
                        raise AssertionError('Nonsequential or over-budget continuation')
                    tree[missing.path] = available_tree[missing.path]; current = missing.path
                else:
                    raise AssertionError('Unexpected observation')
    raise AssertionError('Exceeded finite horizon')


def forced_model(folder, subject='physics'):
    base = FakeBase(FakeObserver(folder / 'base', 400_000_000))
    observer = FakeObserver(folder / 'mentor', 600_000_000)
    base.dynamics = {s: SimpleNamespace(values=lambda *args: np.zeros(3)) for s in base.dynamics}
    def predicted(row, path, tree):
        base.classifier.values(row, path, tree)
        return np.eye(8)[0]
    base.predicted = predicted
    d = mentor_dynamics()
    d.review = np.ones((8, 32)) / 32
    d.reviewed_transition = np.broadcast_to(np.eye(8), (32, 2, 8, 8)).copy()
    d.emission = np.ones((2, 8, 33)) / 33
    d.last_transition = np.broadcast_to(np.eye(8), (2, 2, 8, 8)).copy()
    d.first_costs[:] = 10.; d.last_costs[:] = 10.; d.review_costs[:] = 10.
    old = parent(observer, semantic=False)
    old.mentor.dynamics = {s: d for s in base.dynamics}
    primary = p.base_core.build(base, 'repairability_pomdp', {}, old, {})
    critical = critical_dynamics()
    critical.review = np.ones((16, 8)) / 8
    critical.transition[0] = np.asarray([np.eye(16)[s | 4] for s in range(16)])
    critical.transition[1] = np.eye(16)
    critical.costs[:] = 10.; critical.review_costs[:] = 10.
    return p.Planner(primary, {s: critical for s in ('chemistry', 'physics')}, {})


class SecondReviewTest(unittest.TestCase):
    def test_query_matches_explicit_bayes_and_cost_units(self):
        d = critical_dynamics(); belief = np.arange(1., 9.); belief /= belief.sum()
        for penalty in (0., .08, .5):
            self.assertAlmostEqual(d.query_value(belief, penalty), enumerate_query(d, belief, penalty), places=12)
            self.assertAlmostEqual(d.query_value(.13 * belief, penalty), .13 * d.query_value(belief, penalty), places=12)
        np.testing.assert_allclose(d.alpha(.2)[1], d.transition[0] @ p.REWARD - .2 * d.costs[0] / 1000)
        self.assertLessEqual(d.query_value(belief, .5), d.query_value(belief, 0.))

    def test_root_backup_plans_second_review_only_after_guided(self):
        d, critical = mentor_dynamics(), critical_dynamics()
        belief = np.arange(1., 9.); belief /= belief.sum()
        penalty, threshold, rs = .08, .03, 6
        expected = [float(belief @ p.mentor_core.REWARD)]
        for action in (0, 1):
            before = belief @ d.reviewed_transition[rs, action]
            value = -penalty * float(belief @ d.first_costs[action]) / 1000
            for obs in range(33):
                joint = before * d.emission[action, :, obs]
                mass = joint.sum(); posterior = joint / mass
                plain = (d.alpha(action, penalty) @ posterior).max()
                future = max(plain, enumerate_query(critical, posterior, penalty) - threshold) if action == 0 else plain
                value += mass * future
            expected.append(value)
        np.testing.assert_allclose(p.root_values(d, critical, belief, rs, penalty, threshold), expected, atol=1e-12)
        information = sum(p.root_values(d, critical, belief * d.review[:, r], r, penalty, threshold).max() for r in range(32))
        information -= penalty * float(belief @ d.review_costs) / 1000
        self.assertAlmostEqual(p.root_information(d, critical, belief, penalty, threshold), information)
        np.testing.assert_allclose(p.root_values(d, None, belief, rs, penalty, threshold), d.values(belief, rs, penalty))

    def test_fit_preserves_all_hidden_truth_bits_and_uses_all_pairs(self):
        records = [{'id': str(i), 'subject': 'chemistry' if i < 32 else 'physics',
                    'state8': i % 8, 'critical_correct': (i // 8) % 2,
                    'signal': i % 6, 'outcomes': [i % 2, (i // 2) % 2],
                    'solver_compute': [100. + i, 150. + i], 'review_compute': 500. + i}
                   for i in range(64)]
        for d in p.fit_critical(records).values():
            for matrix in (d.lift, d.review, d.transition):
                np.testing.assert_allclose(matrix.sum(-1), 1.)
            np.testing.assert_allclose(d.lift @ p.REWARD, p.mentor_core.REWARD)
            self.assertEqual(np.count_nonzero(d.transition * (1 - p.terminal_support())[None]), 0)
            self.assertEqual(np.count_nonzero(d.review[p.STATES % 2 == 1, 7]), 0)
            for old in range(8):
                self.assertEqual(set(np.nonzero(d.lift[old])[0]), {2 * old, 2 * old + 1})
        with self.assertRaisesRegex(ValueError, 'all64'):
            p.fit_critical(records[:-1])
        with self.assertRaisesRegex(ValueError, 'all64'):
            p.fit_critical([records[0]] + records[1:63] + [records[0]])

    def test_two_reviews_and_three_solvers_are_blind_and_fully_billed(self):
        with TemporaryDirectory() as folder:
            model = forced_model(Path(folder))
            row = {'id': 'q', 'question': 'Public science?', 'answer': 'B', 'subject': 'physics', 'domain': 'science'}
            tree = {'': node('A'), 'guided': node('A'), 'guided/critical-repair': node('B')}
            judges = {('q', ''): review('A'), ('q', 'guided'): review('B')}
            item, ledger = replay(lambda r, t, j: p.evaluate(r, t, j, model, {'penalty': .01}, 'C'), row, tree, judges, model)
            self.assertEqual(item['path'], 'guided/critical-repair')
            self.assertEqual((item['solver_calls'], item['judge_calls'], item['judge_physical_calls']), (3, 2, 4))
            self.assertEqual(item['verifier_calls'], 2)
            self.assertEqual({(r['kind'], r['path']) for r in ledger}, {('solver', ''), ('solver', 'guided')})
            self.assertEqual(len(item['decisions'][1]['belief']), 16)
            self.assertTrue(item['decisions'][1]['critical_mentor_mode'])
            self.assertAlmostEqual(item['compute'], (4 * 45 + 3 * 32 + 13 * (0.4 + 0.6)) / 3.09)

    def test_capability_matched_b_has_same_reviews_and_zero_neural_calls(self):
        row = {'id': 'q', 'question': 'Public science?', 'answer': 'B', 'subject': 'chemistry', 'domain': 'science'}
        tree = {'': node('A'), 'guided': node('A'), 'guided/critical-repair': node('B')}
        judges = {('q', ''): review('A'), ('q', 'guided'): review('B')}
        setting = {'root_stop': 'action', 'critical_query': 'always', 'critical_stop': 'action'}
        item, ledger = replay(lambda r, t, j: dev.llm_policy(r, t, j, setting), row, tree, judges)
        self.assertEqual(item['path'], 'guided/critical-repair')
        self.assertEqual((item['solver_calls'], item['judge_calls'], item['judge_physical_calls']), (3, 2, 4))
        self.assertEqual(item['verifier_calls'], 0); self.assertEqual(ledger, [])
        # Answer truth is never an input to action selection.
        changed = dev.llm_policy({**row, 'answer': 'SECRET'}, tree, judges, setting)
        self.assertEqual(changed['decisions'], item['decisions'])
        row['subject'] = 'math'; row['domain'] = 'math'
        stopped, _ = replay(lambda r, t, j: dev.llm_policy(r, t, j, setting), row, tree, judges)
        self.assertEqual(stopped['path'], 'guided'); self.assertEqual(stopped['judge_calls'], 1)

    def test_no_query_preserves_selected_base_settings_and_omits_review_cost(self):
        with TemporaryDirectory() as folder:
            model = forced_model(Path(folder))
            model.primary.base_module = 'continuous_repairability'
            model.primary.base_settings = {'penalty': .08, 'posterior_weight': .5}
            row = {'id': 'q', 'question': 'Public?', 'answer': 'A', 'subject': 'physics', 'domain': 'science'}
            tree = {'': node('A')}
            def plain(safe, paid, judges, base, settings, family):
                self.assertEqual(safe['answer'], '')
                self.assertEqual(settings, {'penalty': .24, 'posterior_weight': .5})
                return p.mentor_core.outcome(safe, paid, {}, '', [''], [], [{'path': '', 'action': 'stop', 'queried': False}])
            with patch.object(p.base_core.continuous, 'evaluate', side_effect=plain):
                item = p.evaluate(row, tree, {}, model, {'penalty': .24, 'threshold': 10.}, 'C')
            self.assertEqual(item['judge_calls'], 0); self.assertEqual(item['verifier_calls'], 1)
            self.assertIsNone(model.mentor.classifier._model)
            self.assertEqual(item['correct'], 1)


if __name__ == '__main__':
    unittest.main()
