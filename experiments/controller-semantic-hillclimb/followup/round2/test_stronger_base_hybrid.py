import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import stronger_base_hybrid as h
from prospective_audit import replay_paid_inputs


class FakeObserver:
    def __init__(self, folder, parameters, slot_count=13):
        self.directory = str(folder); self.device = 'cpu'; self.temperature = 1.
        self._model = None; self._cache = {}; self._recipe = {}; self.slot_count = slot_count
        self._tokenizer = None; self._retriever = None; self._backgrounds = {}
        folder.mkdir(exist_ok=True)
        (folder / 'recipe.json').write_text(json.dumps({'parameters': parameters, 'max_length': 512}))

    def load(self):
        if self._model is None:
            self._model = object(); self._recipe = json.loads((Path(self.directory) / 'recipe.json').read_text())

    def entry(self, row, path, tree, judge=None):
        if row.get('answer'):
            raise AssertionError('Reference reached a neural observation')
        if set(tree) != set(h.prefix(tree, path)):
            raise AssertionError('Unpaid branch reached a neural observation')
        return {'question': row['question'], 'answer': (tree[path] if judge is None else judge)['answer']}

    def values(self, row, path, tree, judge=None):
        self.load(); self.entry(row, path, tree, judge)
        return np.zeros(4), np.zeros(8), np.zeros(8), self.slot_count

    def charges(self, row, tree, judges, decisions, include_reviews=False):
        slots = sum(self.values(row, d['path'], h.prefix(tree, d['path']))[3] for d in decisions)
        return slots, len(decisions), slots, self._recipe['parameters'] / 1e9


class FakeFusion:
    def __init__(self, observers):
        self.components = observers

    def load(self):
        for observer in self.components:
            observer.load()

    def values(self, row, path, tree, judge=None):
        values = [o.values(row, path, tree, judge) for o in self.components]
        return np.zeros(8), np.zeros(16), np.zeros(16), sum(v[3] for v in values)


class FakeBase:
    def __init__(self, observer):
        self.classifier = observer
        self.fit_receipt = {'state_count': 8, 'state_bits': ['current_correct', 'repair_correct', 'retry_correct']}
        self.dynamics = {s: SimpleNamespace(values=lambda b, r, p: np.array([1., 0., 0.]))
                         for s in ('math', 'chemistry', 'physics')}

    def predicted(self, row, path, tree):
        self.classifier.values({**row, 'answer': ''}, path, h.prefix(tree, path))
        return np.ones(8) / 8


class FakeRegular:
    def __init__(self, observer):
        self.classifier = observer

    def observed(self, row, path, tree):
        self.classifier.values(row, path, tree)
        return 1


class FakeMentorDynamics:
    def __init__(self, query=True):
        self.query = query
        self.review = np.ones((8, 32)) / 32
        self.emission = np.ones((2, 8, 33)) / 33
        self.reviewed_transition = np.broadcast_to(np.eye(8), (32, 2, 8, 8))

    def information_value(self, b, penalty):
        return 10. if self.query else -10.

    def values(self, b, signal, penalty):
        return np.array([0., 1., 0.])

    def alpha(self, action, penalty):
        return np.asarray([np.zeros(8), np.ones(8), np.zeros(8)])


def parent(observer, query=True, semantic=True):
    transfer = np.asarray([[float(h.action_core.CURRENT[i] == h.mentor_core.REWARD[j]) / 4
                            for j in range(8)] for i in range(8)])
    mentor = SimpleNamespace(classifier=observer, regular=FakeRegular(observer),
                             review_observer=observer if semantic else None, fit_receipt={},
                             dynamics={s: FakeMentorDynamics(query) for s in ('math', 'chemistry', 'physics')})
    return SimpleNamespace(mentor=mentor, transfer={s: transfer for s in mentor.dynamics})


class StrongBaseTest(unittest.TestCase):
    def test_assets_shared_only_for_identical_recipes(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            a = FakeObserver(directory / 'a', 400_000_000)
            same = FakeObserver(directory / 'same', 400_000_000)
            different = FakeObserver(directory / 'different', 600_000_000)
            base = FakeBase(a); model = h.build(base, 'repairability_pomdp', {}, parent(same), {})
            model.prepare_observer(a); model.prepare_observer(same)
            self.assertIs(a._model, same._model); self.assertIs(a._cache, same._cache)
            model = h.build(base, 'repairability_pomdp', {}, parent(different), {})
            model.prepare_observer(a); model.prepare_observer(different)
            self.assertIsNot(a._model, different._model)
            self.assertIsNot(a._cache, different._cache)

    def test_ledger_deduplicates_same_input_but_bills_different_assets(self):
        with TemporaryDirectory() as temporary:
            a = FakeObserver(Path(temporary) / 'a', 400_000_000)
            b = FakeObserver(Path(temporary) / 'b', 600_000_000)
            row = {'id': 'q', 'question': 'Public?', 'answer': 'SECRET'}
            tree = {'': {'answer': 'A'}}; ledger = {}
            h.record_observation(ledger, a, row, '', tree)
            h.record_observation(ledger, a, row, '', tree)
            self.assertEqual(len(ledger), 1)
            h.record_observation(ledger, b, row, '', tree)
            self.assertEqual(len(ledger), 2)
            self.assertAlmostEqual(sum(x['compute'] for x in ledger.values()), 13. / 3.09)

    def test_paid_replay_bills_fusion_root_and_distinct_mentor_passes(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            a = FakeObserver(folder / 'a', 400_000_000)
            b = FakeObserver(folder / 'b', 700_000_000)
            m = FakeObserver(folder / 'm', 600_000_000)
            model = h.build(FakeBase(FakeFusion([a, b])), 'continuous_repairability',
                            {'penalty': .08, 'posterior_weight': 1.}, parent(m), {})
            row = {'id': 'q', 'question': 'Public?', 'answer': 'B', 'subject': 'physics', 'domain': 'science'}
            def node(answer):
                return {'answer': answer, 'text': 'Final: ' + answer, 'input_tokens': 10, 'output_tokens': 5,
                        'prompt_sha256': answer}
            tree = {'': node('A'), 'guided': node('B'), 'guided/reuse': node('B')}
            judge = {'answer': 'B', 'text': 'Review', 'p_correct': .1, 'action': 'repair', 'parse_valid': True,
                     'verdict': {'solution': 'Review'}, 'input_tokens': 10, 'output_tokens': 6,
                     'physical_api_calls': 2, 'calls': [{'input_tokens': 5, 'output_tokens': 3}] * 2}
            reviews = {('q', ''): judge}
            item, ledger = replay_paid_inputs(lambda r, t, j: h.evaluate(r, t, j, model, {'penalty': .08, 'threshold': 0.}, 'C'),
                                               row, tree, reviews, 'C', model)
            self.assertEqual(item['solver_calls'], 3)
            self.assertEqual(item['judge_physical_calls'], 2)
            self.assertEqual(item['verifier_calls'], 4)
            self.assertEqual(len(ledger), 4)
            self.assertEqual(item['path'], 'guided/reuse')

    def test_subject_routing_and_selected_posterior_are_preserved(self):
        base = FakeBase(None)
        routed = SimpleNamespace(models={s: base for s in ('math', 'chemistry', 'physics')},
                                 settings={s: {'penalty': .02, 'posterior_weight': .5} for s in ('math', 'chemistry', 'physics')})
        child, settings = h.resolve_base(routed, 'science_fusion_pomdp', {}, 'physics')
        self.assertIs(child, base); self.assertEqual(settings['posterior_weight'], .5)
        with self.assertRaises(ValueError):
            h.resolve_base(routed, 'repairability_pomdp', {}, 'physics')
        transfer = parent(None).transfer['physics']
        np.testing.assert_allclose(transfer @ h.mentor_core.REWARD, h.action_core.CURRENT)

    def test_no_query_uses_base_blend_and_c_penalty_without_loading_mentor(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            a = FakeObserver(folder / 'a', 400_000_000)
            m = FakeObserver(folder / 'm', 600_000_000)
            model = h.build(FakeBase(a), 'continuous_repairability',
                            {'penalty': .02, 'posterior_weight': .5}, parent(m, query=False), {})
            row = {'id': 'q', 'question': 'Public?', 'answer': 'B', 'subject': 'physics', 'domain': 'science'}
            tree = {'': {'answer': 'B', 'text': 'Final: B', 'input_tokens': 10, 'output_tokens': 5}}
            def plain(safe, paid, judges, base, settings, family):
                self.assertEqual(safe['answer'], '')
                self.assertEqual(settings, {'penalty': .24, 'posterior_weight': .5})
                self.assertEqual(family, 'A')
                decisions = [{'path': '', 'action': 'stop', 'queried': False}]
                return h.mentor_core.outcome(safe, paid, {}, '', [''], [], decisions)
            with patch.object(h.continuous, 'evaluate', side_effect=plain):
                result = h.evaluate(row, tree, {}, model, {'penalty': .24}, 'C')
            self.assertIsNone(m._model)
            self.assertEqual(result['verifier_calls'], 1)
            self.assertEqual(result['judge_calls'], 0)
            self.assertEqual(result['correct'], 1)


if __name__ == '__main__':
    unittest.main()
