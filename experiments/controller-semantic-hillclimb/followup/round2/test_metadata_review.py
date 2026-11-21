import copy
import inspect
import pickle
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import fit_metadata_review as fitting
import metadata_review_hybrid as runtime
import paid_review_metadata as adapter_module
import review_metadata_diagnostic as diagnostic
from test_stronger_base_hybrid import FakeObserver, FakeBase, parent


def review():
    return {'answer': 'B', 'text': 'Review', 'p_correct': .1, 'action': 'repair', 'parse_valid': True,
            'json_complete': True, 'verdict': {'solution': 'However this is an assumption', 'check': 'Check'},
            'input_tokens': 10, 'output_tokens': 6, 'physical_api_calls': 2,
            'calls': [{'input_tokens': 5, 'output_tokens': 3, 'output_cap': 5}] * 2}


def make_adapter():
    records = []
    for i in range(30):
        features = np.zeros(32); features[0] = 1.; features[3] = 1.
        features[14] = i / 30; features[15] = .2
        records.append({'id': f'q{i}', 'features': features.tolist(), 'target': int(i > 14), 'answer_present': True})
    model = diagnostic.train_model('metadata_logistic', records)
    result = adapter_module.Adapter(model, 1., {})
    result.receipt = {'learned_state_sha256': adapter_module.digest(result.state())}
    return result


class MetadataReviewTest(unittest.TestCase):
    def test_paid_features_match_frozen_diagnostic_and_ignore_hidden_fields(self):
        row = {'subject': 'physics', 'answer': '', 'reference': object()}
        current = {'answer': 'A', 'text': 'Paid reasoning', 'secret': object()}; judge = review()
        judge['unpaid_branch'] = object(); judge['reference'] = object()
        entry = adapter_module.public_entry(row, current, judge)
        np.testing.assert_array_equal(diagnostic.features(row, current, judge, 'root'),
            diagnostic.features(entry['row'], entry['current'], entry['judge'], 'root'))
        self.assertNotIn('reference', json.dumps(entry))
        with self.assertRaisesRegex(ValueError, 'Blank the reference'):
            adapter_module.public_entry(dict(row, answer='SECRET'), current, judge)

    def test_adapter_finite_integrity_and_empty_answer(self):
        adapter = make_adapter(); judge = review()
        p, receipt = adapter.predict_paid({'subject': 'physics', 'answer': ''}, {'answer': 'A', 'text': 'Paid'}, judge)
        self.assertTrue(0 < p < 1); self.assertEqual(receipt['neural_calls'], 0)
        judge['answer'] = ''
        self.assertEqual(adapter.predict_paid({'subject': 'physics'}, {'answer': 'A', 'text': 'Paid'}, judge)[0], 1e-6)
        adapter.fitted['model']['model'].coef_[0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'Changed or nonfinite'):
            adapter.validate()

    def test_scalar_probability_uses_original_buckets_and_parse_support(self):
        judge = review(); current = {'answer': 'A'}
        for probability in (0., .2, .21, .5, .51, .8, .81, .97, 1.):
            expected = fitting.mentor.signal(judge, current, np.array([1.-probability, probability, 0., 0.]))
            self.assertEqual(adapter_module.signal(judge, current, probability), expected)
        judge['parse_valid'] = False
        self.assertEqual(adapter_module.signal(judge, current, .7), 30)
        judge['answer'] = ''
        self.assertEqual(adapter_module.signal(judge, current, .7), 31)

    def test_rebuild_changes_likelihood_and_conditional_transition(self):
        rows = [{'subject': 'physics', 'state': 0, 'signal': 0, 'first_states': [0, 2], 'review_cost': 100.},
                {'subject': 'physics', 'state': 0, 'signal': 2, 'first_states': [2, 0], 'review_cost': 200.}]
        first = fitting.rebuild(rows, ['physics', 'math'], 8.)
        changed = copy.deepcopy(rows); changed[1]['signal'] = 0
        second = fitting.rebuild(changed, ['physics', 'math'], 8.)
        self.assertFalse(np.array_equal(first['physics'][0], second['physics'][0]))
        self.assertFalse(np.array_equal(first['physics'][1], second['physics'][1]))
        np.testing.assert_array_equal(first['physics'][2], second['physics'][2])
        support, _ = fitting.support_and_mask()
        for review_likelihood, transition, costs in second.values():
            np.testing.assert_allclose(review_likelihood.sum(-1), 1.)
            np.testing.assert_allclose(transition.sum(-1), 1.)
            self.assertTrue(np.all(transition[..., support == 0] == 0))
            self.assertTrue(np.isfinite(costs).all())
            self.assertTrue(np.all(review_likelihood[1::2, 31] == 0))

    def test_cost_removal_preserves_unseen_fallback(self):
        rows = [{'subject': 'physics', 'state': 0, 'signal': 0, 'first_states': [0, 2], 'review_cost': 100.}]
        plain = fitting.rebuild(rows, ['physics', 'math'], 8.)
        semantic = fitting.rebuild(rows, ['physics', 'math'], 8., encoder_price=70.)
        for subject in plain:
            self.assertAlmostEqual(semantic[subject][2][0] - plain[subject][2][0], 70.)
            self.assertEqual(plain[subject][2][1], 2800.)
            self.assertEqual(semantic[subject][2][1], 2800.)
        self.assertEqual(plain['math'][2][0], 100.)

    def test_root_selection_does_not_count_auxiliary_siblings(self):
        rows, trees, judges, records, probabilities = [], {}, {}, [], []
        for i in range(128):
            row = {'id': f'q{i}', 'answer': 'B', 'subject': 'physics'}; rows.append(row)
            tree = {path: {'answer': 'A'} for path in fitting.mentor.core.FIXED}
            tree['guided'] = {'answer': 'B'}; trees[row['id']] = tree
            judge = review(); judges[row['id'], ''] = judge
            for protocol in ('root', 'critical', 'independent'):
                records.append({'id': row['id'], 'protocol': protocol, 'bank': 'new', 'target': 1,
                    'review_sha256': fitting.fingerprint(judge), 'current_sha256': fitting.fingerprint(tree[''])})
                probabilities.append(.7 if protocol == 'root' else .01)
        root = fitting.root_records(rows, trees, judges, records, probabilities, 'newsource')
        self.assertEqual(len(root), 128)
        self.assertTrue(all(r['nested_probability'] == .7 for r in root))
        changed = copy.deepcopy(records); changed[0]['review_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'paid response changed'):
            fitting.root_records(rows, trees, judges, changed, probabilities, 'newsource')

    def test_runtime_skips_review_encoder_but_bills_child_and_two_phases(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary); a = FakeObserver(folder / 'a', 400_000_000)
            m = FakeObserver(folder / 'm', 600_000_000)
            original = parent(m); original.base = FakeBase(a)
            model = runtime.build(original, make_adapter(), {'penalty': .08}, {})
            row = {'id': 'q', 'question': 'Public?', 'answer': 'B', 'subject': 'physics', 'domain': 'science'}
            def node(answer):
                return {'answer': answer, 'text': 'Final: ' + answer, 'input_tokens': 10, 'output_tokens': 5}
            tree = {'': node('A'), 'guided': node('B'), 'guided/reuse': node('B')}
            raw_values = m.values
            def no_review(*args):
                if len(args) > 3 and args[3] is not None:
                    raise AssertionError('Review neural inference forbidden')
                return raw_values(*args)
            with patch.object(m, 'values', side_effect=no_review):
                item = runtime.evaluate(row, tree, {('q', ''): review()}, model, {'penalty': .08}, 'C')
            self.assertEqual(item['path'], 'guided/reuse'); self.assertEqual(item['solver_calls'], 3)
            self.assertEqual(item['judge_physical_calls'], 2); self.assertEqual(item['judge_calls'], 1)
            self.assertEqual(item['verifier_calls'], 2); self.assertEqual(item['metadata_calls'], 1)
            self.assertEqual(item['review_neural_calls'], 0)
            self.assertTrue(all(r['kind'] == 'solver' for r in item['verifier_components']))
            expected = (4 * 45 + 3 * 16 + .4 * 13 + .6 * 13) / 3.09
            self.assertAlmostEqual(item['compute'], expected)
            self.assertTrue(item['metadata_cpu_seconds'] >= 0)

    def test_no_query_does_not_inspect_or_bill_metadata(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary); a = FakeObserver(folder / 'a', 400_000_000)
            m = FakeObserver(folder / 'm', 600_000_000)
            original = parent(m, query=False); original.base = FakeBase(a)
            model = runtime.build(original, make_adapter(), {'penalty': .08}, {})
            row = {'id': 'q', 'question': 'Public?', 'answer': 'A', 'subject': 'physics', 'domain': 'science'}
            tree = {'': {'answer': 'A', 'text': 'Paid', 'input_tokens': 10, 'output_tokens': 5}}
            def plain(safe, tree, judges, base, settings, family):
                return runtime.mentor_core.outcome(safe, tree, {}, '', [''], [], [{'path': '', 'action': 'stop', 'queried': False}])
            with patch.object(model.metadata_adapter, 'predict_paid', side_effect=AssertionError('Unpaid review')):
                with patch.object(runtime.action_core, 'evaluate', side_effect=plain):
                    result = runtime.evaluate(row, tree, {}, model, {'penalty': .08}, 'C')
            self.assertEqual(result['metadata_calls'], 0); self.assertEqual(result['judge_physical_calls'], 0)
            self.assertEqual(result['correct'], 1)

    def test_missing_review_raises_before_metadata_and_requires_explicit_dispatch(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary); a = FakeObserver(folder / 'a', 400_000_000)
            original = parent(a); original.base = FakeBase(a)
            model = runtime.build(original, make_adapter(), {}, {})
            row = {'id': 'q', 'question': 'Public?', 'subject': 'physics', 'answer': 'SECRET'}
            tree = {'': {'answer': 'A'}}
            with self.assertRaises(runtime.action_core.MissingObservation):
                runtime.evaluate(row, tree, {}, model, {'penalty': .08}, 'C')
            model.fit_receipt['policy_module'] = 'stronger_base_hybrid'
            with self.assertRaisesRegex(ValueError, 'explicit'):
                runtime.evaluate(row, tree, {}, model, {'penalty': .08}, 'C')

    def test_no_neural_fit_guard_and_protected_state_digest(self):
        with fitting.no_neural_inference():
            observer = fitting.mentor.perception.Observer('unused', 1.)
            with self.assertRaises(AssertionError):
                observer.load()
        model = SimpleNamespace(base={'value': 1}, transfer=np.eye(8), mentor=SimpleNamespace(
            classifier={'unchanged': True}, regular={'calibration': [1., 2.]},
            dynamics={'physics': SimpleNamespace(review=np.ones(2), reviewed_transition=np.ones(2),
                       review_costs=np.ones(2), emission=np.ones(2), first_costs=np.ones(2))}))
        before = fitting.preserved_digest(model)
        self.assertEqual(before, fitting.preserved_digest(pickle.loads(pickle.dumps(model))))
        model.transfer = np.asfortranarray(model.transfer)
        self.assertEqual(before, fitting.preserved_digest(model))
        model.mentor.dynamics['physics'].review *= 3
        self.assertEqual(before, fitting.preserved_digest(model))
        model.mentor.dynamics['physics'].first_costs *= 3
        self.assertNotEqual(before, fitting.preserved_digest(model))


if __name__ == '__main__':
    unittest.main()
