import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import metadata_paid_audit as audit
import metadata_review_hybrid as runtime
from test_metadata_review import make_adapter, review
from test_stronger_base_hybrid import FakeObserver, FakeBase, parent


def node(answer):
    return {'answer': answer, 'text': 'Final: '+answer, 'input_tokens': 10, 'output_tokens': 5}


class MetadataPaidAuditTest(unittest.TestCase):
    def test_frozen_linear_arithmetic_matches_original_adapter_across_temperatures(self):
        adapter = make_adapter(); row = {'id': 'q', 'answer': '', 'subject': 'physics'}
        for temperature in (.7, 1., 1.5, 2.):
            adapter.temperature = temperature
            adapter.receipt['learned_state_sha256'] = audit.digest(adapter.state())
            for answer in ('B', ''):
                judge = {**review(), 'answer': answer}
                probability, observed = adapter.predict_paid(row, node('A'), judge)
                expected = audit.expected_observation(adapter.state(), adapter.receipt, 'physics', node('A'), judge)
                audit.verify_observation(observed, expected)
                self.assertAlmostEqual(expected['probability'], probability, places=12)
        state = copy.deepcopy(adapter.state()); state['coefficients'][0][0] += .1
        with self.assertRaisesRegex(ValueError, 'identity'):
            audit.validate_adapter_state(state, adapter.receipt)

    def test_capture_checks_exact_paid_prefix_reference_and_same_attempt(self):
        adapter = make_adapter(); safe = {'id': 'q', 'answer': '', 'subject': 'physics'}
        root, judge = node('A'), review()
        with audit.capture(adapter, safe, {'': root}, {('q', ''): judge}) as captured:
            _, receipt = adapter.predict_paid(safe, root, judge)
            self.assertEqual(captured, [receipt])
            with self.assertRaisesRegex(ValueError, 'nonpaid'):
                adapter.predict_paid({**safe, 'answer': 'SECRET'}, root, judge)
            with self.assertRaisesRegex(ValueError, 'nonpaid'):
                adapter.predict_paid(safe, node('CHANGED'), judge)
            with self.assertRaisesRegex(ValueError, 'nonpaid'):
                adapter.predict_paid(safe, root, {**judge, 'answer': 'A'})

    def fixture(self, directory):
        base_observer = FakeObserver(directory/'base', 400_000_000)
        mentor_observer = FakeObserver(directory/'mentor', 600_000_000)
        inherited = parent(mentor_observer); inherited.base = FakeBase(base_observer)
        model = runtime.build(inherited, make_adapter(), {'penalty': .08}, {})
        row = {'id': 'q', 'question': 'Public?', 'answer': 'SECRET', 'subject': 'physics', 'domain': 'science'}
        tree = {'': node('A'), 'guided': node('B'), 'guided/reuse': node('B')}
        reviews = {('q', ''): review()}
        return model, row, tree, reviews

    def test_real_runtime_replay_keeps_root_child_charges_and_no_review_neural_pass(self):
        with TemporaryDirectory() as temporary:
            model, row, tree, reviews = self.fixture(Path(temporary))
            fn = lambda safe, paid, judges: runtime.evaluate(safe, paid, judges, model, {'penalty': .08}, 'C')
            item, encoders = audit.replay_paid_inputs(fn, row, tree, reviews, 'C', model)
            self.assertNotIn('correct', item)
            self.assertEqual(item['metadata_calls'], 1); self.assertEqual(item['review_neural_calls'], 0)
            self.assertEqual({v['path'] for v in encoders}, {'', 'guided'})
            self.assertTrue(all(v['kind'] == 'solver' for v in encoders))
            self.assertEqual(item['judge_physical_calls'], 2)
            state, receipt = model.metadata_adapter.state(), model.metadata_adapter.receipt
            result = audit.audit_metadata(item, tree, reviews, encoders, state, receipt, 'physics')
            self.assertEqual(result['metadata_ledger_sha256'], audit.digest(item['metadata_observations']))
            for mutation in ('probability', 'cpu_seconds', 'input_sha256', 'neural_calls'):
                altered = copy.deepcopy(item)
                altered['metadata_observations'][0][mutation] = {'probability': .999, 'cpu_seconds': -1.,
                                                               'input_sha256': 'changed', 'neural_calls': 1}[mutation]
                with self.assertRaises(ValueError): audit.audit_metadata(altered, tree, reviews, encoders, state, receipt, 'physics')
            altered = copy.deepcopy(item); altered['metadata_cpu_seconds'] += 1.
            with self.assertRaises(ValueError): audit.audit_metadata(altered, tree, reviews, encoders, state, receipt, 'physics')
            changed_encoder = copy.deepcopy(encoders); changed_encoder[0]['kind'] = 'review'
            with self.assertRaisesRegex(ValueError, 'review neural'):
                audit.audit_metadata(item, tree, reviews, changed_encoder, state, receipt, 'physics')
            with self.assertRaisesRegex(ValueError, 'root/child'):
                audit.audit_metadata(item, tree, reviews, encoders[:1], state, receipt, 'physics')

    def test_skip_review_requires_zero_metadata_work(self):
        adapter = make_adapter()
        item = {'path': '', 'decisions': [{'path': '', 'queried': False}], 'metadata_observations': [],
                'metadata_calls': 0, 'metadata_cpu_seconds': 0., 'metadata_ledger_sha256': audit.digest([]),
                'review_neural_calls': 0, 'include_reviews': False}
        encoders = [{'kind': 'solver', 'path': ''}]
        result = audit.audit_metadata(item, {'': node('A')}, {}, encoders, adapter.state(), adapter.receipt, 'physics')
        self.assertEqual(result['metadata_calls'], 0)
        changed = copy.deepcopy(item); changed['metadata_calls'] = 1
        with self.assertRaises(ValueError):
            audit.audit_metadata(changed, {'': node('A')}, {}, encoders, adapter.state(), adapter.receipt, 'physics')

    def test_runtime_cannot_fabricate_cpu_receipt_or_skip_adapter(self):
        with TemporaryDirectory() as temporary:
            model, row, tree, reviews = self.fixture(Path(temporary))
            def fn(safe, paid, judges):
                result = runtime.evaluate(safe, paid, judges, model, {'penalty': .08}, 'C')
                result['metadata_observations'][0]['cpu_seconds'] += 1.
                return result
            with self.assertRaisesRegex(ValueError, 'same-attempt'):
                audit.replay_paid_inputs(fn, row, tree, reviews, 'C', model)
            original = audit.original.replay_paid_inputs
            with patch.object(audit.original, 'replay_paid_inputs', return_value=('original', [])) as delegate:
                self.assertEqual(audit.replay_paid_inputs(None, {}, {}, {}, 'baseline'), ('original', []))
                delegate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
