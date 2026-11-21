"""Synthetic training-only checks; no model downloads or dataset reads."""
import copy
from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

import expanded_mentor_spec as spec
import expanded_mentor_core as fit


class FakeObserver:
    temperature = 1.
    calibration = None

    def calibration_features(self, row, path, logits):
        if row.get('answer'):
            raise AssertionError('Reference leaked into features')
        return [*logits, float(bool(path)), *[float(row['subject'] == s) for s in spec.SUBJECTS]]


def fixture():
    rows, trees, judges, metadata, states = [], {}, {}, [], []
    rng = np.random.default_rng(7)
    for subject, count in spec.PLAIN_COUNTS.items():
        for i in range(count):
            qid = subject + str(i)
            rows.append({'id': qid, 'subject': subject, 'answer': '1'})
            paths = list(fit.core.PLAIN)
            paired = subject != 'math' or i < 64
            if paired:
                paths += ['guided', 'guided/reuse', 'guided/retry', 'retry/reuse']
            tree = {}
            for j, path in enumerate(paths):
                answer = ('1' if i % 2 else '0') if not path else ('1' if i % 4 < 2 else '0')
                if path == 'guided':
                    answer = '1' if i % 3 else '0'
                tree[path] = {'answer': answer, 'text': 'paid answer ' + answer,
                    'prompt_sha256': qid + ':' + path, 'input_tokens': 100+j, 'output_tokens': 20+j}
            trees[qid] = tree
            hard = int(np.mean([fit.correct(tree[p], '1') for p in fit.core.FIXED]) < .6)
            for path, node in tree.items():
                metadata.append({'id': qid, 'kind': 'solver', 'path': path, 'fold': 'full',
                    'train': path in spec.CALIBRATION_PATHS, 'valid': bool(node['answer']),
                    'label': 2*hard + int(fit.correct(node, '1')), 'prompt_sha256': node['prompt_sha256'],
                    'input_sha256': qid + path})
                states.append(rng.normal(size=4))
            if paired:
                judge = {'answer': '1' if i % 3 else '0', 'input_tokens': 800, 'output_tokens': 600,
                         'parse_valid': True, 'action': 'repair', 'p_correct': .5}
                judges[qid, ''] = judge
                metadata.append({'id': qid, 'kind': 'original-review', 'path': '', 'fold': 'full',
                    'train': False, 'valid': True, 'label': 2*hard + int(fit.correct(judge, '1')),
                    'prompt_sha256': qid + ':review', 'input_sha256': qid + 'review'})
                states.append(rng.normal(size=4))
    return rows, trees, judges, metadata, {'state': np.asarray(states)}


class ExpandedCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixture()
        cls.observer, cls.oof, cls.receipt, cls.records, cls.probabilities = fit.fit_observer(
            cls.fixture[0], cls.fixture[1], cls.fixture[3], cls.fixture[4], FakeObserver())

    def test_question_weight_mass_and_sibling_invariance(self):
        records = [{'id': s + str(i), 'subject': s} for s in spec.SUBJECTS for i in range(2)]
        repeated = [r for r in records for _ in range(7 if r['subject'] == 'physics' else 1)]
        weights = spec.question_weights(repeated)
        receipt = spec.weight_receipt(repeated, weights)
        self.assertEqual(receipt['total_mass'], 6.)
        self.assertEqual(receipt['subject_mass'], {'math': 3., 'chemistry': 1.5, 'physics': 1.5})
        original = dict(zip((r['id'] for r in records), spec.question_weights(records)))
        for qid in original:
            self.assertAlmostEqual(sum(w for r, w in zip(repeated, weights) if r['id'] == qid), original[qid])
        with self.assertRaises(ValueError):
            spec.question_weights([{'id': 'same', 'subject': 'math'}, {'id': 'same', 'subject': 'physics'}])

    def test_complete_calibration_shape_folds_and_reference_blinding(self):
        self.assertEqual(len(self.records), 1856)
        self.assertEqual(self.receipt['weights']['questions'], 512)
        self.assertEqual(self.receipt['weights']['total_mass'], 512.)
        self.assertEqual(self.receipt['terminal_rows_fit'], 0)
        self.assertEqual(self.receipt['weights']['subject_mass'], {'math': 256., 'chemistry': 128., 'physics': 128.})
        for receipt in self.receipt['folds']:
            self.assertFalse(set(receipt['training_ids']) & set(receipt['held_ids']))
            self.assertTrue(all(spec.calibration_fold(q) != receipt['fold'] for q in receipt['training_ids']))
            self.assertAlmostEqual(receipt['weights']['total_mass'], len(receipt['training_ids']))
        self.assertTrue(np.allclose(self.probabilities.sum(1), 1.))

    def test_terminal_logits_and_review_logits_cannot_fit_child_classifier(self):
        rows, trees, _, metadata, outputs = self.fixture
        changed = {'state': outputs['state'].copy()}
        for i, r in enumerate(metadata):
            if r['kind'] != 'solver' or r['path'] not in spec.CALIBRATION_PATHS:
                changed['state'][i] += 10000
        _, _, receipt, _, probabilities = fit.fit_observer(rows, trees, metadata, changed, FakeObserver())
        self.assertEqual(receipt, self.receipt)
        np.testing.assert_array_equal(probabilities, self.probabilities)

    def test_held_question_labels_do_not_change_own_forecast(self):
        rows, trees, _, metadata, outputs = copy.deepcopy(self.fixture)
        held_id = rows[0]['id']; fold = spec.calibration_fold(held_id)
        rows[0]['answer'] = '9'
        for r in metadata:
            if r['id'] == held_id:
                r['label'] = 2
        _, _, receipt, records, probabilities = fit.fit_observer(rows, trees, metadata, outputs, FakeObserver())
        self.assertEqual(receipt['folds'][fold], self.receipt['folds'][fold])
        mask = [spec.calibration_fold(r['id']) == fold for r in records]
        np.testing.assert_array_equal(probabilities[mask], self.probabilities[mask])

    def test_unreviewed_tables_ignore_guided_and_all_terminal_forecasts(self):
        rows, trees, _, _, _ = self.fixture
        regular = fit.plain_regular(rows, trees, self.observer, self.oof, 50., self.receipt)
        changed_trees = copy.deepcopy(trees)
        changed_oof = dict(self.oof)
        for row in rows:
            if 'guided' in changed_trees[row['id']]:
                changed_trees[row['id']]['guided']['answer'] = '999'
                changed_oof[row['id'], 'guided'] = np.asarray([0., 1., 0., 0.])
        changed = fit.plain_regular(rows, changed_trees, self.observer, changed_oof, 50., self.receipt)
        for subject in spec.SUBJECTS:
            for field in ('prior', 'initial', 'transition', 'emission', 'costs'):
                np.testing.assert_array_equal(getattr(regular.dynamics[subject], field),
                                              getattr(changed.dynamics[subject], field))
        self.assertFalse(regular.fit_receipt['guided_in_unreviewed_counts'])

    def test_pooled_math4_local1_and_final_cost_has_no_encoder(self):
        rows, trees, judges, _, _ = self.fixture
        signals = {qid: 0 for qid, path in judges}
        records = fit.paired_records(rows, trees, judges, self.oof, 50., signals, False)
        pooled, local = fit.paired_counts(records, True), fit.paired_counts(records)
        empty = fit.paired_counts([])
        self.assertEqual((pooled[0] - empty[0]).sum(), 512.)
        self.assertEqual((local[0] - empty[0]).sum(), 320.)
        self.assertEqual((pooled[2] - empty[2]).sum(), 1024.)
        self.assertEqual((local[2] - empty[2]).sum(), 640.)
        changed = fit.paired_records(rows, trees, judges, self.oof, 150., signals, False)
        for a, b in zip(records, changed):
            self.assertEqual(a['ends'], b['ends'])
            self.assertEqual(a['review_cost'], b['review_cost'])
            self.assertAlmostEqual(b['first'][0][2] - a['first'][0][2], 100.)

    def test_full_joint_fit_preserves_truth_and_question_coverage(self):
        rows, trees, judges, metadata, outputs = self.fixture
        regular = fit.plain_regular(rows, trees, self.observer, self.oof, 50., self.receipt)
        with patch.object(fit.perception.Observer, 'load', side_effect=AssertionError('No NN allowed')):
            fitted, records = fit.fit_mentor(rows, trees, judges, regular, self.oof, metadata, outputs,
                50., smoothing=8., review_signals={qid: 0 for qid, path in judges})
            transfer = fit.fit_transfer(rows, trees, judges, 8.)
        self.assertEqual(len(records), 320)
        self.assertIsNone(fitted.review_observer)
        support = fit.masks()[0]
        for subject in spec.SUBJECTS:
            d = fitted.dynamics[subject]
            np.testing.assert_allclose(d.reviewed_transition.sum(-1), 1.)
            np.testing.assert_allclose(d.emission.sum(-1), 1.)
            self.assertTrue(np.all(d.last_transition[..., support == 0] == 0))
            np.testing.assert_allclose(transfer[subject] @ fit.mentor.REWARD, fit.hybrid.action_core.CURRENT)
        with self.assertRaises(ValueError):
            fit.fit_mentor(rows, trees, dict(list(judges.items())[:-1]), regular, self.oof,
                           metadata, outputs, 50.)


if __name__ == '__main__':
    unittest.main()
