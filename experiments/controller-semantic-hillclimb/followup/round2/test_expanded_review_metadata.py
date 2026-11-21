"""Synthetic CPU-only calibration and whole-question exclusion checks."""
import copy
from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

import expanded_review_metadata as fit
import expanded_mentor_spec as spec


def fixture():
    rng = np.random.default_rng(29)
    records = []
    for subject, n in spec.REVIEW_COUNTS.items():
        for i in range(n):
            qid = subject + str(i)
            protocols = ['root']
            if subject != 'math' and i < 32:
                protocols += ['critical', 'independent']
            for j, protocol in enumerate(protocols):
                values = rng.normal(size=32)
                target = int((i + j) % 3 != 0)
                records.append({'id': qid, 'subject': subject, 'bank': 'new', 'protocol': protocol,
                    'features': values.tolist(), 'target': target, 'answer_present': True,
                    'current_sha256': qid + '-current', 'review_sha256': qid + '-' + protocol})
    return records


class ExpandedMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = fixture()
        cls.result = fit.fit(cls.records, 'synthetic-registration')

    def test_complete_448_newsource_contract_and_weight_mass(self):
        fit.validate(self.records)
        w = self.result['receipt']['weights']
        self.assertEqual((w['records'], w['questions'], w['total_mass']), (448, 320, 320.))
        self.assertEqual(w['subject_mass'], {'math': 160., 'chemistry': 80., 'physics': 80.})
        for field, value in [('bank', 'old'), ('subject', 'biology')]:
            changed = copy.deepcopy(self.records); changed[0][field] = value
            with self.assertRaises(ValueError): fit.validate(changed)
        with self.assertRaises(ValueError): fit.validate(self.records[:-1])

    def test_all_siblings_grouped_and_outer_held_absent_from_temperature(self):
        for outer in self.result['receipt']['outer']:
            held, training = set(outer['held_ids']), set(outer['training_ids'])
            self.assertFalse(held & training)
            self.assertEqual(held | training, {r['id'] for r in self.records})
            for inner in outer['inner']:
                a, b = set(inner['training_ids']), set(inner['held_ids'])
                self.assertFalse(a & b)
                self.assertFalse((a | b) & held)
                self.assertEqual(a | b, training)
                self.assertEqual(inner['fit_weights']['questions'], len(a))
                self.assertAlmostEqual(inner['fit_weights']['total_mass'], len(a))

    def test_own_held_labels_do_not_change_predictions_or_temperature(self):
        changed = copy.deepcopy(self.records)
        for record in changed:
            if fit.inherited.fold(record['id']) == 0:
                record['target'] = 1 - record['target']
        first = fit.outer_prediction(self.records, 0)
        second = fit.outer_prediction(changed, 0)
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertEqual(first[-1], second[-1])

    def test_adapter_exact_feature_and_runtime_contract(self):
        adapter = self.result['adapter']
        adapter.validate()
        self.assertEqual(adapter.receipt['training_questions'], 320)
        self.assertEqual(adapter.receipt['training_records'], 448)
        self.assertEqual(adapter.state()['features'], fit.features.FEATURES)
        self.assertEqual(adapter.state()['classes'], [0, 1])
        self.assertIn(adapter.temperature, fit.features.TEMPERATURES)
        self.assertFalse(self.result['receipt']['root_tables_use_final_temperature'])
        self.assertTrue(np.isfinite(self.result['raw']).all())
        self.assertTrue(np.isfinite(self.result['nested']).all())

    def test_question_sibling_weight_has_no_science_row_multiplier(self):
        weights = spec.question_weights(self.records)
        sums = Counter()
        for record, value in zip(self.records, weights): sums[record['id']] += value
        for record in self.records:
            expected = 2.5 if record['subject'] == 'math' else .625
            self.assertAlmostEqual(sums[record['id']], expected)
        repeated = [r for r in self.records for _ in range(3 if r['id'] == 'math0' else 1)]
        original, duplicated = fit.train_model(self.records), fit.train_model(repeated)
        np.testing.assert_allclose(original['model']['scale'].mean_, duplicated['model']['scale'].mean_, atol=1e-12)
        np.testing.assert_allclose(original['model']['model'].coef_, duplicated['model']['model'].coef_, atol=1e-12)


if __name__ == '__main__': unittest.main()
