import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import review_metadata_diagnostic as diagnostic
import review_metadata_spec as spec


class ReviewMetadataTest(unittest.TestCase):
    def paid(self):
        row = {'id': 'synthetic', 'subject': 'chemistry', 'answer': 'SECRET_REFERENCE', 'unpaid': 'FUTURE'}
        current = {'answer': 'A', 'text': 'Current paid solution. Final: A'}
        calls = [{'input_tokens': 100, 'output_tokens': 64, 'output_cap': 64, 'finish': 'length'},
                 {'input_tokens': 150, 'output_tokens': 24, 'output_cap': 128, 'finish': 'stop'}]
        judge = {'answer': 'B', 'action': 'repair', 'parse_valid': True, 'json_complete': True,
                 'p_correct': .2, 'physical_api_calls': 2, 'calls': calls,
                 'input_tokens': 250, 'output_tokens': 88,
                 'verdict': {'solution': 'However this assumption may be incorrect. It is unclear.', 'check': 'Check units.'}}
        return row, current, judge

    def test_features_use_only_paid_public_values(self):
        row, current, judge = self.paid()
        before = diagnostic.features(row, current, judge, 'root')
        changed = {**row, 'answer': 'OTHER_REFERENCE', 'unpaid': 'OTHER_FUTURE', 'correct': 1}
        np.testing.assert_array_equal(before, diagnostic.features(changed, current, judge, 'root'))
        self.assertEqual(len(before), len(spec.FEATURES))
        self.assertEqual(before[spec.FEATURES.index('first_at_cap')], 1.)
        self.assertEqual(before[spec.FEATURES.index('last_at_cap')], 0.)
        self.assertGreater(before[spec.FEATURES.index('error_markers_per100')], 0.)
        altered = copy.deepcopy(judge); altered['output_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'totals disagree'):
            diagnostic.features(row, current, altered, 'root')

    def test_siblings_share_fold_and_question_weight(self):
        records = [{'id': 'a', 'protocol': name} for name in ('root', 'critical', 'independent')]
        records.append({'id': 'b', 'protocol': 'root'})
        np.testing.assert_allclose(diagnostic.weights(records), [1/3, 1/3, 1/3, 1.])
        self.assertEqual(len({diagnostic.fold(row['id']) for row in records[:3]}), 1)

    def records(self, n=90):
        result = []
        random = np.random.default_rng(3)
        for i in range(n):
            for protocol in ('root', 'critical', 'independent'):
                target = int((i + (protocol == 'critical')) % 3 != 0)
                values = random.normal(size=len(spec.FEATURES))
                values[spec.FEATURES.index('p_current')] = .8 if target else .2
                result.append({'id': f'q-{i}', 'subject': ('math', 'chemistry', 'physics')[i % 3],
                    'bank': 'new', 'protocol': protocol, 'target': target,
                    'answer_present': True, 'features': values.tolist()})
        return result

    def test_outer_held_labels_cannot_change_classifier_or_temperature(self):
        records = self.records()
        selected = 0
        held, probability, _, receipt = diagnostic.outer_prediction('confidence_agreement_logistic', records, selected)
        changed = copy.deepcopy(records)
        for row in changed:
            if diagnostic.fold(row['id']) == selected:
                row['target'] = 1 - row['target']
        other_held, other_probability, _, other_receipt = diagnostic.outer_prediction(
            'confidence_agreement_logistic', changed, selected)
        self.assertEqual([r['id'] for r in held], [r['id'] for r in other_held])
        np.testing.assert_array_equal(probability, other_probability)
        self.assertEqual(receipt['temperature_search'], other_receipt['temperature_search'])
        held_ids = set(receipt['held_ids'])
        self.assertFalse(held_ids & set(receipt['training_ids']))
        for inner in receipt['inner']:
            self.assertFalse(set(inner['training_ids']) & set(inner['held_ids']))
            self.assertFalse(held_ids & (set(inner['training_ids']) | set(inner['held_ids'])))

    def test_all_fixed_model_families_produce_valid_probabilities(self):
        records = self.records(30)
        for name in spec.MODELS:
            model = diagnostic.train_model(name, records)
            values = diagnostic.predict(model, records)
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(((values > 0) & (values < 1)).all())
        uniform = [dict(row, target=1) for row in records]
        model = diagnostic.train_model('metadata_logistic', uniform)
        self.assertTrue(model['constant'])

    def test_invalid_answer_is_structural_and_not_dropped(self):
        records = self.records(3)
        records[0]['answer_present'] = False; records[0]['target'] = 0
        values = diagnostic.calibrated(np.full(len(records), .9), 3., records)
        self.assertEqual(len(values), len(records))
        self.assertEqual(values[0], 1e-6)

    def test_primary_panels_and_metrics_do_not_mix_extra_siblings(self):
        records = self.records(6)
        selected = [r for r in records if diagnostic.panel(r, 'new_science_root')]
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(r['protocol'] == 'root' and r['subject'] != 'math' for r in selected))
        probabilities = np.asarray([.8 if r['target'] else .2 for r in selected])
        result = diagnostic.metrics(selected, probabilities)
        self.assertAlmostEqual(result['brier'], .04)
        comparison = diagnostic.comparisons(selected, {'subject_protocol_prior': np.full(4, .5),
            'confidence_agreement_logistic': np.full(4, .5), 'metadata_logistic': probabilities})
        self.assertLess(comparison['metadata_logistic_minus_subject_protocol_prior']['brier']['ci95'][1], 0)

    def test_missing_full_independent_completion_blocks_all_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            registration = {'specification_sha256': diagnostic.sha(spec.__file__),
                'features': spec.FEATURES, 'models': spec.MODELS, 'temperature_grid': spec.TEMPERATURES,
                'source_directories': {'independent': str(folder / 'unfinished')}}
            (folder / 'registration.json').write_text(json.dumps(registration))
            with mock.patch.object(diagnostic, 'load_records', side_effect=AssertionError('Must not read labels')):
                with self.assertRaisesRegex(ValueError, 'Complete every independent pair'):
                    diagnostic.run(SimpleNamespace(output=folder, wait=False))

    def test_novelty_counts_only_success_beyond_all_prior_paths(self):
        import review_path_novelty as novelty
        rows = [{'id': f'q{i}', 'subject': ('chemistry', 'physics')[i], 'answer': 'A'} for i in range(2)]
        trees = {row['id']: {path: {'answer': 'B'} for path in novelty.PLAIN} for row in rows}
        trees['q1']['']['answer'] = 'A'
        strong = {(row['id'], 'solver', path): {'answer': 'B'} for row in rows
                  for path in ('guided', 'guided/reuse', 'retry/reuse')}
        earlier = {(row['id'], 'solver'): {'answer': 'B'} for row in rows}
        new = {(row['id'], 'solver'): {'answer': 'A'} for row in rows}
        result = novelty.calculate(rows, trees, strong, earlier, new)
        self.assertEqual(result['all']['questions'], 2)
        self.assertEqual(result['all']['before_union'], 1)
        self.assertEqual(result['all']['independent_correct'], 2)
        self.assertEqual(result['all']['novel_success'], 1)
        self.assertEqual(result['all']['after_union'], 2)


if __name__ == '__main__':
    unittest.main()
