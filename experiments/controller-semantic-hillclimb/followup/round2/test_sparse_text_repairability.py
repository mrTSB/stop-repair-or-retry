import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sparse_text_repairability as s
import training_sparse_text as driver


def synthetic(n=36):
    ids, texts, numeric, labels, paths, subjects = [], [], [], [], [], []
    rng = np.random.default_rng(71)
    for i in range(n):
        for j, path in enumerate(s.PATHS):
            ids.append(f'question-{i}'); paths.append(path)
            texts.append(f'Public Co CO molecule solution pathway {i % 7} stage {j}')
            numeric.append(rng.normal(size=7)); labels.append((i + 2*j) % 8)
            subjects.append(('math', 'chemistry', 'physics')[i % 3])
    return {'ids': np.asarray(ids), 'texts': texts, 'numeric': np.asarray(numeric),
            'labels': np.asarray(labels), 'paths': np.asarray(paths), 'subjects': np.asarray(subjects)}


class SparseTextTest(unittest.TestCase):
    def test_paid_text_whitelist_and_case_preservation(self):
        row = {'id': 'SECRET_ID', 'answer': 'SECRET_REFERENCE', 'question': 'Compare Co and CO', 'unpaid': 'SECRET'}
        tree = {'': {'text': 'Paid root'}, 'repair': {'text': 'Paid current'}, 'retry': {'text': 'FORBIDDEN_FUTURE'}}
        text = s.paid_text(row, 'repair', tree)
        self.assertEqual(text, s.paid_text({**row, 'id': 'OTHER_ID', 'answer': 'OTHER_REFERENCE'}, 'repair',
                                         {**tree, 'retry': {'text': 'CHANGED_FUTURE'}}))
        self.assertNotIn('SECRET', text); self.assertNotIn('FORBIDDEN', text)
        self.assertEqual(s.paid_text(row, '', tree).count('Paid root'), 1)
        for name in s.VARIANTS[1:]:
            vectorizer = s.vectorizer(name).fit([text])
            self.assertFalse(vectorizer.lowercase)
            keys = set(vectorizer.vocabulary_)
            if name == 'word12_C1':
                self.assertTrue({'Co', 'CO'} <= keys)
            else:
                self.assertTrue(any('Co' in key for key in keys))
                self.assertTrue(any('CO' in key for key in keys))

    def test_numeric_control_matches_existing_standardized_C1(self):
        data = synthetic()
        model = s.fit_forecaster(data['texts'], data['numeric'], data['labels'], 'numeric_C1')
        reference = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000)).fit(data['numeric'], data['labels'])
        expected = np.full((len(data['labels']), 8), 1e-9)
        expected[:, reference.classes_.astype(int)] += reference.predict_proba(data['numeric'])
        expected /= expected.sum(1, keepdims=True)
        np.testing.assert_allclose(model.predict_proba(data['texts'], data['numeric']), expected, atol=1e-7)

    def test_held_question_labels_cannot_change_forecast_or_preprocessing(self):
        data = synthetic(); changed = copy.deepcopy(data)
        held = np.asarray([s.fold_for(qid) == 0 for qid in data['ids']])
        changed['labels'][held] = (changed['labels'][held] + 1) % 8
        for name in s.VARIANTS:
            indices, probability, model, receipt = s.outer_prediction(data, name, 0)
            other_indices, other_probability, other, other_receipt = s.outer_prediction(changed, name, 0)
            np.testing.assert_array_equal(indices, other_indices)
            np.testing.assert_array_equal(probability, other_probability)
            np.testing.assert_array_equal(model.classifier.coef_, other.classifier.coef_)
            for key in ('vocabulary_sha256', 'idf_sha256', 'scaler_mean_sha256', 'scaler_scale_sha256'):
                self.assertEqual(receipt[key], other_receipt[key])
            self.assertEqual(receipt['temperature'], 1.)
            self.assertFalse(receipt['additional_calibration'])
            self.assertEqual(receipt['held_rows'], receipt['held_questions'] * 3)

    def test_held_text_and_numeric_values_do_not_fit_vocabulary_idf_or_scaler(self):
        data = synthetic(); held = np.asarray([s.fold_for(qid) == 1 for qid in data['ids']])
        data['numeric'][held] += 1000.
        for i in np.flatnonzero(held):
            data['texts'][i] = 'HELDONLYTOKEN ZZXYYZZ'
        _, _, model, _ = s.outer_prediction(data, 'word12_C1', 1)
        self.assertNotIn('HELDONLYTOKEN', model.text.vocabulary_)
        self.assertNotIn('ZZXYYZZ', model.text.vocabulary_)
        np.testing.assert_allclose(model.scaler.mean_, data['numeric'][~held].mean(0))
        reference = s.vectorizer('word12_C1').fit([data['texts'][i] for i in np.flatnonzero(~held)])
        self.assertEqual(model.text.vocabulary_, reference.vocabulary_)
        np.testing.assert_array_equal(model.text.idf_, reference.idf_)

    def test_no_neural_load_or_prediction_allowed(self):
        observer = driver.perception.Observer('unused', 1.)
        with driver.no_model_calls():
            for function in (observer.load, driver.perception.predict, driver.original.predict):
                with self.assertRaisesRegex(AssertionError, 'forbids neural'):
                    function()

    def test_cache_alignment_rejects_changed_paid_inputs(self):
        class Observer:
            _recipe = {'max_length': 512}
            def entry(self, row, path, tree):
                if row['answer']:
                    raise AssertionError('Reference reached cache alignment')
                return {'question': row['question'], 'path': path, 'root': tree['']['text'], 'current': tree[path]['text']}
        observer = Observer(); rows = [{'id': 'q', 'question': 'Public', 'answer': 'SECRET'}]
        trees = {'q': {path: {'text': 'Paid ' + path, 'prompt_sha256': path} for path in s.PATHS}}
        metadata = [{'id': 'q', 'path': path, 'prompt_sha256': path,
                     'input_sha256': driver.cached.fingerprint(observer.entry({**rows[0], 'answer': ''}, path, trees['q']))}
                    for path in s.PATHS]
        values = {name: np.zeros((3, width)) for name, width in (('state', 4), ('plain', 8), ('guided', 8))}
        lookup, _ = driver.cached.verify_features(rows, trees, observer, metadata, values)
        self.assertEqual(len(lookup), 3)
        metadata[1]['input_sha256'] = 'CHANGED'
        with self.assertRaisesRegex(ValueError, 'exact paid'):
            driver.cached.verify_features(rows, trees, observer, metadata, values)

    def test_metrics_are_question_grouped_and_joint_bits_are_correct(self):
        data = synthetic(12); n = len(data['labels'])
        uniform, exact = np.ones((n, 8)) / 8, np.eye(8)[data['labels']]
        report = s.diagnostics(data, {'numeric_C1': uniform, 'word12_C1': exact, 'char35_C1': exact})
        for panel in report.values():
            for metric in ('current_brier', 'repair_brier', 'retry_brier'):
                self.assertAlmostEqual(panel['models']['numeric_C1'][metric], .25)
                self.assertEqual(panel['models']['word12_C1'][metric], 0.)
                self.assertEqual(panel['paired_differences']['word12_C1_minus_numeric_C1'][metric]['ci95'], [-.25, -.25])

    def test_registration_precedes_fitting_or_scoring(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            data = synthetic()
            def stop_before_fit(*args):
                self.assertTrue((output / 'registration.json').exists())
                self.assertFalse((output / 'summary.json').exists())
                raise RuntimeError('Synthetic stop before any classifier fit')
            with patch.object(driver, 'load_inputs', return_value=(data, [], {})), \
                    patch.object(driver, 'registration', return_value={'source_files': {}, 'input_evidence': {}}), \
                    patch.object(s, 'outer_prediction', side_effect=stop_before_fit):
                with self.assertRaisesRegex(RuntimeError, 'before any classifier fit'):
                    driver.run(SimpleNamespace(output=output))


if __name__ == '__main__':
    unittest.main()
