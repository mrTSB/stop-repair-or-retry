import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import nested_dense_diagnostic as study


def fixture():
    rng = np.random.default_rng(17)
    records = [{'id': f'q{q}', 'path': path, 'subject': 'chemistry' if q % 2 else 'physics',
                'fold': q % 3} for q in range(48) for path in ('', 'repair', 'retry')]
    x = rng.normal(size=(144, 130))
    y = np.where(x[:, 0] > 0, 8, 0)
    return {'records': records, 'features': {study.FAMILIES[0]: x,
            study.FAMILIES[1]: np.concatenate((x, rng.normal(size=(144, 256))), axis=1)},
            'labels': y, 'valid': np.ones(144, dtype=bool)}


class NestedDenseTests(unittest.TestCase):
    def test_run_rejects_another_active_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            with study.io.lock(args.output / '.diagnostic.lock'), patch.object(study, 'run_locked') as execute:
                with self.assertRaises(BlockingIOError): study.run(args, {}, {})
            execute.assert_not_called()

    def test_fixed_grid_gives_both_families_equal_regularization_search(self):
        control, enriched = (study.candidates(f) for f in study.FAMILIES)
        self.assertEqual([v['C'] for v in control], list(study.CS))
        self.assertEqual(len(enriched), 8)
        self.assertEqual({v['pca_components'] for v in enriched}, {None, 32})
        with self.assertRaises(ValueError): study.candidates('third_unregistered_family')

    def test_group_overlap_rejected(self):
        records = [{'id': 'x'}, {'id': 'x'}, {'id': 'y'}]
        with self.assertRaises(ValueError):
            study.group_receipt(records, [True, False, False], [False, True, True])

    def test_outer_held_labels_cannot_change_selection_or_model(self):
        data = fixture()
        altered = copy.deepcopy(data)
        held = np.asarray([r['fold'] == 0 for r in data['records']])
        altered['labels'][held] ^= 1
        with threadpool_limits(limits=1):
            a = study.outer_prediction(data, study.FAMILIES[0], 0)
            b = study.outer_prediction(altered, study.FAMILIES[0], 0)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2][-1].coef_, b[2][-1].coef_)
        self.assertEqual(a[3]['selected'], b[3]['selected'])
        self.assertEqual(a[3]['inner_search'], b[3]['inner_search'])
        np.testing.assert_allclose(a[2][0].mean_, data['features'][study.FAMILIES[0]][~held].mean(0))

    def test_every_inner_fit_excludes_the_outer_held_questions(self):
        data = fixture(); held = np.asarray([r['fold'] == 0 for r in data['records']])
        x = data['features'][study.FAMILIES[0]]
        # Marker is not a label, and no held marker may reach a scaler/learner.
        x[:, 1] = np.arange(len(x))
        original = study.learner; seen = []
        def checked(features, labels, settings):
            indexes = features[:, 1].astype(int)
            self.assertFalse(held[indexes].any())
            seen.append(len(indexes))
            return original(features, labels, settings)
        with threadpool_limits(limits=1), patch.object(study, 'learner', side_effect=checked):
            study.outer_prediction(data, study.FAMILIES[0], 0)
        self.assertEqual(len(seen), 4 * 3 + 1)
        self.assertEqual(seen[-1], 96)
        self.assertTrue(all(n < 96 for n in seen[:-1]))

    def test_PCA_and_scalers_use_only_supplied_training_rows(self):
        data = fixture(); x = data['features'][study.FAMILIES[1]]
        with threadpool_limits(limits=1):
            model = study.learner(x[:96], data['labels'][:96], {'C': .01, 'pca_components': 32})
        dense = model[0].named_transformers_['dense']
        np.testing.assert_allclose(dense[0].mean_, x[:96, 130:].mean(0))
        self.assertEqual(dense[1].n_samples_, 96)
        self.assertEqual(dense[1].n_components_, 32)
        self.assertEqual(model[-1].n_features_in_, 162)

    def test_predict_masks_invalid_current_state(self):
        data = fixture(); x = data['features'][study.FAMILIES[0]]
        with threadpool_limits(limits=1):
            model = study.learner(x, data['labels'], {'C': .1, 'pca_components': None})
        p = study.predict(model, x[:4], [False] * 4)
        np.testing.assert_array_equal(p[:, study.fixed.core.CURRENT == 1], 0.)
        np.testing.assert_allclose(p.sum(1), 1.)

    def test_dense_nested_route_produces_whole_question_predictions(self):
        data = fixture()
        with threadpool_limits(limits=1):
            held, p, model, receipt = study.outer_prediction(data, study.FAMILIES[1], 1)
        self.assertEqual(int(held.sum()), 48)
        self.assertEqual(p.shape, (48, 16))
        self.assertEqual(len(receipt['inner_search']), 8)
        self.assertEqual(receipt['training_questions'], 32)
        self.assertEqual(receipt['held_questions'], 16)
        self.assertFalse(receipt['outer_held_labels_used_for_fit_or_selection'])
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(1), 1.)


if __name__ == '__main__':
    unittest.main()
