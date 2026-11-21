import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import nonlinear_repairability as nonlinear
import prospective_freeze as freeze
import preview_nonlinear as preview


class NonlinearProspectiveTests(unittest.TestCase):
    def test_nonlinear_role_preserves_prespecified_family_and_excluded_calibration(self):
        parameters = {'max_leaf_nodes': 7, 'max_iter': 100, 'learning_rate': .1, 'l2_regularization': 5.,
            'min_samples_leaf': 30, 'max_bins': 255, 'early_stopping': False,
            'random_state': nonlinear.SEED, 'categorical_features': None}
        folds = [{'fold': outer, 'question_overlap': 0, 'fit_questions': 6, 'held_questions': 3,
            'temperature': 1., 'inner_folds': [{'fold': inner, 'fit_questions': 4,
                'calibration_questions': 2, 'outer_held_overlap': 0} for inner in range(3)]} for outer in range(3)]
        model = SimpleNamespace(forecaster=nonlinear.Forecaster(nonlinear.ConstantModel(0), 1.),
            fit_receipt={'state_count': 8, 'paired_questions': 9, 'forecaster_hyperparameters': parameters,
                'temperature_grid': list(nonlinear.TEMPERATURES),
                'forecaster_and_temperature_exclude_own_question_in_emission_fit': True,
                'structural_transitions_and_costs_unchanged': True,
                'calibration': {'final_temperature': 1., 'emission_temperature_excludes_own_question': True,
                    'final_temperature_fits_only_training_OOF_labels': True, 'outer_folds': folds}})
        freeze.validate_nonlinear_forecaster(model)
        changed = copy.deepcopy(model)
        changed.fit_receipt['calibration']['outer_folds'][0]['inner_folds'][0]['outer_held_overlap'] = 1
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            freeze.validate_nonlinear_forecaster(changed)
        changed = copy.deepcopy(model); changed.fit_receipt['forecaster_hyperparameters']['early_stopping'] = True
        with self.assertRaisesRegex(ValueError, 'prespecified'):
            freeze.validate_nonlinear_forecaster(changed)

    def test_fit_and_development_caches_are_bound_without_parsing_question_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); model_dir = root / 'models'; model_dir.mkdir()
            base = root / 'base'; base.mkdir(); data = root / 'data'; (data / 'splits').mkdir(parents=True)
            cache = root / 'cache'; cache.mkdir()
            def write(directory, name, value):
                path = directory / name; path.write_text(json.dumps(value)); return path
            # Opaque bytes intentionally cannot be decoded as question records.
            for split in ('train', 'dev'):
                (data / 'splits' / (split + '-questions.jsonl')).write_bytes(b'opaque')
            for name in ('identity.json', 'training-neural.npz', 'training-neural.jsonl'):
                (base / name).write_bytes(b'opaque fitting input')
            identity = {'base_directory': str(base), 'fitting_questions': 512, 'decision_nodes': 1536,
                'train_questions_sha256': freeze.sha(data / 'splits/train-questions.jsonl'),
                'dev_questions_sha256': freeze.sha(data / 'splits/dev-questions.jsonl'),
                'base_identity_sha256': freeze.sha(base / 'identity.json'),
                'training_neural_sha256': freeze.sha(base / 'training-neural.npz'),
                'training_neural_metadata_sha256': freeze.sha(base / 'training-neural.jsonl'),
                'encoder_recipe_sha256': 'recipe'}
            write(model_dir, 'fitting-registration.json', identity)
            alignment = write(model_dir, 'training-alignment.json', [])
            hashes = {}
            for leaves in (7, 15):
                for smoothing in (8, 32):
                    name = f'nonlinear-leaf{leaves}-smooth{smoothing}.pkl'
                    (model_dir / name).write_bytes(name.encode()); hashes[name] = freeze.sha(model_dir / name)
            identity = {**identity, 'model_sha256': hashes, 'training_alignment_sha256': freeze.sha(alignment)}
            path = write(model_dir, 'identity.json', identity)
            write(model_dir, 'fit-complete.json', {'identity_sha256': freeze.sha(path), 'models': hashes,
                  'training_only': True, 'no_neural_inference': True, 'test_used': False})
            for name in ('neural.npz', 'metadata.jsonl'):
                (cache / name).write_bytes(b'opaque development features')
            complete = write(cache, 'complete.json', {'dev_questions_sha256': identity['dev_questions_sha256'],
                'encoder_recipe_sha256': 'recipe', 'nodes': 768, 'test_used': False,
                'neural_sha256': freeze.sha(cache / 'neural.npz'), 'metadata_sha256': freeze.sha(cache / 'metadata.jsonl')})
            summary = {'development_feature_directory': str(cache), 'development_feature_complete_sha256': freeze.sha(complete)}
            candidate = {'fit_module': 'nonlinear_repairability'}; model_path = model_dir / next(iter(hashes))
            files = freeze.nonlinear_fitting_evidence(model_dir, candidate, identity, data, model_path, summary)
            self.assertIn(cache / 'neural.npz', files)
            self.assertIn(model_path, files)
            (cache / 'neural.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'observation cache changed'):
                freeze.nonlinear_fitting_evidence(model_dir, candidate, identity, data, model_path, summary)

    def test_preview_retains_pending_references_without_final_selection(self):
        def candidate(accuracy):
            return {'dev': {'accuracy': accuracy, 'mean_compute': 100.}}
        output = preview.report([candidate(.60), candidate(.65)], [candidate(.68)], ['unfinished-summary.json'],
                                {'Repair twice': {'accuracy': .50}})
        self.assertNotIn('selected_A', output)
        self.assertFalse(output['final_selection_performed'])
        self.assertEqual(output['nonlinear_A']['dev']['accuracy'], .65)
        self.assertEqual(output['best_completed_A_preview']['dev']['accuracy'], .68)
        self.assertEqual(output['pending_A_references'], ['unfinished-summary.json'])
        self.assertEqual(freeze.infer_a_module({}, {'source_sha256': freeze.sha(freeze.HERE / 'development_nonlinear.py')}),
                         'continuous_repairability')


if __name__ == '__main__':
    unittest.main()
