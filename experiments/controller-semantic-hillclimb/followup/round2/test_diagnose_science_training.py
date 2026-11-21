import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import diagnose_science_training as diagnostic


class TrainingDiagnosticTests(unittest.TestCase):
    def test_uniform_scores_have_known_listwise_loss_and_zero_margin(self):
        samples = [{'id': str(q), 'label': int(i == 0)} for q, count in enumerate((4, 10)) for i in range(count)]
        result = diagnostic.additional_metrics(np.zeros(14), samples, 3.)
        self.assertAlmostEqual(result['listwise_cross_entropy'], (np.log(4) + np.log(10)) / 2)
        self.assertEqual(result['correct_minus_best_wrong_margin_mean'], 0.)
        self.assertAlmostEqual(result['random_choice_expected_accuracy'], (.25 + .1) / 2)

    def test_complete_fixed_selection_is_saved_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); checkpoint = root / 'checkpoint'; checkpoint.mkdir()
            source = root / 'source.jsonl'
            rows = [{'uuid': f'q{i}', 'field': 'chemistry' if i < 720 else 'physics',
                     'question': f'Synthetic question {i}', 'options': ['a', 'b', 'c', 'd'], 'answer_letter': 'A'}
                    for i in range(1440)]
            source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps({'train_ids': [r['uuid'] for r in rows], 'calibration_ids': [],
                                            'source_sha256': diagnostic.sha(source)}))
            recipe = checkpoint / 'recipe.json'
            recipe.write_text(json.dumps({'input_profile': diagnostic.data.PROFILE, 'numeric_width': 1, 'weights': {'fake': 'hash'}}))
            calibration = root / 'calibration.json'
            calibration.write_text(json.dumps({'recipe_sha256': diagnostic.sha(recipe),
                                               'temperature_selected_on_auxiliary_calibration': 3.}))
            args = SimpleNamespace(output=root / 'output', metadata=metadata, source=source, checkpoint=checkpoint,
                                   calibration_summary=calibration, device='cpu')
            class StopBeforeInference(Exception):
                pass
            def forbidden_load(path):
                registration = json.loads((args.output / 'registration.json').read_text())
                self.assertEqual(len(registration['selection']), 32)
                self.assertEqual(registration['subject_counts'], {'chemistry': 16, 'physics': 16})
                self.assertFalse(registration['inference_started'])
                self.assertEqual(registration['gradient_updates'], 0)
                self.assertEqual(registration['inputs_sha256'], diagnostic.sha(args.output / 'inputs.jsonl'))
                raise StopBeforeInference()
            with patch.object(diagnostic.network, 'load', side_effect=forbidden_load):
                with self.assertRaises(StopBeforeInference):
                    diagnostic.run(args)
            self.assertFalse((args.output / 'outputs.npz').exists())


if __name__ == '__main__':
    unittest.main()
