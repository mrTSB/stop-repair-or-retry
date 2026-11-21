import copy
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import fit_expanded_mentor as run
import test_expanded_mentor_core as core_fixture
import test_expanded_review_metadata as metadata_fixture


class Anchor:
    def __init__(self, directory):
        self.classifier = run.core.perception.Observer(str(directory), 1.)
        self.fit_receipt = {'state_count': 8, 'state_bits': ['current_correct', 'repair_correct', 'retry_correct']}
        self.dynamics = {s: np.eye(8) for s in run.spec.SUBJECTS}


class ExpandedFitTests(unittest.TestCase):
    def test_only_ambient_caller_manifest_normalizes_and_both_maps_are_bound(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            parent, caller = directory/'parent.py', directory/'caller.py'
            parent.write_text('parent-source'); caller.write_text('new-fit-source')
            prior = {str(parent): run.io.sha(parent)}
            current = {**prior, str(caller): run.io.sha(caller)}
            expected = {'validated_parent_provenance': {'source_files': prior, 'tree_sha256': 'same'},
                        'metadata_sha256': 'metadata'}
            actual = copy.deepcopy(expected); actual['validated_parent_provenance']['source_files'] = current
            result = run.bind_registered_input_identity({'identity': actual}, {'input_identity': expected})
            self.assertEqual(result['identity'], expected)
            self.assertEqual(result['ambient_source_manifest_normalization']['fitting_entrypoint_sources'], current)
            changed = copy.deepcopy(actual); changed['metadata_sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'beyond'):
                run.bind_registered_input_identity({'identity': changed}, {'input_identity': expected})
            parent.write_text('changed-source')
            with self.assertRaisesRegex(ValueError, 'changed'):
                run.bind_registered_input_identity({'identity': actual}, {'input_identity': expected})

    def test_full_synthetic_six_model_fit_and_no_overwrite_resume(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); recipe_dir = root/'observer'; recipe_dir.mkdir()
            run.io.write(recipe_dir/'recipe.json', {'parameters': 435_000_000, 'max_length': 512})
            base_path = root/'base.pkl'; base_path.write_bytes(pickle.dumps(Anchor(recipe_dir)))
            rows, trees, judges, metadata, outputs = core_fixture.fixture()
            records = metadata_fixture.fixture()
            for record in records:
                if record['protocol'] == 'root':
                    qid = record['id']
                    record['current_sha256'] = run.inputs.fingerprint(trees[qid][''])
                    record['review_sha256'] = run.inputs.fingerprint(judges[qid, ''])
            bundle = {'rows': rows, 'trees': trees, 'judges': judges, 'metadata': metadata,
                'base_path': base_path, 'base_spec': {'model': 'base', 'settings': {'penalty': .08}},
                'identity': {'inherited_temperature': 1., 'encoder_recipe_sha256': run.io.sha(recipe_dir/'recipe.json')}}
            output = root/'fit'; output.mkdir()
            with patch.object(run.core.perception.Observer, 'load', side_effect=AssertionError('No neural work')):
                models, a4 = run.fit_models(output, 'synthetic-registration', bundle, outputs, records)
            self.assertEqual(len(models), 6)
            self.assertEqual(sum(r['metadata_review'] for r in models), 2)
            self.assertEqual(a4['policy_module'], 'dynamic_grounded_pomdp')
            snapshots = {p.name: run.io.sha(p) for p in output.iterdir()}
            with patch.object(run.core.perception.Observer, 'load', side_effect=AssertionError('No neural work')):
                replay, again_a4 = run.fit_models(output, 'synthetic-registration', bundle, outputs, records)
            self.assertEqual(replay, models); self.assertEqual(again_a4, a4)
            self.assertEqual(snapshots, {p.name: run.io.sha(p) for p in output.iterdir()})
            for record in models:
                model = pickle.loads((output/(record['key']+'.pkl')).read_bytes())
                if record['metadata_review']:
                    self.assertIsInstance(model, run.metadata_runtime.Planner)
                    self.assertIsNone(model.review_observer)
                    self.assertFalse(model.fit_receipt['base_is_strongest_completed_A'])
                    self.assertEqual(model.metadata_adapter.receipt['training_questions'], 320)
                self.assertEqual(model.mentor.fit_receipt['paired_root_questions'], 320)

    def test_structural_identity_and_existing_outputs_fail_closed(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = {'coefficients': np.arange(12.).reshape(3, 4), 'temperature': 1.5}
            before = run.save_model(directory/'model.pkl', value)
            equivalent = {**value, 'coefficients': np.asfortranarray(value['coefficients'])}
            self.assertEqual(run.save_model(directory/'model.pkl', equivalent), before)
            changed = copy.deepcopy(value); changed['coefficients'][0, 0] = 99
            with self.assertRaisesRegex(ValueError, 'differs'):
                run.save_model(directory/'model.pkl', changed)
            run.save_arrays(directory/'arrays.npz', {'p': np.asarray([1., 2.])})
            with self.assertRaisesRegex(ValueError, 'differ'):
                run.save_arrays(directory/'arrays.npz', {'p': np.asarray([2., 1.])})
            run.save_json(directory/'receipt.json', {'value': 1})
            with self.assertRaisesRegex(ValueError, 'differs'):
                run.save_json(directory/'receipt.json', {'value': 2})


if __name__ == '__main__': unittest.main()
