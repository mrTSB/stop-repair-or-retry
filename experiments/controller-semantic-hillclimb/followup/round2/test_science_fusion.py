"""Synthetic paid-prefix, alignment, and component-compute fusion checks."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from science_fusion_pomdp import FusionObserver, SubjectPlanner, evaluate
from development_science_fusion import verified_component_lookup, fingerprint


class FakeObserver:
    def __init__(self, directory, profile, slots, shift):
        self.directory = str(directory)
        self.device = 'cpu'
        self._recipe = {'parameters': 400000000, 'max_length': 512, 'input_profile': profile}
        self.slots, self.shift = slots, shift
        self.seen = []

    def load(self):
        pass

    def entry(self, row, path, tree):
        assert row['answer'] == ''
        assert set(tree) == ({'', path} if path else {''})
        return {'left': row['question'], 'right': tree[path]['text'], 'numeric': [0.]}

    def values(self, row, path, tree, judge=None):
        self.entry(row, path, tree)
        self.seen.append((path, set(tree)))
        return np.arange(4) + self.shift, np.arange(8) + self.shift, np.arange(8) + self.shift, self.slots + (10 if path else 0)


class ScienceFusionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        observers = []
        for i, profile in enumerate(('full-options-openstax-v1', 'supergpqa-option-correctness-v1')):
            path = self.directory / str(i); path.mkdir(); (path / 'recipe.json').write_text(json.dumps({'profile': profile}))
            observers.append(FakeObserver(path, profile, 100 - 50 * i, 100 * i))
        self.observer = FusionObserver(observers)
        self.row = {'id': 'synthetic', 'subject': 'chemistry', 'answer': 'D', 'question': 'Question'}
        self.tree = {path: {'text': 'Node ' + path, 'prompt_sha256': 'hash-' + path}
                     for path in ('', 'repair', 'retry', 'repair/repair')}

    def tearDown(self):
        self.temporary.cleanup()

    def test_two_complementary_arrays_only_paid_prefix(self):
        values = self.observer.values(self.row, 'repair', self.tree)
        self.assertEqual([len(v) for v in values[:3]], [8, 16, 16])
        np.testing.assert_array_equal(values[0], [0, 1, 2, 3, 100, 101, 102, 103])
        self.assertEqual(values[3], 170)
        self.assertEqual(self.observer._recipe['parameters'], 800000000)
        self.assertEqual(self.observer._recipe['per_forward_parameters'], 400000000)
        for component in self.observer.components:
            self.assertEqual(component.seen, [('repair', {'', 'repair'})])

    def test_every_component_pass_charged(self):
        decisions = [{'path': '', 'queried': False}, {'path': 'repair', 'queried': False}]
        charges = self.observer.charges(self.row, self.tree, {}, decisions, include_reviews=False)
        self.assertEqual(charges, (320, 4, 320, .4))
        ledger = self.observer.component_charges(self.row, self.tree, {}, decisions, include_reviews=False)
        self.assertEqual(len(ledger), 4)
        self.assertAlmostEqual(sum(r['compute'] for r in ledger), .4 * 320 / 3.09)
        self.assertTrue(all(r['kind'] == 'solver' and r['calls'] == 1 and len(r['recipe_sha256']) == 64 for r in ledger))
        future = self.observer._recipe['parameters'] / 1e9 * self.observer._recipe['max_length'] / 3.09
        self.assertAlmostEqual(future, .4 * (512 + 512) / 3.09)

    def test_evaluator_preserves_cost_and_exposes_physical_weights(self):
        decisions = [{'path': '', 'queried': False}]
        sub = SimpleNamespace(classifier=self.observer)
        wrapper = SubjectPlanner({'chemistry': sub}, {'chemistry': {'penalty': .1}}, {})
        item = {'compute': 123., 'decisions': decisions, 'verifier_parameters_billions': .4, 'verifier_token_slots': 150}
        with patch('science_fusion_pomdp.policy.evaluate', return_value=item) as called:
            result = evaluate(self.row, self.tree, {}, wrapper, {}, 'A')
            self.assertEqual(result['compute'], 123.)
            self.assertEqual(result['verifier_physical_parameters_billions'], .8)
            called.assert_called_once_with(self.row, self.tree, {}, sub, {'penalty': .1}, 'A')
        with self.assertRaises(ValueError):
            evaluate(self.row, self.tree, {}, wrapper, {}, 'C')

    def test_component_metadata_requires_exact_input_and_prompt(self):
        folder = self.directory / 'features'; folder.mkdir()
        component = self.observer.components[0]
        metadata = []
        for path in ('', 'repair', 'retry'):
            tree = {p: self.tree[p] for p in ('', path)}
            entry = component.entry({**self.row, 'answer': ''}, path, tree)
            metadata.append({'id': self.row['id'], 'path': path, 'prompt_sha256': self.tree[path]['prompt_sha256'],
                             'input_sha256': fingerprint(entry)})
        path = folder / 'training-neural.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in metadata))
        np.savez_compressed(folder / 'training-neural.npz', state=np.zeros((3, 4)), plain=np.zeros((3, 8)), guided=np.zeros((3, 8)))
        lookup, checks = verified_component_lookup(folder, [self.row], {self.row['id']: self.tree}, component, 3)
        self.assertEqual(len(lookup), 3)
        self.assertEqual(len(checks), 3)
        metadata[0]['input_sha256'] = 'changed'
        path.write_text(''.join(json.dumps(r) + '\n' for r in metadata))
        with self.assertRaises(ValueError):
            verified_component_lookup(folder, [self.row], {self.row['id']: self.tree}, component, 3)


if __name__ == '__main__':
    unittest.main()
