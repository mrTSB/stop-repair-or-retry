import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import joint_observer_pomdp as run
import fit_joint_observer_pomdp as fit


class InputBuilder:
    def __init__(self, directory):
        self.directory = str(directory)
        self._retriever = SimpleNamespace(sha256='synthetic-corpus')

    def entry(self, row, path, paid):
        if row['answer'] or set(paid) != {'', path}:
            raise AssertionError('Reference or unpaid state leaked into observation')
        return {'left': row['question'], 'right': paid[path]['text'],
            'numeric': [len(paid['']['text']), len(paid[path]['text'])]}


class JointRuntimeTests(unittest.TestCase):
    def setup_observer(self, folder):
        initial = folder/'initial'; initial.mkdir()
        run.io.write(initial/'recipe.json', {'corpus_sha256': 'synthetic-corpus'})
        directory = folder/'joint'; directory.mkdir()
        recipe = {'architecture': run.network.ARCHITECTURE,
            'initial_directory': str(initial.resolve()),
            'initial_recipe_sha256': run.io.sha(initial/'recipe.json'),
            'corpus_sha256': 'synthetic-corpus', 'parameters': 434305808,
            'max_length': 512, 'prior': [1/16]*16}
        run.io.write(directory/'recipe.json', recipe)
        observer = run.Observer(str(directory), InputBuilder(initial), run.io.sha(directory/'recipe.json'))
        return observer, recipe

    def test_paid_inputs_cache_and_exact_inference_charges(self):
        with TemporaryDirectory() as tmp:
            observer, recipe = self.setup_observer(Path(tmp))
            row = {'id': 'q', 'question': 'synthetic', 'answer': '', 'subject': 'physics'}
            tree = {'': {'text': 'root', 'answer': 'A'}, 'repair': {'text': 'child', 'answer': ''}}
            def encode(tokenizer, entries, length):
                return {'attention_mask': torch.ones((1, int(sum(entries[0]['numeric']))))}
            model = SimpleNamespace(to=lambda device: model)
            with patch.object(run.network, 'load', return_value=(model, object(), recipe)), \
                 patch.object(run.network.initial, 'encode', side_effect=encode), \
                 patch.object(run.network, 'predict', return_value=np.zeros((1, 16))) as predict:
                observer.values(row, '', {'': tree['']})
                p, slots = observer.values(row, 'repair', tree)
                np.testing.assert_array_equal(p[8:], 0.)
                self.assertEqual(slots, 9)
                decisions = [{'path': '', 'queried': False}, {'path': 'repair', 'queried': False}]
                self.assertEqual(observer.charges(row, tree, {}, decisions), (17, 2, 17, .434305808))
                self.assertEqual(predict.call_count, 2)
                changed = copy.deepcopy(tree); changed['']['text'] = 'longer-root'
                observer.values(row, 'repair', changed)
                self.assertEqual(predict.call_count, 3)
                with self.assertRaises(ValueError): observer.charges(row, tree, {}, decisions*2)
                with self.assertRaises(ValueError): observer.charges(row, tree, {}, decisions, True)
            self.assertIsNone(observer.__getstate__()['_model'])
            self.assertFalse(observer.__getstate__()['_cache'])

    def test_reference_future_state_and_changed_corpus_rejected(self):
        with TemporaryDirectory() as tmp:
            observer, _ = self.setup_observer(Path(tmp))
            row = {'id': 'q', 'question': 'q', 'answer': ''}
            tree = {'': {'text': 'root', 'answer': 'A'}}
            with self.assertRaises(ValueError): observer.entry({**row, 'answer': 'A'}, '', tree)
            with self.assertRaises(ValueError): observer.entry(row, '', {**tree, 'repair': tree['']})
            observer.input_builder._retriever.sha256 = 'changed'
            with self.assertRaises(ValueError): observer.entry(row, '', tree)

    def test_fit_requires_passed_sealed_screen(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            run.io.write(folder/'registration.json', {'schema': fit.screen.SCHEMA})
            run.io.write(folder/'complete.json', {'schema': fit.screen.SCHEMA,
                'registration_sha256': run.io.sha(folder/'registration.json'), 'screen_passed': False})
            run.io.write(folder/'summary.json', {'screen': {'passed': False}})
            with self.assertRaisesRegex(ValueError, 'successful TRAIN-only'):
                fit.inputs(SimpleNamespace(screen=folder))


if __name__ == '__main__': unittest.main()
