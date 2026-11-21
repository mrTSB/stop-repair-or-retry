"""Leakage and local-training integration checks without benchmark examples."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import reserve
import refine


class DeduplicationTest(unittest.TestCase):
    def test_option_changes_do_not_hide_a_duplicate_stem(self):
        a = 'Determine the force acting upon this particle.\n\nOptions:\nA. First\nB. Second'
        b = 'Determine the force acting upon this particle.\n\nOptions:\nA. Second\nB. First'
        self.assertEqual(reserve.normalized(a), reserve.normalized(b))

    def test_small_edits_and_distinct_questions(self):
        index = reserve.DuplicateIndex()
        index.add(' '.join(f'word{i}' for i in range(100)))
        self.assertTrue(index.duplicate(' '.join(f'word{i}' if i != 50 else 'edited' for i in range(100))))
        self.assertFalse(index.duplicate(' '.join(f'different{i}' for i in range(100))))


class LocalTrainingTest(unittest.TestCase):
    def test_every_old_and_new_question_gets_an_excluded_fold(self):
        seen = {i: set() for i in (0, 1, 2, 'full')}
        snapshots = {}

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.layer = nn.ModuleList([nn.Linear(2, 2) for _ in range(6)])

            def forward(self, x):
                for layer in self.encoder.layer:
                    x = torch.tanh(layer(x))
                return x

        class Tiny(nn.Module):
            def __init__(self, fold):
                super().__init__()
                self.fold = fold
                self.encoder = Encoder()
                self.state = nn.Linear(2, 4)
                self.plain = nn.Linear(2, 8)
                self.guided = nn.Linear(2, 8)

            def forward(self, input_ids, numeric, attention_mask):
                if self.training and torch.is_grad_enabled():
                    seen[self.fold].update(input_ids[:, 0].tolist())
                x = self.encoder(numeric)
                return self.state(x), self.plain(x), self.guided(x)

        def fake_encode(tokenizer, rows, length):
            ids = torch.tensor([[r['index']] for r in rows], dtype=torch.long)
            return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                    'numeric': torch.tensor([[r['index'] / 12., 1.] for r in rows])}

        def fake_save(model, tokenizer, directory, recipe):
            directory.mkdir(parents=True)
            snapshots[str(directory)] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            (directory / 'recipe.json').write_text(json.dumps(recipe))

        def fake_load(directory):
            recipe = json.loads((directory / 'recipe.json').read_bytes())
            model = Tiny(recipe['fold'])
            if str(directory) in snapshots:
                model.load_state_dict(snapshots[str(directory)])
            return model, None, recipe

        def fake_predict(model, tensors, device, batch):
            model.eval()
            with torch.inference_mode():
                return tuple(x.numpy() for x in model(**tensors))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / 'initial'
            for fold in (0, 1, 2, 'full'):
                folder = initial / ('full' if fold == 'full' else 'folds/' + str(fold))
                folder.mkdir(parents=True)
                (folder / 'recipe.json').write_text(json.dumps({'fold': fold}))
            rows = []
            for i in range(12):
                rows.append({'id': f'q{i}', 'index': i, 'fold': (i // 2) % 3,
                             'label': i % 2, 'kind': 'solver', 'path': '', 'train': True,
                             'valid': True, 'plain': [-1.] * 8, 'guided': [-1.] * 8})
            inputs = root / 'train.jsonl'
            inputs.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (root / 'manifest.json').write_text(json.dumps({
                'input_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
                'development_or_test_used': False, 'nodes': 12, 'new_nodes': 6,
                'new_questions': 6, 'corpus_sha256': 'synthetic-test'}))
            args = Namespace(inputs=inputs, initial=initial, output=root / 'output', device='cpu',
                             epochs=1, batch=2, replay_questions=6)
            with patch.object(refine.AutoTokenizer, 'from_pretrained', return_value=object()), \
                 patch.object(refine, 'encode', side_effect=fake_encode), \
                 patch.object(refine, 'load', side_effect=fake_load), \
                 patch.object(refine, 'save', side_effect=fake_save), \
                 patch.object(refine, 'predict', side_effect=fake_predict):
                refine.run(args)
            for fold in (0, 1, 2):
                self.assertEqual(seen[fold], {r['index'] for r in rows if r['fold'] != fold})
            self.assertEqual(seen['full'], set(range(12)))
            outputs = np.load(args.output / 'oof.npz')
            self.assertEqual(outputs['state'].shape, (12, 4))
            self.assertTrue(np.isfinite(outputs['state']).all())


if __name__ == '__main__':
    unittest.main()
