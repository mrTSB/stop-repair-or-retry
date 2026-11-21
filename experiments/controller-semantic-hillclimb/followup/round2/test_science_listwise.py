import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import train_science_listwise as trainer


class GroupedLossTest(unittest.TestCase):
    def test_equal_question_weight_with_unequal_option_counts(self):
        logits = torch.tensor([1., 0., -.3, .5, -.4, .1, .8, -.3, .2, -.5], requires_grad=True)
        labels = torch.tensor([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        total, ce, bce = trainer.grouped_loss(logits, labels, [4, 6])
        first = trainer.grouped_loss(logits[:4], labels[:4], [4])
        second = trainer.grouped_loss(logits[4:], labels[4:], [6])
        for actual, a, b in zip((total, ce, bce), first, second):
            torch.testing.assert_close(actual, (a + b) / 2)
        total.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool((logits.grad != 0).all()))

    def test_binary_term_supervises_common_margin_offset(self):
        logits = torch.zeros(4, requires_grad=True)
        labels = torch.tensor([1, 0, 0, 0])
        base = trainer.grouped_loss(logits, labels, [4])
        offset = trainer.grouped_loss(logits + 3., labels, [4])
        torch.testing.assert_close(base[1], offset[1])
        self.assertGreater(float(offset[2].detach()), float(base[2].detach()))
        self.assertGreater(float(offset[0].detach()), float(base[0].detach()))

    def test_balanced_bce_matches_existing_weight_semantics(self):
        rows = [{'id': 'q1', 'subject': 'chemistry', 'question': 'q1', 'answer': 'A', 'options': ['a', 'b', 'c', 'd']},
                {'id': 'q2', 'subject': 'physics', 'question': 'q2', 'answer': 'B', 'options': ['a', 'b', 'c', 'd', 'e', 'f']}]
        samples = trainer.data.examples(rows, 3)
        z = np.linspace(-1., 1., len(samples))
        metrics = trainer.scores(z, samples)
        self.assertAlmostEqual(metrics['balanced_bce'], metrics['question_balanced_bce'], places=7)
        self.assertAlmostEqual(metrics['joint_loss'], metrics['balanced_bce'] + metrics['listwise_cross_entropy'], places=7)

    def test_incomplete_or_ambiguous_targets_fail_closed(self):
        for labels, sizes in [([1, 1, 0, 0], [4]), ([0, 0, 0, 0], [4]), ([1, 0, 0, 0], [3]),
                              ([1, .5, 0, 0], [4])]:
            with self.assertRaises(ValueError):
                trainer.grouped_loss(torch.zeros(4), torch.tensor(labels), sizes)

    def test_sanity_selection_ignores_labels_and_order(self):
        rows = [{'id': f'{subject}-{i}', 'subject': subject, 'answer': 'A'}
                for subject in ('chemistry', 'physics') for i in range(10)]
        chosen = trainer.select_sanity(rows)
        changed = [dict(row, answer='WRONG') for row in reversed(rows)]
        self.assertEqual([row['id'] for row in chosen], [row['id'] for row in trainer.select_sanity(changed)])
        self.assertEqual(trainer.Counter(row['subject'] for row in chosen), {'chemistry': 2, 'physics': 2})

    def test_only_last_six_encoder_layers_and_state_heads_train(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.encoder = torch.nn.Module()
                self.encoder.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(8)])
                self.encoder.embedding = torch.nn.Linear(2, 2)
                self.state = torch.nn.Linear(2, 4)
                self.norm = torch.nn.LayerNorm(2)
                self.plain = torch.nn.Linear(2, 8)
                self.guided = torch.nn.Linear(2, 8)
                self.register_buffer('numeric_mean', torch.ones(2))
                self.register_buffer('numeric_std', torch.ones(2) * 3)
        model = FakeModel()
        encoder, heads = trainer.configure(model)
        self.assertEqual(len(encoder), 12)
        self.assertTrue(all(p.requires_grad for p in encoder + heads))
        self.assertTrue(all(not p.requires_grad for p in model.plain.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.guided.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.encoder.encoder.layer[0].parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.encoder.embedding.parameters()))
        torch.testing.assert_close(model.numeric_mean, torch.zeros(2))
        torch.testing.assert_close(model.numeric_std, torch.ones(2))

    def test_source_reader_decodes_registered_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train = [f't-{i}' for i in range(1440)]; calibration = [f'c-{i}' for i in range(160)]
            raw = [{'uuid': key, 'field': 'chemistry' if i < 720 else 'physics', 'question': 'synthetic',
                    'options': ['first', 'second', 'third', 'fourth'], 'answer_letter': 'B', 'answer': 'second'}
                   for i, key in enumerate(train)]
            source = directory / 'source.jsonl'
            # Unregistered values are deliberately invalid JSON. Any attempt
            # to decode them would fail this test rather than read labels.
            source.write_text('\n'.join(json.dumps(row) for row in raw) + '\n'
                              + '{"uuid":"c-0", INVALID_CALIBRATION_JSON\n'
                              + '{"uuid":"protected", INVALID_PROTECTED_JSON\n')
            metadata = {'train_ids': train, 'calibration_ids': calibration,
                        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            path = directory / 'metadata.json'; path.write_text(json.dumps(metadata))
            rows = trainer.load_registered(path, source, 'train')
            self.assertEqual(len(rows), 1440)
            with self.assertRaises(ValueError):
                trainer.load_registered(path, source, 'test')

    def test_recipe_retains_existing_online_observer_contract(self):
        import science_option_observer
        observer = science_option_observer.ScienceOptionObserver('unused', 1., device='cpu')
        with mock.patch.object(science_option_observer, 'load', return_value=(torch.nn.Identity(), None, trainer.RECIPE.copy())):
            observer.load()
        self.assertEqual(observer._recipe['input_profile'], trainer.data.PROFILE)


if __name__ == '__main__':
    unittest.main()
