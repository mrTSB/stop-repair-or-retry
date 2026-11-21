"""Removing masked padding preserves the encoder output and its actual token bill."""

import unittest

import torch
from transformers import DebertaV2Config, DebertaV2Model

from network import Network
from dynamic_grounded_pomdp import Observer, trim_tensors


class DynamicPadding(unittest.TestCase):
    def test_real_encoder_logits_are_preserved_when_padding_is_removed(self):
        torch.manual_seed(7)
        config = DebertaV2Config(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32,
                                max_position_embeddings=64, max_relative_positions=32,
                                relative_attention=True, pos_att_type=['p2c', 'c2p'],
                                position_biased_input=False, hidden_dropout_prob=0.,
                                attention_probs_dropout_prob=0.)
        model = Network(DebertaV2Model(config), 2).eval()
        batch = {'input_ids': torch.tensor([[1, 7, 4, 2, 9, 2, 0, 0, 0, 0]]),
                 'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]),
                 'token_type_ids': torch.zeros((1, 10), dtype=torch.long),
                 'numeric': torch.tensor([[.2, .8]])}
        trimmed = trim_tensors(batch)
        self.assertEqual(trimmed['input_ids'].shape, (1, 6))
        self.assertEqual(trimmed['numeric'].shape, (1, 2))
        with torch.inference_mode():
            for padded, actual in zip(model(**batch), model(**trimmed)):
                torch.testing.assert_close(padded, actual, rtol=1e-5, atol=1e-6)

    def test_actual_unpadded_slots_are_charged_for_every_paid_pass(self):
        observer = Observer('unused', 1.)
        observer.load = lambda: None
        observer._recipe = {'max_length': 512, 'parameters': 184_000_000}
        observer.values = lambda row, path, tree, judge=None: (None, None, None, 7 if judge is None else 11)
        decisions = [{'path': '', 'queried': False}, {'path': 'repair', 'queried': True}]
        tokens, calls, slots, _ = observer.charges({'id': 'q'}, {}, {('q', 'repair'): {}}, decisions)
        self.assertEqual((tokens, calls, slots), (25, 3, 25))

    def test_noncontiguous_padding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'right-padded'):
            trim_tensors({'input_ids': torch.tensor([[0, 1, 2]]),
                          'attention_mask': torch.tensor([[0, 1, 1]]),
                          'numeric': torch.zeros((1, 2))})


if __name__ == '__main__':
    unittest.main()
