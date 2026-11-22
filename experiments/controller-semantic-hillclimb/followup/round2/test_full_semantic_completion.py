from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import full_semantic_completion_probe as probe
import nested_full_semantic_diagnostic as diagnostic
import extract_full_semantic_completion as extraction


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(hidden_size=1024)
        self.embedding = torch.nn.Embedding(8, 1024); self.calls = 0
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FullSemanticTests(unittest.TestCase):
    def test_both_activations_share_one_forward_and_hooks_are_removed(self):
        model = extraction.network.Network(Encoder(), 2)
        tensors = {'input_ids': torch.ones((2, 8), dtype=torch.long),
            'attention_mask': torch.ones((2, 8), dtype=torch.long), 'numeric': torch.zeros((2, 2))}
        values = probe.capture(model, tensors)
        self.assertEqual(model.encoder.calls, 1); self.assertEqual(values['text1024'].shape, (2, 1024))
        self.assertFalse(model.norm._forward_hooks); self.assertFalse(model.project._forward_hooks)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with torch.inference_mode():
            np.testing.assert_array_equal(values['state'], model.state(torch.tensor(values['joint256'])).numpy())
        with patch.object(model, 'forward', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError): probe.capture(model, tensors)
        self.assertFalse(model.norm._forward_hooks); self.assertFalse(model.project._forward_hooks)

    def test_outer_held_labels_do_not_change_scaler_or_selected_model(self):
        rng = np.random.default_rng(238); n = 90
        rows = [{'id': str(i//3), 'fold': i//3 % 3} for i in range(n)]
        x = rng.normal(size=(n, 12)); y = np.arange(n) % 8
        data = {'records': rows, 'features': {diagnostic.FAMILIES[1]: x}, 'labels': y,
            'valid': np.ones(n, dtype=bool)}
        with patch.object(diagnostic, 'CS', (.01,)):
            held, before, _, first = diagnostic.outer_prediction(data, diagnostic.FAMILIES[1], 1)
            changed = {**data, 'labels': y.copy()}; changed['labels'][held] = 15-y[held]
            _, after, _, second = diagnostic.outer_prediction(changed, diagnostic.FAMILIES[1], 1)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(first['selected'], second['selected'])
        self.assertFalse(first['outer_held_labels_used_for_fit_or_selection'])


if __name__ == '__main__': unittest.main()
