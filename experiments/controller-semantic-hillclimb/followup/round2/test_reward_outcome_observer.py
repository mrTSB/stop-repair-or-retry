import unittest

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForSequenceClassification

import reward_outcome_observer as reward


class Tokenizer:
    bos_token = 'BOS'

    def apply_chat_template(self, messages, **kwargs): return 'BOS'+messages[-1]['content']
    def __call__(self, text, **kwargs): return {'input_ids': list(range(len(text)))}
    def pad(self, values, **kwargs):
        width = max(len(v['input_ids']) for v in values)
        ids = [v['input_ids']+[0]*(width-len(v['input_ids'])) for v in values]
        return {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor([[int(i != 0) for i in row] for row in ids])}


class RewardObserverTests(unittest.TestCase):
    def test_reference_and_unpaid_states_are_rejected(self):
        row = {'question': 'visible', 'answer': ''}; paid = {'': {'text': 'root'}, 'repair': {'text': 'current'}}
        self.assertEqual(reward.conversation(row, 'repair', paid),
            [{'role': 'user', 'content': 'visible'}, {'role': 'assistant', 'content': 'current'}])
        for bad_row, path, bad_paid in (({**row, 'answer': 'B'}, 'repair', paid),
            (row, 'repair', {**paid, 'retry': {'text': 'future'}}), (row, '', paid), (row, 'repair/repair', paid)):
            with self.assertRaises(ValueError): reward.conversation(bad_row, path, bad_paid)

    def test_truncation_is_explicit_and_preserves_ends(self):
        entries = reward.tokenize(Tokenizer(), [[{'role': 'assistant', 'content': 'x'*1800}]])
        self.assertTrue(entries[0]['truncated']); self.assertEqual(entries[0]['original_tokens'], 1800)
        self.assertEqual(len(entries[0]['input_ids']), 1536)
        self.assertEqual(entries[0]['input_ids'][:768], list(range(768)))
        self.assertEqual(entries[0]['input_ids'][-768:], list(range(1032, 1800)))

    def test_padding_pooling_matches_single_request_scores(self):
        torch.manual_seed(238)
        model = Qwen3ForSequenceClassification(Qwen3Config(vocab_size=32, hidden_size=16,
            intermediate_size=24, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, pad_token_id=0, num_labels=1)).eval()
        entries = [{'input_ids': [1, 2, 3]}, {'input_ids': [1, 4]}]
        scores, hidden = reward.score_batch(model, Tokenizer(), entries, 'cpu')
        for i, entry in enumerate(entries):
            single, vector = reward.score_batch(model, Tokenizer(), [entry], 'cpu')
            np.testing.assert_allclose(scores[i], single[0], atol=1e-6)
            np.testing.assert_allclose(hidden[i], vector[0], atol=1e-6)

    def test_fitted_transforms_do_not_change_on_held_features(self):
        rng = np.random.default_rng(22); base = rng.normal(size=(96, 3)); scores = rng.normal(size=96)
        hidden = rng.normal(size=(96, 40)); labels = np.arange(96)%16
        for setting in ('score-forest', 'dense-linear-0.1'):
            model = reward.fit(base, scores, hidden, labels, setting)
            if model.pca is not None:
                np.testing.assert_allclose(model.pca.mean_, hidden.mean(0))
                np.testing.assert_allclose(model.scaler.mean_, np.column_stack((base, scores)).mean(0))
                before = model.pca.components_.copy()
            p = model.predict(base[:2]*100, scores[:2]*100, hidden[:2]*100, [False, True])
            self.assertTrue(np.isfinite(p).all()); np.testing.assert_allclose(p.sum(1), 1)
            np.testing.assert_array_equal(p[0, 8:], 0.)
            if model.pca is not None: np.testing.assert_array_equal(model.pca.components_, before)


if __name__ == '__main__': unittest.main()
