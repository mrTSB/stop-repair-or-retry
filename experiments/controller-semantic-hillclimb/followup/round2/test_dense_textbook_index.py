import unittest
from unittest.mock import patch

import numpy as np
import torch

import dense_textbook_index as index


class Tokenizer:
    def __call__(self, texts, **kwargs):
        self.texts = texts
        return {'input_ids': torch.ones((len(texts), 7), dtype=torch.long),
                'attention_mask': torch.tensor([[1,1,1,1,0,0,0]]*len(texts))}


class Model:
    def __call__(self, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=torch.arange(1, 1+kwargs['input_ids'].shape[0]*7*3,
            dtype=torch.float32).reshape(kwargs['input_ids'].shape[0],7,3))


class DenseIndexTests(unittest.TestCase):
    def test_cls_normalization_query_instruction_and_padded_compute(self):
        encoder = object.__new__(index.Encoder)
        encoder.tokenizer, encoder.model, encoder.parameters = Tokenizer(), Model(), 1234
        vectors, receipt = encoder.encode(['first passage', 'second passage'])
        self.assertEqual(encoder.tokenizer.texts, ['first passage','second passage'])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-6)
        np.testing.assert_allclose(vectors[0], np.array([1,2,3])/np.sqrt(14), atol=1e-6)
        self.assertEqual(receipt, {'actual_forward_token_slots':14,'nonpadding_tokens':8,'model_parameters':1234})
        encoder.encode(['scientific concept'], query=True)
        self.assertEqual(encoder.tokenizer.texts, [index.INSTRUCTION+'scientific concept'])

    def test_zero_or_nonfinite_embedding_fails(self):
        for value in ([[0,0]], [[np.nan,1]], [[np.inf,2]]):
            with self.assertRaises(ValueError): index.normalize(value)


if __name__ == '__main__': unittest.main()
