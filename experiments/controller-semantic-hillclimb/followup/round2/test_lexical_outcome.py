import unittest

import numpy as np

import lexical_outcome_forecast as forecast


class LexicalTests(unittest.TestCase):
    def test_only_paid_text_and_blank_reference(self):
        row = {'question': 'Public question', 'answer': ''}
        paid = {'': {'text': 'root'}, 'repair': {'text': 'current'}}
        text = forecast.public_text(row, 'repair', paid)
        self.assertIn('root', text); self.assertIn('current', text)
        with self.assertRaises(ValueError): forecast.public_text({**row, 'answer': 'A'}, 'repair', paid)
        with self.assertRaises(ValueError): forecast.public_text(row, '', paid)
        with self.assertRaises(ValueError): forecast.public_text(row, 'repair', {**paid, 'retry': {'text': 'future'}})

    def test_fit_vocabulary_scaling_and_prior_use_only_supplied_training(self):
        texts = ['common field physics' if i%2 else 'common field chemistry' for i in range(20)]
        x = np.arange(60).reshape(20, 3).astype(float); y = np.array([0, 8]*10)
        model = forecast.fit(texts, x, y, 1.)
        self.assertNotIn('held_unique_token', model.vectorizer.vocabulary_)
        np.testing.assert_array_equal(model.scaler.mean_, x.mean(0))
        before = dict(model.vectorizer.vocabulary_)
        p = model.predict(['held_unique_token common', 'held_unique_token physics'], np.zeros((2, 3)), [False, True])
        self.assertEqual(before, model.vectorizer.vocabulary_)
        np.testing.assert_array_equal(p[0, 8:], 0.)
        np.testing.assert_allclose(p.sum(1), 1.)
        self.assertTrue(np.isfinite(p).all())
        constant = forecast.fit(texts, x, np.zeros(20, int), .1)
        self.assertEqual(constant.constant, 0)
        self.assertTrue(np.isfinite(constant.predict(texts[:1], x[:1], [False])).all())


if __name__ == '__main__': unittest.main()
