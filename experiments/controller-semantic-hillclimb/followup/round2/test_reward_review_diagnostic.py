import unittest

import numpy as np

import reward_review_diagnostic as run


class RewardReviewTests(unittest.TestCase):
    def test_reference_and_missing_review_charges_rejected(self):
        row = {'question': 'public', 'answer': ''}
        review = {'text': 'paid reasoning and verdict', 'calls': [{'input_tokens': 20, 'output_tokens': 10}],
            'physical_api_calls': 1, 'input_tokens': 20, 'output_tokens': 10}
        self.assertEqual(run.conversation(row, review)[1]['content'], review['text'])
        with self.assertRaises(ValueError): run.conversation({**row, 'answer': 'A'}, review)
        with self.assertRaises(ValueError): run.conversation(row, {**review, 'input_tokens': 0})
        with self.assertRaises(ValueError): run.conversation(row, {**review, 'calls': []})

    def test_unrelated_held_values_and_labels_cannot_change_prediction(self):
        rng = np.random.default_rng(238); scores = rng.normal(size=90); hidden = rng.normal(size=(90, 40))
        labels = np.arange(90)%2; train = np.arange(90) < 60; held = ~train
        for setting in (('score', .1), ('dense', .1)):
            before = run.fit_predict(scores, hidden, labels, train, held, setting)
            changed_labels = labels.copy(); changed_labels[held] = 1-changed_labels[held]
            changed_hidden = hidden.copy(); changed_hidden[61:] *= 100
            changed_scores = scores.copy(); changed_scores[61:] *= 100
            after = run.fit_predict(changed_scores, changed_hidden, changed_labels, train, held, setting)
            self.assertAlmostEqual(before[0], after[0], places=12)
        labels[train] = 1
        np.testing.assert_array_equal(run.fit_predict(scores, hidden, labels, train, held, ('score', .1)), 1.)


if __name__ == '__main__': unittest.main()
