"""Synthetic OOF alignment and explicit question-exclusion checks; no encoder."""

import copy
import unittest

import numpy as np

from development_refined_observer import aligned_lookup, fold_for, EXCLUSION_NOTE


class RefinedObserverTests(unittest.TestCase):
    def samples(self):
        return [{'id': qid, 'kind': 'solver', 'path': path, 'fold': fold_for(qid),
                 'left': 'Question ' + qid, 'right': 'Answer ' + path, 'numeric': [0., 1.],
                 'train': True, 'label': 1, 'valid': True, 'plain': [-1.] * 8, 'guided': [-1.] * 8}
                for qid in ('q1', 'q2') for path in ('', 'repair', 'retry')]

    def outputs(self):
        return {key: np.arange(6 * width).reshape(6, width).astype(float)
                for key, width in (('state', 4), ('plain', 8), ('guided', 8))}

    def prompts(self, expected):
        return {(r['id'], r['kind'], r['path']): 'prompt-' + r['id'] + r['path'] for r in expected}

    def test_join_matches_nodes_in_actual_oof_order(self):
        expected = self.samples(); metadata = list(reversed(expected)); outputs = self.outputs()
        lookup, receipt = aligned_lookup(metadata, outputs, expected, self.prompts(expected), {'q1', 'q2'})
        np.testing.assert_array_equal(lookup['q1', ''][0], outputs['state'][5])
        self.assertEqual(receipt[0]['oof_index'], 5)
        self.assertEqual(receipt[0]['fold'], fold_for('q1'))
        self.assertEqual(receipt[0]['prompt_sha256'], 'prompt-q1')
        self.assertIn('not nested CV', EXCLUSION_NOTE)

    def test_wrong_fold_rejected(self):
        expected = self.samples(); altered = copy.deepcopy(expected)
        altered[0]['fold'] = (altered[0]['fold'] + 1) % 3
        with self.assertRaises(ValueError):
            aligned_lookup(altered, self.outputs(), expected, self.prompts(expected), {'q1', 'q2'})

    def test_changed_input_or_flag_rejected(self):
        expected = self.samples()
        for name, value in (('left', 'tampered input'), ('train', False), ('label', 0)):
            altered = copy.deepcopy(expected); altered[0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                aligned_lookup(altered, self.outputs(), expected, self.prompts(expected), {'q1', 'q2'})

    def test_wrong_shapes_nonfinite_and_missing_prompt_rejected(self):
        expected = self.samples(); prompts = self.prompts(expected)
        wrong = self.outputs(); wrong['guided'] = np.zeros((5, 8))
        with self.assertRaises(ValueError):
            aligned_lookup(expected, wrong, expected, prompts, {'q1', 'q2'})
        wrong = self.outputs(); wrong['state'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            aligned_lookup(expected, wrong, expected, prompts, {'q1', 'q2'})
        prompts.pop(('q1', 'solver', ''))
        with self.assertRaises(ValueError):
            aligned_lookup(expected, self.outputs(), expected, prompts, {'q1', 'q2'})


if __name__ == '__main__':
    unittest.main()
