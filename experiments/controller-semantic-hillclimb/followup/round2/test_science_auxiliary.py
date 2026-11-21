"""Targeted checks for supervision balance, blind input, and binary objective."""

from collections import defaultdict
import unittest
from unittest.mock import patch

import numpy as np
import torch

from science_auxiliary_data import SCIENCE_SUFFIX, examples, option_entry
from train_science_auxiliary import margin, metrics, weighted_loss
from science_option_observer import ScienceOptionObserver
from subject_science_pomdp import SubjectPlanner, evaluate as subject_evaluate


class AuxiliaryTests(unittest.TestCase):
    def row(self, count=4, answer='A'):
        options = [f'choice {i}' for i in range(count)]
        return {'id': f'synthetic-{count}', 'subject': 'chemistry', 'answer': answer, 'options': options,
                'question': 'Synthetic question\n\nOptions:\n' + '\n'.join(f'{chr(65+i)}. {v}' for i, v in enumerate(options)) + SCIENCE_SUFFIX}

    def test_question_and_class_mass(self):
        samples = examples([self.row(4), self.row(10, 'J')], 98)
        masses = defaultdict(lambda: [0., 0.])
        for r in samples:
            masses[r['id']][r['label']] += r['weight']
        np.testing.assert_allclose(list(masses.values()), [[3.5, 3.5], [3.5, 3.5]])
        self.assertEqual(len(samples), 14)

    def test_reference_blind_canonical_entry(self):
        row = self.row(10, 'J')
        entry = option_entry(row, 'B', 98)
        changed = {**row, 'answer': 'C'}
        self.assertEqual(entry, option_entry(changed, 'B', 98))
        without_options = {k: v for k, v in row.items() if k != 'options'}
        self.assertEqual(entry, option_entry(without_options, 'B', 98))
        self.assertEqual(entry['right'], 'Proposed answer: B. choice 1')
        self.assertEqual(entry['numeric'], [0.] * 98)
        self.assertIn('invalid', option_entry(row, '', 98)['right'])

    def test_loss_has_fixed_denominator_and_gradient(self):
        x = torch.tensor([[0., 2., 0., 2.], [0., -1., 0., -1.]], requires_grad=True)
        np.testing.assert_allclose(margin(x).detach(), [2., -1.])
        y = torch.tensor([1., 0.]); w = torch.tensor([.5, .125])
        loss = weighted_loss(x, y, w, 16)
        expected = (torch.nn.functional.softplus(torch.tensor(-2.)) * .5
                    + torch.nn.functional.softplus(torch.tensor(-1.)) * .125) / 16
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=7)
        loss.backward()
        self.assertLess(float(x.grad[0, 1]), 0.)
        self.assertGreater(float(x.grad[1, 1]), 0.)

    def test_ranking_ties_independent_of_answer(self):
        samples = examples([self.row(4, 'D')], 98)
        result = metrics(np.zeros(4), samples, 1.)
        self.assertEqual(result['ranking_correct'], 0)
        self.assertAlmostEqual(result['balanced_bce'], np.log(2))

    def test_actual_slots_and_neutral_unused_heads(self):
        row = self.row()
        observer = ScienceOptionObserver('unused', 1.)
        observer._recipe = {'numeric_width': 98, 'max_length': 8, 'parameters': 434306836}
        tree = {'': {'answer': 'B', 'text': 'Untrusted rationale'},
                'repair': {'answer': 'B', 'text': 'Changed rationale'}}
        encoded = {'input_ids': torch.ones((1, 8), dtype=torch.long),
                   'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
                   'numeric': torch.zeros((1, 98))}
        outputs = (np.array([[0., 2., 0., 2.]]), np.ones((1, 8)), np.ones((1, 8)))
        with patch.object(observer, 'load'), patch('science_option_observer.encode', return_value=encoded), \
                patch('science_option_observer.predict', return_value=outputs) as predicted:
            value = observer.values(row, '', tree)
            same = observer.values({**row, 'answer': 'D'}, '', tree)
            self.assertIs(value, same)
            self.assertEqual(predicted.call_count, 1)
            self.assertEqual(predicted.call_args.args[1]['input_ids'].shape, (1, 5))
            np.testing.assert_array_equal(value[1], np.zeros(8))
            np.testing.assert_array_equal(value[2], np.zeros(8))
            charged = observer.charges(row, tree, {}, [{'path': '', 'queried': False},
                                                       {'path': 'repair', 'queried': False}])
            self.assertEqual(charged, (10, 2, 10, .434306836))

    def test_subject_dispatch_executes_one_observer(self):
        model = SubjectPlanner({'math': 'math-model', 'chemistry': 'chem-model'},
                               {'math': {'penalty': .01}, 'chemistry': {'penalty': .02}}, {})
        row = {'subject': 'chemistry', 'answer': 'D'}
        with patch('subject_science_pomdp.policy.evaluate', return_value={'compute': 123.}) as called:
            self.assertEqual(subject_evaluate(row, {}, {}, model, {}, 'A'), {'compute': 123.})
            called.assert_called_once_with(row, {}, {}, 'chem-model', {'penalty': .02}, 'A')
        with self.assertRaises(ValueError):
            subject_evaluate(row, {}, {}, model, {}, 'C')


if __name__ == '__main__':
    unittest.main()
