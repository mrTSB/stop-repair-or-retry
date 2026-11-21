"""Grounded observations and prompts must ignore benchmark references."""

import unittest
from unittest.mock import Mock

from grounded_review_pilot import grounded_prompt
from grounded_pomdp import Observer
from examples import text_pair
from evidence import left_text


class GroundedInputs(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'chem1', 'domain': 'science', 'subject': 'chemistry',
                    'question': 'What is a mole?\n\nOptions:\nA. A number of particles\nB. A pressure',
                    'answer': 'SECRET_GOLD'}
        self.node = {'answer': 'A', 'text': 'A mole counts particles.\nFinal: A',
                     'input_tokens': 30, 'output_tokens': 12, 'confidence': None,
                     'prompt_sha256': 'abc', 'finish': 'stop'}

    def test_reference_does_not_enter_review(self):
        actual = grounded_prompt(self.row, self.node, 'The mole is a unit of amount of substance.')
        changed = grounded_prompt({**self.row, 'answer': 'DIFFERENT_GOLD'}, self.node,
                                  'The mole is a unit of amount of substance.')
        self.assertEqual(actual, changed)
        self.assertNotIn('SECRET_GOLD', actual)

    def test_runtime_entry_matches_training_text(self):
        observer = Observer('unused', 1.)
        observer._retriever = Mock()
        observer._retriever.context.return_value = ('Fixed background', [])
        actual = observer.entry(self.row, '', {'': self.node})
        self.assertEqual(actual['left'], left_text(self.row, 'Fixed background'))
        self.assertEqual(actual['right'], text_pair(self.row, self.node)[1])
        self.assertEqual(observer._retriever.context.call_args.args[0]['answer'], '')
        changed = observer.entry({**self.row, 'answer': 'DIFFERENT_GOLD'}, '', {'': self.node})
        self.assertEqual(actual, changed)


if __name__ == '__main__':
    unittest.main()
