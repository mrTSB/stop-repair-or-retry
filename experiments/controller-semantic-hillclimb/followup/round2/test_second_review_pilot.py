import unittest
from unittest.mock import patch

import second_review_pilot as pilot
import long_review_pilot as original


class SecondReviewTest(unittest.TestCase):
    row = {'id': 'synthetic', 'domain': 'science', 'question': 'Public question?', 'answer': 'SECRET'}
    root = {'text': 'Already paid root review'}
    current = {'text': 'Already paid current solution'}

    def request(self, texts):
        values = iter(texts)
        def call(endpoint, payload):
            self.assertEqual(endpoint, '/api/generate')
            self.assertNotIn('SECRET', payload['prompt'])
            return {'done': True, 'model': pilot.old.JUDGE, 'response': next(values),
                    'prompt_eval_count': 23, 'eval_count': 17, 'done_reason': 'stop'}
        return call

    def test_two_phase_mechanics_match_original_with_same_prompt(self):
        texts = ['Bounded thought without a completed decision', 'B", "p_correct": 0.2, "action": "repair", "check": "Decisive fact"}']
        got = pilot.review(self.request(texts), self.row, self.root, self.current)
        with patch.object(original, 'reasoning_prompt', return_value=pilot.critical_prompt(self.row, self.root, self.current)):
            expected = original.review(self.request(texts), self.row, self.current, 1024, 512)
        for name in ('text', 'answer', 'raw_verdict', 'verdict', 'input_tokens', 'output_tokens',
                     'physical_api_calls', 'output_cap', 'parse_valid', 'json_complete', 'p_correct', 'action'):
            self.assertEqual(got[name], expected[name])
        self.assertEqual(got['physical_api_calls'], 2)
        self.assertEqual(got['input_tokens'], 46)
        self.assertEqual(got['output_tokens'], 34)
        self.assertNotEqual(got['calls'][0]['seed'], expected['calls'][0]['seed'])
        pilot.verify_review(self.row, self.root, self.current, got)

    def test_completed_first_phase_is_not_repeated_by_answer_correctness(self):
        text = 'reason</think>{"answer":"A","p_correct":0.9,"action":"stop","check":"fact"}'
        got = pilot.review(self.request([text]), self.row, self.root, self.current)
        self.assertEqual(got['physical_api_calls'], 1)
        self.assertEqual(got['answer'], 'A')
        pilot.verify_review(self.row, self.root, self.current, got)

    def test_prompt_uses_only_public_question_and_paid_prefix(self):
        before = pilot.critical_prompt(self.row, self.root, self.current)
        after = pilot.critical_prompt({**self.row, 'answer': 'ANOTHER_REFERENCE', 'unpaid': 'UNPAID'}, self.root, self.current)
        self.assertEqual(before, after)
        self.assertNotIn('SECRET', before)
        self.assertIn('Avoid anchoring', before)


if __name__ == '__main__':
    unittest.main()
