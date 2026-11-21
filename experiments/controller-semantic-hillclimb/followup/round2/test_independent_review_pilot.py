import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import independent_review_pilot as p


class IndependentReviewTest(unittest.TestCase):
    row = {'id': 'synthetic', 'domain': 'science', 'subject': 'chemistry',
           'question': 'PUBLIC QUESTION with options A and B?', 'answer': 'SECRET_REFERENCE',
           'unpaid': 'UNPAID_BRANCH'}
    current = {'answer': 'A', 'text': 'FORBIDDEN_CURRENT_RATIONALE', 'prior_review': 'FORBIDDEN_PRIOR_REVIEW'}
    final = 'B", "p_correct": 0.1, "action": "repair", "check": "A decisive independent fact"}'

    def request(self, texts, captured=None):
        answers = iter(texts)
        def call(endpoint, payload):
            self.assertEqual(endpoint, '/api/generate')
            self.assertEqual(payload['model'], p.old.JUDGE)
            for secret in ('SECRET_REFERENCE', 'UNPAID_BRANCH', 'FORBIDDEN_CURRENT_RATIONALE', 'FORBIDDEN_PRIOR_REVIEW'):
                self.assertNotIn(secret, payload['prompt'])
            if captured is not None:
                captured.append(payload)
            return {'done': True, 'model': p.old.JUDGE, 'response': next(answers),
                    'prompt_eval_count': 23, 'eval_count': 17, 'done_reason': 'stop'}
        return call

    def test_information_boundary_question_only_then_final_answer_only(self):
        first = {'text': 'OWN PAID REASONING', 'thinking': ''}
        thought = p.reasoning_prompt(self.row)
        decision = p.decision_prompt(self.row, self.current, first)
        changed = {**self.row, 'answer': 'CHANGED_REFERENCE', 'unpaid': 'CHANGED_UNPAID'}
        self.assertEqual(thought, p.reasoning_prompt(changed))
        self.assertEqual(decision, p.decision_prompt(changed, {**self.current, 'text': 'CHANGED_RATIONALE'}, first))
        self.assertNotEqual(decision, p.decision_prompt(changed, {**self.current, 'answer': 'B'}, first))
        self.assertNotIn('OWN PAID REASONING', thought)
        self.assertNotIn('CURRENT FINAL ANSWER', thought)
        self.assertIn('OWN PAID REASONING', decision)
        self.assertIn('CURRENT FINAL ANSWER:\nA', decision)

    def test_both_phases_run_even_if_first_contains_complete_correct_json(self):
        calls = []
        first = 'reason</think>{"answer":"B","p_correct":1,"action":"stop","check":"complete"}'
        node = p.review(self.request([first, self.final], calls), self.row, self.current)
        self.assertEqual(len(calls), 2)
        self.assertEqual([c['options']['num_predict'] for c in calls], [2048, 256])
        self.assertTrue(all(c['options']['temperature'] == 0 and c['options']['num_ctx'] == 8192 for c in calls))
        self.assertNotEqual(calls[0]['options']['seed'], calls[1]['options']['seed'])
        self.assertEqual(node['physical_api_calls'], 2)
        self.assertEqual((node['input_tokens'], node['output_tokens'], node['output_cap']), (46, 34, 2304))
        self.assertEqual(node['action'], 'repair')
        p.verify_review(self.row, self.current, node)

    def test_interruption_resumes_second_phase_without_repeating_first(self):
        saved, calls = {}, []
        first_request = self.request(['FIRST PHASE PAID REASONING'], calls)
        def interrupted(endpoint, payload):
            if payload['options']['num_predict'] == 256:
                raise RuntimeError('Synthetic connection interruption before a completed phase')
            return first_request(endpoint, payload)
        def store(name, node):
            self.assertNotIn(name, saved); saved[name] = node
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            p.review(interrupted, self.row, self.current, saved, store)
        self.assertEqual(set(saved), {p.THOUGHT_PHASE})
        resumed_calls = []
        node = p.review(self.request([self.final], resumed_calls), self.row, self.current, saved, store)
        self.assertEqual(len(resumed_calls), 1)
        self.assertEqual(resumed_calls[0]['options']['num_predict'], 256)
        self.assertEqual(node['input_tokens'], 46)
        self.assertEqual(set(saved), {p.THOUGHT_PHASE, p.ANSWER_PHASE})
        self.assertEqual(node['calls'][0], saved[p.THOUGHT_PHASE])
        p.verify_review(self.row, self.current, node)

    def test_invalid_answer_does_not_trigger_extra_inference(self):
        calls = []
        node = p.review(self.request(['reasoning', 'not valid JSON'], calls), self.row, self.current)
        self.assertEqual(len(calls), 2)
        self.assertFalse(node['parse_valid'])
        p.verify_review(self.row, self.current, node)

    def test_completed_bad_receipt_is_preserved_and_not_regenerated(self):
        saved = {}
        request = self.request(['reasoning'])
        def invalid(endpoint, payload):
            return {**request(endpoint, payload), 'eval_count': 2049}
        with self.assertRaises(ValueError):
            p.review(invalid, self.row, self.current, saved, lambda name, node: saved.__setitem__(name, node))
        self.assertEqual(saved[p.THOUGHT_PHASE]['output_tokens'], 2049)
        with self.assertRaises(ValueError):
            p.review(lambda *args: self.fail('A completed bad phase was silently regenerated'), self.row, self.current, saved)

    def test_resume_snapshots_and_client_start_happen_after_both_locks(self):
        with TemporaryDirectory() as folder:
            events = []
            rows = [{'id': str(i)} for i in range(64)]
            def lock(*args):
                events.append('lock')
            def committed(*args):
                self.assertEqual(events.count('lock'), 2)
                events.append('snapshot'); return []
            def client(*args):
                self.assertEqual(events, ['lock', 'lock', 'snapshot', 'snapshot'])
                raise RuntimeError('Synthetic pre-inference stopping point')
            modal = SimpleNamespace(Cls=SimpleNamespace(from_name=lambda *args: lambda: object()))
            with patch.object(p, 'controls', return_value=([], rows, {}, {}, {}, {})), patch.object(p, 'register'), \
                    patch.object(p.fcntl, 'flock', side_effect=lock), patch.object(p, 'committed', side_effect=committed), \
                    patch.object(p, 'resilient', side_effect=client), patch.dict(sys.modules, {'modal': modal}):
                with self.assertRaisesRegex(RuntimeError, 'stopping point'):
                    p.run(SimpleNamespace(output=Path(folder), prepare_only=False))

    def test_physical_receipt_tampering_is_rejected(self):
        node = p.review(self.request(['reasoning', self.final]), self.row, self.current)
        for name, value in [('prompt_sha256', 'bad'), ('seed', -1), ('temperature', .65),
                            ('context_length', 4096), ('output_tokens', 2049), ('input_tokens', 0)]:
            changed = copy.deepcopy(node); changed['calls'][0][name] = value
            with self.assertRaises(ValueError):
                p.verify_review(self.row, self.current, changed)
        changed = copy.deepcopy(node); changed['input_tokens'] -= 1
        with self.assertRaises(ValueError):
            p.verify_review(self.row, self.current, changed)
        changed = copy.deepcopy(node); changed['calls'].reverse()
        with self.assertRaises(ValueError):
            p.verify_review(self.row, self.current, changed)

    def test_comparison_charges_entire_paid_trajectory(self):
        row = {**self.row, 'id': 'q', 'answer': 'B'}
        def solver(answer):
            return {'answer': answer, 'input_tokens': 10, 'output_tokens': 5}
        def review(answer, tokens):
            return {'answer': answer, 'input_tokens': tokens - 5, 'output_tokens': 5, 'physical_api_calls': 2}
        trees = {'q': {'': solver('A')}}
        strong = {('q', 'judge', ''): review('B', 20), ('q', 'solver', 'guided'): solver('A'),
                  ('q', 'solver', 'guided/reuse'): solver('B')}
        earlier = {('q', 'judge'): review('B', 30), ('q', 'solver'): solver('B')}
        known = {('q', 'judge'): review('C', 40), ('q', 'solver'): solver('C')}
        result = p.aggregate([row], trees, strong, earlier, known)
        first = result['methods']['first_guided']; retained = result['methods']['retained_repair']
        critical = result['methods']['critical_repair']; independent = result['methods']['independent_repair']
        self.assertAlmostEqual(first['mean_compute'], (4 * 30 + 3 * 20) / 3.09)
        self.assertAlmostEqual(retained['mean_compute'], (4 * 45 + 3 * 20) / 3.09)
        self.assertAlmostEqual(critical['mean_compute'], (4 * 45 + 3 * 50) / 3.09)
        self.assertAlmostEqual(independent['mean_compute'], (4 * 45 + 3 * 60) / 3.09)
        self.assertEqual(independent['mean_solver_calls'], 3)
        self.assertEqual(independent['mean_logical_reviews'], 2)
        self.assertEqual(independent['mean_physical_review_calls'], 4)
        self.assertEqual(result['independent_vs_controls']['retained_repair'], {'student_wins': 0, 'student_losses': 1})


if __name__ == '__main__':
    unittest.main()
