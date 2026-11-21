"""Role, paid-prefix, physical-call accounting, and horizon checks for B."""

from types import SimpleNamespace
import copy
import json
from pathlib import Path
import tempfile
import unittest

from matched_review_b import policy, validate_review_charge, choose, copy_completed_bank, sha
from matched_review_b import review, mentor_pairs, old
from difficulty_pomdp import MissingObservation


class MatchedBTests(unittest.TestCase):
    def row(self):
        return {'id': 'synthetic', 'domain': 'science', 'subject': 'chemistry', 'answer': 'B'}

    def node(self, answer):
        return {'answer': answer, 'input_tokens': 1, 'output_tokens': 2}

    def judge(self, action='repair', answer='B'):
        return {'answer': answer, 'action': action, 'p_correct': .2, 'input_tokens': 30, 'output_tokens': 3,
                'calls': [{'input_tokens': 10, 'output_tokens': 1}, {'input_tokens': 20, 'output_tokens': 2}],
                'physical_api_calls': 2}

    def settings(self, **extra):
        return {'root_stop': 'agreement', 'second_stop': 'agreement', 'second_action': 'reuse', **extra}

    def test_llm_role_and_horizon_and_physical_cost(self):
        row = self.row()
        tree = {'': self.node('A'), 'guided': self.node('C'), 'guided/reuse': self.node('B')}
        result = policy(row, tree, {(row['id'], ''): self.judge()}, self.settings())
        self.assertEqual(result['path'], 'guided/reuse')
        self.assertEqual(result['solver_calls'], 3)
        self.assertEqual(result['judge_calls'], 1)
        self.assertEqual(result['judge_physical_calls'], 2)
        self.assertEqual(result['verifier_calls'], 0)
        self.assertAlmostEqual(result['compute'], (4 * 9 + 3 * 33) / 3.09)
        self.assertTrue(all('belief' not in decision for decision in result['decisions']))

    def test_agreement_stops_and_reference_does_not_change_actions(self):
        row = self.row(); tree = {'': self.node('A'), 'guided': self.node('B')}
        judges = {(row['id'], ''): self.judge()}
        first = policy(row, tree, judges, self.settings())
        second = policy({**row, 'answer': 'C'}, tree, judges, self.settings())
        self.assertEqual(first['solver_calls'], 2)
        self.assertEqual(first['decisions'], second['decisions'])
        self.assertEqual(first['path'], second['path'])

    def test_root_llm_retry_and_missing_prefix(self):
        row = self.row(); tree = {'': self.node('A'), 'retry': self.node('C'), 'retry/retry': self.node('B')}
        settings = self.settings(root_stop='action', second_stop='never', second_action='root_action')
        judges = {(row['id'], ''): self.judge('retry')}
        result = policy(row, tree, judges, settings)
        self.assertEqual(result['path'], 'retry/retry')
        with self.assertRaises(MissingObservation) as caught:
            policy(row, {'': tree['']}, judges, settings)
        self.assertEqual((caught.exception.kind, caught.exception.path), ('solver', 'retry'))

    def test_omitted_review_phase_is_rejected(self):
        judge = self.judge(); judge['input_tokens'] = 20
        with self.assertRaises(ValueError):
            validate_review_charge(judge)

    def test_one_phase_completion_charged_once(self):
        judge = self.judge()
        judge.update(calls=judge['calls'][:1], physical_api_calls=1, input_tokens=10, output_tokens=1)
        validate_review_charge(judge)
        row = self.row()
        result = policy(row, {'': self.node('B')}, {(row['id'], ''): judge}, self.settings())
        self.assertEqual(result['judge_physical_calls'], 1)
        self.assertEqual(result['judge_tokens'], 11)

    def test_physical_prompt_receipt_replay(self):
        row = {**self.row(), 'answer': '', 'question': 'Synthetic problem with options.'}
        current = {**self.node('A'), 'text': 'Final: A'}
        seen = []
        def fake_request(endpoint, payload):
            seen.append(payload)
            text = ('A short synthetic thought.' if len(seen) == 1 else
                    'B", "p_correct": 0.2, "action": "repair", "check": "Synthetic check"}')
            return {'done': True, 'model': old.JUDGE, 'response': text,
                    'prompt_eval_count': 10, 'eval_count': 5, 'done_reason': 'stop'}
        judge = review(fake_request, row, current, 1024, 512)
        self.assertEqual([p['options']['num_predict'] for p in seen], [1024, 512])
        validate_review_charge(judge)
        mentor_pairs.verify_review(row, current, judge, 1024, 512)
        tampered = copy.deepcopy(judge)
        tampered['calls'][0]['seed'] += 1
        with self.assertRaises(AssertionError):
            mentor_pairs.verify_review(row, current, tampered, 1024, 512)

    def test_best_ordinary_B_is_retained(self):
        old = {'dev': {'accuracy': .8, 'mean_compute': 2000}}
        new = {'dev': {'accuracy': .79, 'mean_compute': 1000}}
        self.assertIs(choose(old, new), old)

    def test_completed_copy_preserves_source_and_detects_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'C'; output = root / 'B'
            source.mkdir(); output.mkdir()
            bank = source / 'dev-observations.jsonl'; bank.write_text('{"sanitized":true}\n')
            (source / 'summary.json').write_text(json.dumps({'development_questions': 256, 'bank_sha256': sha(bank)}))
            (source / 'identity.json').write_text(json.dumps({'dev_questions_sha256': 'same-panel'}))
            args = SimpleNamespace(hybrid=source, output=output)
            copied = copy_completed_bank(args, {'dev_questions_sha256': 'same-panel'})
            self.assertEqual(copied.read_bytes(), bank.read_bytes())
            with copied.open('a') as stream:
                stream.write('{"additional":true}\n')
            self.assertEqual(copy_completed_bank(args, {'dev_questions_sha256': 'same-panel'}), copied)
            self.assertEqual(bank.read_text(), '{"sanitized":true}\n')
            bank.write_text('{"changed":true}\n')
            with self.assertRaises(ValueError):
                copy_completed_bank(args, {'dev_questions_sha256': 'same-panel'})


if __name__ == '__main__':
    unittest.main()
