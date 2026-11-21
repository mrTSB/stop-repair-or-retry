import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import prospective_collect as collect
import prospective_grade as grade


class PipelineTests(unittest.TestCase):
    def test_ordinary_late_review_parser_and_prompt_are_verified(self):
        row = {'id': 'gsm8k-synthetic', 'question': 'What is 1 plus 1?', 'subject': 'math', 'domain': 'math', 'answer': ''}
        tree = {'': {'text': 'Final: 1', 'answer': '1'}, 'repair': {'text': 'Final: 2', 'answer': '2'}}
        spec = collect.specification(row, 'ordinary', 'judge', 'repair', tree, {})
        def fake_request(endpoint, payload):
            self.assertEqual(endpoint, '/api/generate')
            return {'done': True, 'model': collect.old.JUDGE, 'response': json.dumps(
                    {'answer': '2', 'p_correct': .9, 'action': 'stop', 'feedback': 'Addition checks out.'}),
                    'prompt_eval_count': 10, 'eval_count': 20, 'done_reason': 'stop',
                    'srr_backend': {'provider': 'synthetic'}}
        node = collect.sanitized(collect.inference(fake_request, row, spec[0], 'brief', True, 'repair'))
        collect.validate_node(row, 'ordinary', 'judge', 'repair', node, tree, {})
        node['p_correct'] = .1
        with self.assertRaisesRegex(ValueError, 'frozen parser'):
            collect.validate_node(row, 'ordinary', 'judge', 'repair', node, tree, {})

    def test_matched_B_one_phase_review_is_replayed_and_charged_once(self):
        row = {'id': 'gsm8k-synthetic', 'question': 'What is 1 plus 1?', 'subject': 'math', 'domain': 'math', 'answer': ''}
        calls = []
        def fake_request(endpoint, payload=None):
            self.assertEqual(endpoint, '/api/generate')
            self.assertEqual(payload['model'], collect.old.JUDGE)
            calls.append('review')
            return {'done': True, 'model': collect.old.JUDGE,
                    'response': json.dumps({'answer': '2', 'solution': 'One plus one is two.', 'p_correct': .1,
                                            'action': 'repair', 'check': 'Addition'}),
                    'prompt_eval_count': 10, 'eval_count': 20, 'done_reason': 'stop',
                    'srr_backend': {'provider': 'synthetic'}}
        def fake_initial(safe, path, tree, reviews, is_judge):
            self.assertFalse(is_judge); self.assertEqual(path, '')
            calls.append('initial')
            spec = collect.specification(safe, 'plain', 'solver', '', tree, reviews)
            return {'text': 'Final: 1', 'answer': '1', 'prompt_sha256': collect.hashlib.sha256(spec[0].encode()).hexdigest(),
                    'seed': spec[1], 'model_digest': spec[2], 'temperature': spec[3],
                    'input_tokens': 10, 'output_tokens': 5, 'confidence': None}
        def fake_actor(request, safe, prompt, seeded, cap):
            self.assertEqual(safe['answer'], ''); self.assertEqual(cap, 128)
            calls.append('guided')
            return {'text': 'Final: 2', 'answer': '2', 'prompt_sha256': collect.hashlib.sha256(prompt.encode()).hexdigest(),
                    'seed': seeded, 'model_digest': collect.old.MODEL_DIGEST, 'temperature': .65,
                    'input_tokens': 15, 'output_tokens': 5, 'confidence': None, 'output_cap': cap}
        frozen = {'observer_assets': {}, 'selected': {'B': {'protocol': 'matched-retained-1024-512-128',
                    'settings': {'root_stop': 'agreement', 'second_stop': 'agreement', 'second_action': 'reuse'}}}}
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / 'plain.jsonl').open('a') as plain, (Path(directory) / 'strong.jsonl').open('a') as strong:
                store = collect.ResponseStore({'plain': plain, 'strong': strong})
                acquire = collect.Acquisition(store, fake_request)
                with patch.object(collect.old, 'call', side_effect=fake_initial), patch.object(collect, 'actor', side_effect=fake_actor):
                    result = collect.make_decision(frozen, {}, 'B', row, store, acquire)
                    self.assertEqual(result['item']['judge_calls'], 1)
                    self.assertEqual(result['item']['judge_physical_calls'], 1)
                    self.assertEqual(result['item']['judge_tokens'], 30)
                    self.assertEqual(result['item']['solver_calls'], 2)
                    collect.verify_decision(frozen, result, store)
                    repeated = collect.make_decision(frozen, {}, 'B', row, store, acquire)
                    self.assertEqual(result, repeated)
                    self.assertEqual(calls, ['initial', 'review', 'guided'])

    def test_public_parser_never_decodes_reference_value(self):
        full = {'id': 'gsm8k-synthetic', 'question': 'Quoted "word", braces {x} and line\nbreak',
                'answer': {'secret-reference': ['unread-value', {'nested': 2}]}, 'subject': 'math', 'domain': 'math'}
        original = collect.json.loads
        def guarded(text, *args, **kwargs):
            if 'secret-reference' in text or 'unread-value' in text:
                raise AssertionError('Reference value was decoded')
            return original(text, *args, **kwargs)
        with patch.object(collect.json, 'loads', side_effect=guarded):
            public = collect.parse_public_line(json.dumps(full))
        self.assertEqual(public, collect.public_row(full))
        with self.assertRaises(ValueError):
            collect.parse_public_line('{"id":"a","id":"b"}')

    def test_fixed_policy_acquisition_resume_and_independent_charges(self):
        row = {'id': 'gsm8k-synthetic', 'question': 'What is 1 plus 1?', 'subject': 'math', 'domain': 'math', 'answer': ''}
        calls = []
        def fake_call(safe, path, tree, reviews, is_judge):
            self.assertEqual(safe['answer'], '')
            self.assertFalse(is_judge)
            calls.append(path)
            spec = collect.specification(safe, 'plain', 'solver', path, tree, reviews)
            return {'text': 'Final: 2', 'answer': '2', 'prompt_sha256': collect.hashlib.sha256(spec[0].encode()).hexdigest(),
                    'seed': spec[1], 'model_digest': spec[2], 'temperature': spec[3],
                    'input_tokens': 10, 'output_tokens': 5, 'finish': 'stop', 'confidence': None}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plain.jsonl'
            with path.open('a') as stream:
                store = collect.ResponseStore({'plain': stream})
                acquire = collect.Acquisition(store, lambda *args: self.fail('No network call is permitted'))
                frozen = {'observer_assets': {}, 'selected': {}}
                with patch.object(collect.old, 'call', side_effect=fake_call):
                    record = collect.make_decision(frozen, {}, 'Repair twice', row, store, acquire)
                    self.assertEqual(calls, ['', 'repair', 'repair/repair'])
                    self.assertEqual(record['item']['solver_calls'], 3)
                    self.assertAlmostEqual(record['item']['compute'], 4 * 45 / 3.09)
                    self.assertNotIn('correct', record['item'])
                    collect.verify_decision(frozen, record, store)
                    acquire.ensure(row, 'plain', 'solver', 'repair/repair')
                    self.assertEqual(len(calls), 3)
            restored = collect.ResponseStore(records={'plain': collect.strict_records(path)})
            collect.verify_decision(frozen, record, restored)
            changed = copy.deepcopy(record); changed['item']['compute'] -= 1
            with self.assertRaises(ValueError):
                collect.verify_decision(frozen, changed, restored)
            with path.open('ab') as output:
                output.write(b'{"partial":')
            with self.assertRaisesRegex(ValueError, 'explicit recovery'):
                collect.strict_records(path)

    def test_completed_invalid_answer_is_preserved_without_quality_retry(self):
        row = {'id': 'gsm8k-synthetic', 'question': 'Question', 'subject': 'math', 'domain': 'math', 'answer': ''}
        spec = collect.specification(row, 'plain', 'solver', '', {}, {})
        node = {'text': 'No parseable final answer', 'answer': '',
                'prompt_sha256': collect.hashlib.sha256(spec[0].encode()).hexdigest(), 'seed': spec[1],
                'model_digest': spec[2], 'temperature': spec[3], 'input_tokens': 20, 'output_tokens': 8}
        store = collect.ResponseStore(records={'plain': [{'id': row['id'], 'kind': 'solver', 'path': '', 'response': node}]})
        acquire = collect.Acquisition(store, lambda *args: self.fail('Completed invalid answer must not be regenerated'))
        self.assertIs(acquire.ensure(row, 'plain', 'solver', ''), node)

    def test_pure_retry_reuse_requires_exact_prompt_seed_and_model(self):
        row = {'id': 'gsm8k-synthetic', 'question': 'Question', 'subject': 'math', 'domain': 'math', 'answer': ''}
        tree = {'': {'text': 'Initial'}, 'repair': {'text': 'Repaired'}, 'guided': {'text': 'Different repaired context'}}
        plain = collect.specification(row, 'plain', 'solver', 'repair/retry', tree, {})
        strong = collect.specification(row, 'strong', 'solver', 'guided/retry', tree, {})
        self.assertEqual(plain, strong)
        node = {'text': 'Final: 2', 'answer': '2', 'prompt_sha256': collect.hashlib.sha256(plain[0].encode()).hexdigest(),
                'seed': plain[1] + 1, 'model_digest': plain[2], 'temperature': plain[3], 'input_tokens': 10, 'output_tokens': 5}
        with self.assertRaises(ValueError):
            collect.validate_node(row, 'strong', 'solver', 'guided/retry', node, tree, {})

    def test_grade_rejects_incomplete_panel_before_reference_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory); target = study / 'collection'; target.mkdir()
            complete = {'passed': True, 'freeze_sha256': 'x', 'questions': 1024,
                        'method_question_audits': 8191, 'methods': list(collect.METHODS), 'digests': {}}
            (target / 'complete.json').write_text(json.dumps(complete))
            (target / 'collection-audit.json').write_text(json.dumps(complete))
            with patch.object(grade, 'verify', return_value={'data_directory': '/does/not/exist'}):
                with self.assertRaisesRegex(ValueError, 'all eight methods'):
                    grade.prerequisites(study, 'x')

    def test_paired_margins_recompute_best_baseline_and_best_controller(self):
        rows = [{'id': 'gsm8k-' + str(i), 'subject': 'math'} for i in range(8)]
        values = {}
        for method in collect.METHODS:
            bits = [0] * 8
            if method == 'Stop': bits = [1, 1, 1, 1, 0, 0, 0, 0]
            if method == 'Repair twice': bits = [0, 0, 0, 0, 1, 1, 1, 1]
            if method == 'A': bits = [1] * 6 + [0] * 2
            if method == 'B': bits = [0] * 2 + [1] * 6
            if method == 'C': bits = [1] * 8
            values[method] = [{'correct': bit, 'compute': 90. if method == 'C' else 100.} for bit in bits]
        margins = grade.paired_margins(rows, values, repetitions=200)
        self.assertAlmostEqual(margins['A_above_highest_baseline']['point'], .25)
        self.assertAlmostEqual(margins['C_above_highest_A_B']['point'], .25)
        self.assertAlmostEqual(margins['C_compute_fraction_saved_vs_B']['point'], .1)
        # In every bootstrap sample, the better complementary baseline is at
        # least50%; its maximum must be recomputed rather than fixing a winner.
        self.assertLessEqual(margins['A_above_highest_baseline']['ci95'][1], .5)


if __name__ == '__main__':
    unittest.main()
