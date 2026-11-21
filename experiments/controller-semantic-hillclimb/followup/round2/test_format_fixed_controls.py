"""Synthetic no-cloud checks for matched formatting, completeness and resumption."""

from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import format_fixed_controls as pilot


class NoReference(dict):
    def __getitem__(self, key):
        if key == 'answer':
            raise AssertionError('Reference access is forbidden in prompt/selection')
        return super().__getitem__(key)


def raw_result(text='Final: B\nA short justification.'):
    return {'done': True, 'model': pilot.old.MODEL, 'response': text, 'prompt_eval_count': 17,
            'eval_count': 8, 'done_reason': 'stop', 'logprobs': [{'token': text, 'logprob': -.1}],
            'srr_backend': {'provider': 'synthetic'}}


class FormatControlTests(unittest.TestCase):
    def row(self, domain='science'):
        return NoReference(id='q', question='Original question\n\nOptions:\nA. a\nB. b',
                           domain=domain, subject='chemistry' if domain == 'science' else 'math', answer='SECRET')

    def test_only_format_sentence_changes_and_retries_are_fresh(self):
        for domain in ('math', 'science'):
            row = self.row(domain); tree = {'': {'text': 'Final: A'}, 'repair': {'text': 'Final: B'}}
            unit = 'one option letter' if domain == 'science' else 'a number'
            before = f'Give at most 90 words of scratch work, then a line "Final: X" where X is {unit}.'
            after = f'Write a line "Final: X" FIRST, where X is {unit}, then at most 35 words justifying it.'
            for path in pilot.PATHS:
                new = pilot.prompt(row, path, tree)
                original = pilot.old.solver_prompt(row, path, tree, {})
                self.assertEqual(new.replace(after, before), original)
                self.assertNotIn('SECRET', new)
            self.assertEqual(pilot.prompt(row, 'retry', tree), pilot.prompt(row, 'retry', {}))
            self.assertEqual(pilot.prompt(row, 'retry/retry', tree), pilot.prompt(row, 'retry/retry', {}))

    def test_original_logical_seeds_and_saved_seed_tamper_rejected(self):
        row = self.row(); text = pilot.prompt(row, 'repair', {'': {'text': 'Final: A'}})
        for path in pilot.CONTROL_PATHS:
            expected = int(hashlib.sha256(f'{pilot.old.SEED}:q:{path}'.encode()).hexdigest()[:8], 16) % 2**31
            self.assertEqual(pilot.logical_seed('q', path), expected)
        record = {'request': pilot.request_identity('q', 'repair', text, 'registration'),
                  'result': raw_result('No valid final line'), 'latency_seconds': 1.}
        result = pilot.replay_node(row, 'repair', text, record, 'registration')
        self.assertEqual(result['answer'], '')
        record['request']['seed'] += 1
        with self.assertRaises(ValueError):
            pilot.replay_node(row, 'repair', text, record, 'registration')

    def training_fixture(self, root):
        data = root / 'data'; mentor = root / 'mentor'; output = root / 'pilot'
        (data / 'splits').mkdir(parents=True); mentor.mkdir()
        rows = [dict(id=f'{subject}-{i}', question=f'Question {subject}-{i}',
                     domain='math' if subject == 'math' else 'science', subject=subject,
                     answer='1' if subject == 'math' else 'A')
                for subject, count in (('math', 256), ('chemistry', 128), ('physics', 128)) for i in range(count)]
        chosen = [dict(id=r['id'], subject=s, bank='new') for s, count in pilot.COUNTS.items()
                  for r in [r for r in rows if r['subject'] == s][:count]]
        chosen.reverse()
        (mentor / 'contract.json').write_text(json.dumps({'selection': chosen}))
        (data / 'splits/train-questions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        receipts = []
        for row in rows:
            tree = {}
            for path in pilot.CONTROL_PATHS:
                text = 'Final: ' + row['answer']
                prompt = pilot.old.solver_prompt({**row, 'answer': ''}, path, tree, {})
                node = {'model': pilot.old.MODEL, 'model_digest': pilot.old.MODEL_DIGEST, 'text': text,
                        'answer': row['answer'], 'seed': pilot.logical_seed(row['id'], path),
                        'temperature': .65 if path else 0, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                        'input_tokens': 10, 'output_tokens': 4, 'finish': 'stop'}
                tree[path] = node
                receipts.append({'id': row['id'], 'kind': 'solver', 'path': path, 'response': node})
        bank = data / 'train-observations.jsonl'
        bank.write_text(''.join(json.dumps(r) + '\n' for r in receipts))
        (data / 'train-plain-complete.json').write_text(json.dumps({'base_records': len(receipts), 'base_bank_sha256': pilot.sha(bank)}))
        return Namespace(data=data, mentor=mentor, output=output), rows, chosen

    def test_selection_uses_all_registered_members_regardless_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            args, rows, chosen = self.training_fixture(Path(folder))
            selected = pilot.selection([NoReference(r) for r in rows], {'selection': chosen})
            self.assertEqual(len(selected), 128)
            changed = [{**r, 'answer': 'totally different'} for r in rows]
            self.assertEqual(selected, pilot.selection(changed, {'selection': chosen}))
            with self.assertRaises(ValueError):
                pilot.selection(rows, {'selection': chosen[:-1]})

    def test_full_grid_resume_preserves_completed_invalid_raw_call_and_scores_only_after_seal(self):
        with tempfile.TemporaryDirectory() as folder:
            args, _, _ = self.training_fixture(Path(folder))
            with patch('builtins.print'):
                contract = pilot.register(args)
            contract, rows, original = pilot.registered_view(args)
            item = contract['selection'][0]; qid = item['id']; text = pilot.prompt(rows[qid], 'repair', original[qid])
            raw = {'id': qid, 'path': 'repair', 'request': pilot.request_identity(qid, 'repair', text, pilot.sha(args.output / 'registration.json')),
                   'result': raw_result('No valid final line'), 'latency_seconds': 1.}
            (args.output / 'raw-calls.jsonl').write_text(json.dumps(raw) + '\n')
            with patch.object(pilot, 'lines', side_effect=AssertionError('No question/reference read before complete')):
                with self.assertRaises(FileNotFoundError):
                    pilot.grade(args)
            calls = []
            def request(endpoint, payload):
                calls.append(copy.deepcopy(payload))
                self.assertEqual(payload['model'], pilot.old.MODEL)
                self.assertEqual(payload['options']['num_predict'], 256)
                self.assertEqual(payload['options']['temperature'], .65)
                return raw_result('Final: 0' if 'a number' in payload['prompt'] else 'Final: B')
            with patch('builtins.print'):
                complete = pilot.collect(args, request=request)
            self.assertEqual(len(calls), 511)
            self.assertEqual(complete['solver_nodes'], 512)
            self.assertEqual(len(pilot.lines(args.output / 'raw-calls.jsonl')), 512)
            saved = pilot.index_records(pilot.lines(args.output / 'train.jsonl'), contract['selection'])
            self.assertEqual(saved[qid, 'repair']['response']['answer'], '')
            with patch('builtins.print'):
                pilot.collect(args, request=lambda *args: self.fail('Resume must not regenerate any completed answer'))
                result = pilot.grade(args)
            self.assertEqual(result['acquisition']['new_solver_calls'], 512)
            self.assertEqual(result['acquisition']['input_tokens'], 512 * 17)
            self.assertAlmostEqual(result['acquisition']['compute'], 512 * (17 + 8) * 4 / 3.09)
            self.assertEqual(result['paired_format_vs_original']['Stop']['losses'], 0)
            self.assertEqual(result['paired_format_vs_original']['Repair once']['losses'], 128)
            self.assertEqual(result['diagnostics']['all']['answer_first']['repair']['valid_answers'], 127)
            self.assertEqual(result['methods']['Answer-first Repair twice']['solver_calls_per_question'], 3)
            with (args.output / 'train.jsonl').open('a') as stream:
                stream.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'Sealed fixed-control input changed'):
                pilot.grade(args)

    def test_partial_or_duplicate_receipts_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bank.jsonl'; path.write_text('{')
            with self.assertRaises(ValueError):
                pilot.lines(path)
        records = [{'id': 'q', 'path': 'repair'}] * 2
        with self.assertRaises(ValueError):
            pilot.index_records(records, [{'id': 'q'}])


if __name__ == '__main__':
    unittest.main()
