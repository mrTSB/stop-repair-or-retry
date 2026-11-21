import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import development_retained_path_audit as run


def rows():
    return [{'id': f'q{i}', 'subject': 'math' if i < 128 else 'chemistry' if i < 192 else 'physics',
             'domain': 'math' if i < 128 else 'science', 'question': 'Public question?', 'answer': ''}
            for i in range(256)]


def node():
    return {'answer': 'A', 'text': 'Final: A', 'input_tokens': 10, 'output_tokens': 5}


def fixture(root):
    data = root / 'data'; (data / 'splits').mkdir(parents=True)
    public = rows()
    (data / 'splits/dev-questions.jsonl').write_text(''.join(json.dumps({**r, 'answer': 'A'}) + '\n' for r in public))
    plain = {r['id']: {p: node() for p in run.PLAIN} for r in public}
    missing = {('q128', 'guided'), ('q128', 'guided/reuse'), ('q129', 'retry/reuse')}
    records = []
    for row in public[128:]:
        records.append({'id': row['id'], 'kind': 'judge', 'path': '', 'response': {'answer': 'A', 'verdict': {'check': 'paid'}}})
        for path in run.ADDITIONAL:
            if (row['id'], path) not in missing:
                records.append({'id': row['id'], 'kind': 'solver', 'path': path, 'response': node()})
    payload = ''.join(json.dumps(r) + '\n' for r in records).encode()
    cache = pickle.dumps({('solver', 'q128', '', 'a' * 64, 'paid'): (np.zeros(4), np.zeros(8), np.zeros(8), 10)})
    def view(data, actual):
        trees, judges = copy.deepcopy(plain), {}
        for record in actual:
            if record['kind'] == 'judge':
                judges[record['id'], record['path']] = record['response']
            else:
                trees[record['id']][record['path']] = record['response']
        return copy.deepcopy(public), trees, judges
    evidence = {'source_cache_sha256': hashlib.sha256(cache).hexdigest(),
        'source_raw_sha256': hashlib.sha256(b'').hexdigest(),
        'source_raw_generations': 0,
        'dev_questions_sha256': run.sha(data / 'splits/dev-questions.jsonl'),
        'B_correct': 177, 'B_n': 256, 'minimum_C_correct_above_current_B': 203}
    def source(args):
        current, trees, judges = view(data, records)
        return current, trees, judges, payload, cache, b'', copy.deepcopy(evidence)
    args = SimpleNamespace(data=data, source=root / 'source', source_models=root / 'models',
                           b_reference=root / 'B/summary.json', output=root / 'output')
    return args, source, view, payload, cache


class RetainedPathAuditTest(unittest.TestCase):
    def test_plan_uses_all_science_even_when_existing_answers_wrong_or_invalid(self):
        public = rows(); trees = {r['id']: {p: node() for p in run.PLAIN} for r in public}
        before = run.missing_plan(public, trees)
        for tree in trees.values():
            for value in tree.values():
                value['answer'] = ''; value['text'] = 'invalid'
        after = run.missing_plan(public, trees)
        self.assertEqual(before, after)
        self.assertEqual(before['maximum_new_compact_calls'], 384)
        self.assertEqual(len({r['id'] for r in before['missing']}), 128)
        self.assertEqual(before['required_question_path_cells'], 1280)
        with self.assertRaisesRegex(ValueError, 'all128'):
            run.missing_plan(public[:-1], trees)

    def test_public_parser_does_not_decode_reference_values(self):
        # The intentionally invalid reference token would fail json.loads;
        # the frozen lexical public reader does not decode that field at all.
        line = '{"id":"q","question":"Public?","domain":"science","subject":"physics","answer":SECRET_REFERENCE}'
        public = run.serial.parse_public_line(line)
        self.assertEqual(public['answer'], '')
        self.assertNotIn('SECRET', json.dumps(public))

    def test_acquisition_forbids_teacher_math_reference_and_unregistered_actions(self):
        public = rows(); row = public[128]
        acquire = run.ScienceAcquisition(public, {row['id']: {}}, {}, None, None, None)
        for actual, kind, path in ((row, 'judge', ''), (public[0], 'solver', 'guided'),
                                  ({**row, 'answer': 'SECRET'}, 'solver', 'guided'), (row, 'solver', 'retry/retry')):
            with self.assertRaises(ValueError):
                acquire.acquire(actual, run.previous.study.MissingObservation(kind, path))
        with self.assertRaisesRegex(ValueError, 'existing root'):
            acquire.acquire(row, run.previous.study.MissingObservation('solver', 'guided'))

    def test_registration_retains_entire_bank_and_cache_and_is_immutable(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            args, source, view, payload, cache = fixture(Path(temporary))
            stack.enter_context(patch.object(run, 'source_inputs', side_effect=source))
            stack.enter_context(patch.object(run.previous.fitting, 'source_files', return_value={}))
            contract = run.register(args)
            self.assertEqual(contract['plan']['maximum_new_compact_calls'], 3)
            self.assertEqual((args.output / 'dev-observations.jsonl').read_bytes(), payload)
            self.assertEqual((args.output / 'private-neural-cache.pkl').read_bytes(), cache)
            self.assertEqual(len(contract['science_ids']), 128)
            self.assertEqual(contract['new_teacher_calls'], 0)
            run.register(args)
            with patch.object(run, 'SCHEMA', 'changed'):
                with self.assertRaisesRegex(ValueError, 'Changed registered'):
                    run.register(args)

    def test_partial_tail_and_missing_grid_fail_before_grading(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            args, source, view, _, _ = fixture(Path(temporary))
            stack.enter_context(patch.object(run, 'source_inputs', side_effect=source))
            stack.enter_context(patch.object(run, 'public_view', side_effect=view))
            stack.enter_context(patch.object(run.previous.fitting, 'source_files', return_value={}))
            run.register(args)
            with self.assertRaisesRegex(ValueError, 'sealed1280'):
                run.grade(args)
            self.assertFalse((args.output / 'graded').exists())
            bank = args.output / 'dev-observations.jsonl'
            bank.write_bytes(bank.read_bytes() + b'{"partial":')
            with self.assertRaisesRegex(ValueError, 'partial'):
                run.registered_view(args)

    def test_full1280_grid_raw_phases_invalid_answers_resume_and_grade_gate(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            args, source, view, payload, cache = fixture(Path(temporary))
            stack.enter_context(patch.object(run, 'source_inputs', side_effect=source))
            stack.enter_context(patch.object(run, 'public_view', side_effect=view))
            stack.enter_context(patch.object(run.previous.fitting, 'source_files', return_value={}))
            calls = []
            def request(endpoint, body=None):
                if endpoint == '/api/tags':
                    return {'models': [{'name': run.previous.study.old.MODEL, 'digest': run.previous.study.old.MODEL_DIGEST}]}
                self.assertEqual(body['model'], run.previous.study.old.MODEL)
                self.assertEqual(body['options']['num_predict'], 128)
                calls.append(body)
                return {'done': True, 'model': body['model'], 'prompt_eval_count': 10, 'eval_count': 128,
                        'response': '', 'logprobs': [], 'done_reason': 'length'}
            worker = SimpleNamespace(request=SimpleNamespace(autoscaler=SimpleNamespace(update=lambda **kw: None)))
            modal = SimpleNamespace(Cls=SimpleNamespace(from_name=lambda *args: lambda: worker))
            stack.enter_context(patch.dict('sys.modules', {'modal': modal}))
            stack.enter_context(patch.object(run.previous.study, 'resilient', return_value=request))
            contract = run.register(args)
            run.collect(args)
            self.assertEqual(len(calls), 3)
            saved = run.serial.strict_records(args.output / 'dev-observations.jsonl')
            added = saved[len(payload.splitlines()):]
            self.assertTrue(all(r['response']['answer'] == '' for r in added))
            self.assertEqual(len(added), 3)
            marker = run.read(args.output / 'collection-complete.json')
            self.assertEqual(marker['science_question_path_cells'], 1280)
            self.assertEqual(marker['raw_coverage']['unique_raw_generations_covered'], 3)
            self.assertEqual(marker['new_completed_physical_generations'], 3)
            self.assertEqual(marker['new_compact_output_tokens'], 384)
            self.assertFalse(marker['grades_opened'])
            self.assertEqual((args.output / 'private-neural-cache.pkl').read_bytes(), cache)
            run.collect(args)
            self.assertEqual(len(calls), 3)
            report = run.grade(args)
            self.assertEqual(report['minimum_C_correct_above_current_B'], 203)
            self.assertEqual(report['complete_science_union_correct'], 128)
            self.assertFalse(report['current_single_review_tree_ruled_out_even_with_hindsight'])
            self.assertFalse(report['fresh_validation'])
            self.assertIn('not controller accuracy', report['scope'])
            with self.assertRaisesRegex(ValueError, 'Never overwrite'):
                run.grade(args)
            raw = args.output / 'raw-generations.jsonl'; old = raw.read_bytes()
            raw.write_bytes(old + b'{}\n')
            with self.assertRaises((ValueError, KeyError)):
                run.verify_complete(args)

    def test_missing_science_path_prevents_completion_even_if_cached_answer_correct(self):
        public = rows(); trees = {r['id']: {p: node() for p in run.PATHS} for r in public}
        del trees['q128']['retry/reuse']
        with self.assertRaisesRegex(ValueError, 'all128science'):
            run.completion_payload(SimpleNamespace(output=Path('/unused')), {}, public, trees, [], [])


if __name__ == '__main__':
    unittest.main()
