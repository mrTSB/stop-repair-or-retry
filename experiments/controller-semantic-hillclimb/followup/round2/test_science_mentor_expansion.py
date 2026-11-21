"""Synthetic-only collection tests: no private questions, models, or cloud."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import science_mentor_expansion as run


def row(i=0, subject='chemistry'):
    return {'id': f'synthetic-{subject}-{i}', 'subject': subject,
            'domain': 'math' if subject == 'math' else 'science',
            'question': f'Public synthetic question {i}?\nA. One\nB. Two', 'answer': ''}


def cohort():
    rows = [row(i, subject) for subject, n in (('math', 256), ('chemistry', 128), ('physics', 128)) for i in range(n)]
    selection = [dict(id=r['id'], subject=r['subject'], bank='new') for r in rows
                 if int(r['id'].rsplit('-', 1)[1]) < (64 if r['subject'] == 'math' else 32)]
    selection += [dict(id=f'old-{i}', subject='math', bank='old') for i in range(64)]
    return rows, {'selection': selection}


def plain(row):
    tree = {}
    for path in run.protocol.PLAIN:
        prompt, seed, digest, temp, cap, context = run.protocol.specification(row, 'plain', 'solver', path, tree, {})
        text = 'Final: 2' if row['domain'] == 'math' else 'Final: B'
        tree[path] = {'model': run.protocol.old.MODEL, 'model_digest': digest, 'seed': seed,
            'temperature': temp, 'output_cap': cap, 'context_length': context,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'text': text,
            'answer': run.protocol.old.parse_answer(row['domain'], text), 'input_tokens': 20,
            'output_tokens': 3, 'finish': 'stop', 'confidence': [0.] * 13}
    return tree


class Remote:
    def __init__(self, two_phase=False):
        self.two_phase = two_phase
        self.calls = []
        self.lock = threading.Lock()
        self.active = self.peak = 0
    def __call__(self, endpoint, payload):
        assert endpoint == '/api/generate'
        assert 'PRIVATE_REFERENCE' not in payload['prompt']
        with self.lock:
            self.calls.append(copy.deepcopy(payload)); self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(.001)
        if payload['model'] == run.protocol.old.JUDGE:
            if self.two_phase and payload['options']['num_predict'] == 1024:
                text = 'Checking the public options.'
            elif self.two_phase:
                text = 'A", "p_correct": 0.3, "action": "repair"}'
            else:
                text = '{"answer":"A","p_correct":0.3,"action":"repair"}'
        else:
            # Invalid complete answers must be retained, never resampled.
            text = 'A completed but invalid answer.'
        with self.lock:
            self.active -= 1
        return {'model': payload['model'], 'done': True, 'response': text,
                'prompt_eval_count': 20, 'eval_count': 8, 'done_reason': 'stop',
                'logprobs': [{'token': text, 'logprob': -.2}]}


class ExpansionTests(unittest.TestCase):
    def test_membership_all_remaining_science_is_outcome_independent(self):
        rows, contract = cohort()
        chosen = run.selection(rows, contract)
        self.assertEqual(len(chosen), 192)
        self.assertEqual(Counter(r['subject'] for r in chosen), {'chemistry': 96, 'physics': 96})
        self.assertTrue(all(int(r['id'].rsplit('-', 1)[1]) >= 32 for r in chosen))
        changed = [{**r, 'answer': 'PRIVATE_REFERENCE', 'correct': i % 2, 'confidence': i / 512}
                   for i, r in enumerate(reversed(rows))]
        self.assertEqual(chosen, run.selection(changed, contract))
        self.assertEqual([r['subject'] for r in chosen], ['chemistry', 'physics'] * 96)
        altered = copy.deepcopy(contract); altered['selection'][0]['id'] = 'not-a-training-question'
        with self.assertRaises(ValueError): run.selection(rows, altered)

    def test_public_loader_lexically_skips_reference_values(self):
        rows, _ = cohort()
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary); (data / 'splits').mkdir()
            # Reference intentionally is not valid JSON; the lexical public parser
            # must never invoke JSON decoding on this private field.
            lines = [json.dumps({k: v for k, v in r.items() if k != 'answer'})[:-1] + ',"answer": PRIVATE_REFERENCE}' for r in rows]
            (data / 'splits/train-questions.jsonl').write_text('\n'.join(lines) + '\n')
            self.assertEqual(run.public_train(data), rows)

    def test_real_prompt_receipts_one_or_two_review_phases_and_exact_alias(self):
        for two_phase in (False, True):
            with self.subTest(two_phase=two_phase), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary); safe = row(); controls = {safe['id']: plain(safe)}
                trees, reviews, known, _ = run.restore([safe], controls, [], [])
                remote = Remote(two_phase)
                with (directory/'bank').open('a') as stream, (directory/'raw').open('a') as raw:
                    journal = run.RequestJournal(remote, raw)
                    collector = run.PairCollector([safe], trees, reviews, known, stream, journal)
                    collector.one(safe); collector.one(safe)
                records = run.protocol.strict_records(directory/'bank'); raw = run.protocol.strict_records(directory/'raw')
                _, _, restored, coverage = run.restore([safe], controls, records, raw)
                self.assertEqual(len(remote.calls), 4 + int(two_phase))
                self.assertEqual(len(restored), 5)
                self.assertEqual(len(raw), len(remote.calls))
                self.assertEqual(coverage['logical_response_nodes'], 4)
                self.assertEqual(records[0]['response']['physical_api_calls'], 1 + int(two_phase))
                self.assertEqual(records[-1]['response'], controls[safe['id']]['repair/retry'])
                self.assertTrue(records[-1]['exact_control_reused'])
                self.assertEqual([r['response']['answer'] for r in records[1:4]], ['', '', ''])
                compact = [p for p in remote.calls if p['model'] == run.protocol.old.MODEL]
                self.assertEqual(len(compact), 3)
                self.assertTrue(all(p['options']['num_predict'] == 128 and p['options']['temperature'] == .65 for p in compact))
                for record in records[1:4]:
                    self.assertEqual(record['response']['seed'], run.mentor_pairs.seed_for(safe['id'], record['path']))

    def test_question_parallelism_and_duplicate_submission_are_singleflight(self):
        rows = [row(i) for i in range(8)]; controls = {r['id']: plain(r) for r in rows}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); remote = Remote()
            trees, reviews, known, _ = run.restore(rows, controls, [], [])
            with (directory/'bank').open('a') as stream, (directory/'raw').open('a') as raw:
                journal = run.RequestJournal(remote, raw, slots=4)
                collector = run.PairCollector(rows, trees, reviews, known, stream, journal)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(collector.one, rows + rows))
            records = run.protocol.strict_records(directory/'bank')
            run.restore(rows, controls, records, run.protocol.strict_records(directory/'raw'))
            self.assertEqual(len(records), 40); self.assertEqual(len(remote.calls), 32)
            self.assertGreater(remote.peak, 1); self.assertLessEqual(remote.peak, 4)
            for r in rows:
                self.assertEqual([(v['kind'], v['path']) for v in records if v['id'] == r['id']], list(run.PATHS))

    def test_returned_response_survives_failure_before_logical_append(self):
        safe = row(); controls = {safe['id']: plain(safe)}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); remote = Remote()
            trees, reviews, known, _ = run.restore([safe], controls, [], [])
            with (directory/'bank').open('a') as stream, (directory/'raw').open('a') as raw:
                journal = run.RequestJournal(remote, raw)
                collector = run.PairCollector([safe], trees, reviews, known, stream, journal)
                with patch.object(run, 'durable_line', side_effect=RuntimeError('simulated write interruption')):
                    with self.assertRaises(RuntimeError): collector.one(safe)
                self.assertTrue(collector.stopped.is_set())
            saved_raw = run.protocol.strict_records(directory/'raw')
            self.assertEqual(len(saved_raw), 1)
            self.assertEqual(run.protocol.strict_records(directory/'bank'), [])
            trees, reviews, known, _ = run.restore([safe], controls, [], saved_raw)
            with (directory/'bank').open('a') as stream, (directory/'raw').open('a') as raw:
                journal = run.RequestJournal(remote, raw, saved_raw)
                run.PairCollector([safe], trees, reviews, known, stream, journal).one(safe)
            self.assertEqual(len(remote.calls), 4)
            run.restore([safe], controls, run.protocol.strict_records(directory/'bank'), run.protocol.strict_records(directory/'raw'))

    def test_resume_rejects_tampered_prompt_missing_raw_duplicate_and_partial_tail(self):
        safe = row(); controls = {safe['id']: plain(safe)}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); trees, reviews, known, _ = run.restore([safe], controls, [], [])
            with (directory/'bank').open('a') as stream, (directory/'raw').open('a') as raw:
                run.PairCollector([safe], trees, reviews, known, stream, run.RequestJournal(Remote(), raw)).one(safe)
            records = run.protocol.strict_records(directory/'bank'); raw = run.protocol.strict_records(directory/'raw')
            with self.assertRaisesRegex(ValueError, 'raw receipt'): run.restore([safe], controls, records, [])
            with self.assertRaisesRegex(ValueError, 'duplicate'): run.restore([safe], controls, records + records[:1], raw)
            changed = copy.deepcopy(records); changed[1]['response']['seed'] += 1
            with self.assertRaisesRegex(ValueError, 'seed'): run.restore([safe], controls, changed, raw)
            changed = copy.deepcopy(records); changed[-1]['response']['finish'] = 'length'
            with self.assertRaisesRegex(ValueError, 'exactly'): run.restore([safe], controls, changed, raw)
            (directory/'bank').write_bytes((directory/'bank').read_bytes() + b'{"partial":')
            with self.assertRaisesRegex(ValueError, 'Partial'): run.protocol.strict_records(directory/'bank')

    def test_private_reference_rejected_before_request(self):
        safe = row(); trees = {safe['id']: plain(safe)}
        collector = run.PairCollector([safe], trees, {}, {}, None, lambda *_: self.fail('No request'))
        with self.assertRaisesRegex(ValueError, 'blank'): collector.one({**safe, 'answer': 'PRIVATE_REFERENCE'})

    def fixture_gates(self, root):
        data = root/'data'; data.mkdir(); bdir = root/'b'; bdir.mkdir(); priority = root/'metadata'; priority.mkdir()
        record = {'solver_digest': 'solver', 'reviewer_digest': 'reviewer', 'train_question_sha256': 'train-panel'}
        args = SimpleNamespace(data=data, b_reference=bdir/'summary.json', priority_completion=[priority/'metadata-replay-complete.json'])
        for split, count in (('train', 512), ('dev', 256)):
            contract = {'mode': 'ordinary', 'source_sha256': run.sha(run.HERE/'collect.py'),
                'solver_digest': 'solver', 'reviewer_digest': 'reviewer', 'split_sha256': split+'-panel'}
            run.write(data/(split+'-ordinary-contract.json'), contract)
            for suffix in ('-observations.jsonl', '-late-observations.jsonl'):
                (data/(split+suffix)).write_bytes(b'opaque bank bytes: never decode me\n')
            run.write(data/(split+'-ordinary-complete.json'), {'questions': count, 'base_records': count*11,
                'late_records': count*6, 'references_supplied': False,
                'contract_sha256': run.sha(data/(split+'-ordinary-contract.json')),
                'base_bank_sha256': run.sha(data/(split+'-observations.jsonl')),
                'late_bank_sha256': run.sha(data/(split+'-late-observations.jsonl'))})
        (bdir/'dev-observations.jsonl').write_bytes(b'also opaque\n')
        run.write(args.b_reference, {'bank_sha256': run.sha(bdir/'dev-observations.jsonl')})
        run.write(bdir/'identity.json', {'fixture': True})
        files = ('registration.json','identity.json','cache-import.json','summary.json','search.json',
                 'dev-observations.jsonl','private-neural-cache.pkl','raw-generations.jsonl')
        for name in files: (priority/name).write_bytes(b'opaque sealed artifact\n')
        run.write(args.priority_completion[0], {'schema': 'metadata-review-development-v1',
            'source_sha256': run.sha(run.HERE/'development_metadata_review.py'),
            'files': {name: run.sha(priority/name) for name in files}})
        return args, record

    def test_all_priority_gates_and_opaque_bank_hashes_before_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, record = self.fixture_gates(Path(temporary))
            with patch.object(run.controls, 'validate_b') as validate:
                receipts, missing = run.prerequisite_receipts(args, record)
                self.assertEqual(missing, []); self.assertIn('matched_B', receipts)
                self.assertEqual(validate.call_args.args[2], 'dev-panel')
                args.priority_completion[0].unlink()
                _, missing = run.prerequisite_receipts(args, record)
                self.assertEqual(missing, [str(args.priority_completion[0])])
                (args.data/'train-ordinary-complete.json').unlink()
                _, missing = run.prerequisite_receipts(args, record)
                self.assertEqual(len(missing), 2)

    def test_changed_completed_banks_priority_seals_and_counts_fail_closed(self):
        for tamper in ('dev-bank', 'metadata-cache', 'count'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temporary:
                args, record = self.fixture_gates(Path(temporary))
                if tamper == 'dev-bank': (args.data/'dev-observations.jsonl').write_bytes(b'changed')
                elif tamper == 'metadata-cache': (args.priority_completion[0].parent/'private-neural-cache.pkl').write_bytes(b'changed')
                else:
                    path = args.data/'train-ordinary-complete.json'; value = run.read(path); value['questions'] -= 1; run.write(path, value)
                with patch.object(run.controls, 'validate_b'), self.assertRaises(ValueError):
                    run.prerequisite_receipts(args, record)

    def test_unfinished_gates_prevent_even_importing_cloud_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary); record = {'source_files': {}}
            run.write(output/'registration.json', record)
            args = SimpleNamespace(output=output, registration_sha256=run.sha(output/'registration.json'), wait=False)
            with patch.object(run, 'registration', return_value=record), patch.object(run, 'prerequisite_receipts', return_value=({}, ['unfinished'])), patch.dict('sys.modules', {'modal': None}):
                with self.assertRaisesRegex(ValueError, 'unfinished'): run.collect(args)
            self.assertFalse((output/'train.jsonl').exists())

    def test_registration_requires_specific_final_metadata_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(data=root/'data', priority_completion=[])
            with patch.object(run, 'inputs', side_effect=AssertionError('No question read before gate validation')):
                with self.assertRaisesRegex(ValueError, 'metadata-newsource'): run.registration(args)
                args.priority_completion = [root/'development-metadata-mixed-v1/metadata-replay-complete.json']
                with self.assertRaisesRegex(ValueError, 'metadata-newsource'): run.registration(args)

    def test_source_closure_binds_runtime_and_excludes_unrelated_mutable_training(self):
        import test_development_metadata_review  # Loaded does not make a dependency.
        paths = set(run.source_manifest())
        self.assertIn(str(Path(run.__file__).resolve()), paths)
        self.assertIn(str((run.HERE.parent/'modal_solver.py').resolve()), paths)
        self.assertIn(str((run.HERE/'development_metadata_review.py').resolve()), paths)
        self.assertTrue(any(p.endswith('/transfer_variants.py') for p in paths))
        self.assertFalse(any('site-packages' in p or '/.venv/' in p or Path(p).name.startswith('test_') for p in paths))
        self.assertFalse(any(Path(p).name in ('train_science_listwise.py', 'science_training_checkpoint.py') for p in paths))


if __name__ == '__main__':
    unittest.main()
