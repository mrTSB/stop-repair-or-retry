"""Synthetic-only scoring observation contracts; no models or cloud calls."""
import builtins
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import scoring_observation_pilot as run
import scoring_observation_diagnostic as diagnostic
from test_science_mentor_expansion import row, cohort, plain


def response(selected='B', top=None):
    if top is None:
        top = [(' A', .3), ('B', .2), (' C', .1)]
    logprobs = [{'token': token, 'bytes': list(token.encode()), 'logprob': math.log(value)} for token, value in top]
    current = next((item for item in logprobs if item['token'] == selected),
                   {'token': selected, 'bytes': list(selected.encode()), 'logprob': math.log(.05)})
    return {'done': True, 'model': run.protocol.old.MODEL, 'response': selected,
        'prompt_eval_count': 100, 'prompt_eval_cached_count': 80, 'eval_count': 1,
        'srr_backend': {'ollama_version': '0.40.16'},
        'logprobs': [{**current, 'top_logprobs': logprobs}]}


class Remote:
    def __init__(self):
        self.calls = []
    def __call__(self, endpoint, payload):
        assert endpoint == '/api/generate'
        assert 'PRIVATE_REFERENCE' not in payload['prompt']
        assert payload['model'] == run.protocol.old.MODEL
        self.calls.append(copy.deepcopy(payload))
        return response()


def synthetic_data(n=64):
    rows = [row(i, subject) for subject in ('chemistry', 'physics') for i in range(n // 2)]
    trees = {r['id']: plain(r) for r in rows}
    ids = np.asarray([r['id'] for r in rows for _ in run.PATHS])
    subjects = np.asarray([r['subject'] for r in rows for _ in run.PATHS])
    paths = np.asarray(list(run.PATHS) * n)
    numeric = np.random.default_rng(11).normal(size=(3*n, 5))
    labels = np.arange(3*n) % 8
    return rows, trees, {'ids': ids, 'subjects': subjects, 'paths': paths, 'numeric': numeric,
        'labels': labels, 'texts': ['paid synthetic text'] * (3*n)}


class ScoringObservationTests(unittest.TestCase):
    def test_prefix_last_marker_no_future_or_reference_and_exact_action_identity(self):
        safe = row(); tree = plain(safe)
        tree['']['text'] = 'My answer: tentative. More paid reasoning.\nFinal answer: B trailing text'
        prompt, receipt = run.scoring_prompt(safe, '', tree)
        original = run.protocol.old.solver_prompt(safe, '', {'': tree['']}, {})
        self.assertEqual(prompt, original + 'My answer: tentative. More paid reasoning.\nFinal answer:')
        future = copy.deepcopy(tree); future['repair']['text'] = 'FUTURE_SECRET'
        self.assertEqual(run.scoring_prompt(safe, '', future), (prompt, receipt))
        with self.assertRaises(ValueError): run.scoring_prompt({**safe, 'answer': 'PRIVATE_REFERENCE'}, '', tree)
        with self.assertRaises(ValueError): run.scoring_prompt(safe, 'repair/repair', tree)
        altered = copy.deepcopy(tree); altered['']['seed'] += 1
        with self.assertRaises(ValueError): run.scoring_prompt(safe, '', altered)
        tree['']['text'] = 'Unfinished paid calculation'
        self.assertTrue(run.scoring_prompt(safe, '', tree)[0].endswith('Unfinished paid calculation\nFinal:'))
        self.assertFalse(run.scoring_prompt(safe, '', tree)[1]['marker_present'])

    def test_tokens_are_censored_and_duplicate_selected_token_is_not_double_counted(self):
        item = run.feature_record({'answer': 'B'}, {'marker_present': True}, response())
        self.assertAlmostEqual(item['current_letter_mass'], .2)
        self.assertAlmostEqual(item['features'][4], .6)
        self.assertAlmostEqual(item['returned_letter_masses']['A'], .3)
        missing = run.feature_record({'answer': 'J'}, {'marker_present': True}, response())
        self.assertIsNone(missing['current_letter_mass']); self.assertIsNone(missing['current_letter_best_logprob'])
        self.assertFalse(missing['current_letter_observed']); self.assertEqual(missing['features'][11], 0.)
        self.assertNotIn('J', missing['returned_letter_masses']); self.assertIsNone(missing['entropy'])
        alternate = run.feature_record({'answer': 'A'}, {'marker_present': True},
                                      response(' A', [(' A', .3), ('A', .2)]))
        self.assertAlmostEqual(alternate['current_letter_mass'], .5)
        empty = run.feature_record({'answer': ''}, {'marker_present': False}, {'logprobs': []})
        self.assertFalse(empty['original_answer_valid']); self.assertEqual(len(empty['features']), 16)
        bad = response(); bad['logprobs'][0]['logprob'] = -.001
        with self.assertRaises(ValueError): run.alternatives(bad)
        bad = response(); bad['logprobs'][0]['top_logprobs'] *= 7
        with self.assertRaises(ValueError): run.alternatives(bad)

    def test_original_selection_stays_all64_science_regardless_of_outcomes(self):
        rows, contract = cohort()
        chosen = [v for v in run.fixed.selection(rows, contract) if v['subject'] in run.COUNTS]
        changed = [{**r, 'answer': 'PRIVATE_REFERENCE', 'correct': i % 2} for i, r in enumerate(rows)]
        self.assertEqual(chosen, [v for v in run.fixed.selection(changed, contract) if v['subject'] in run.COUNTS])
        self.assertEqual(len(chosen), 64)

    def test_one_token_caps_original_seed_full_input_cost_and_singleflight(self):
        safe = row(); trees = {safe['id']: plain(safe)}
        with TemporaryDirectory() as directory:
            folder = Path(directory); remote = Remote(); known = {}
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw:
                journal = run.RequestJournal(remote, raw, slots=2)
                collector = run.Collector(trees, known, bank, journal)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(collector.one, [(safe, ''), (safe, '')]))
            self.assertEqual(len(remote.calls), 1)
            payload = remote.calls[0]
            self.assertEqual(payload['options'], {'temperature': trees[safe['id']]['']['temperature'],
                'seed': run.fixed.logical_seed(safe['id'], ''), 'num_predict': 1, 'num_ctx': 4096})
            self.assertEqual(payload['top_logprobs'], 20)
            self.assertFalse(payload['truncate']); self.assertFalse(payload['shift'])
            item = known[safe['id'], '']
            self.assertEqual(item['observer_input_tokens'], 100)
            self.assertAlmostEqual(item['observer_compute'], 4*101/3.09)
            self.assertEqual(item['original_answer'], trees[safe['id']]['']['answer'])
            self.assertEqual(item['answer_changing_continuations'], 0)
            raw_rows = run.protocol.strict_records(folder/'raw')
            raw_rows[0]['response']['prompt_eval_cached_count'] = 101
            raw_rows[0]['response_sha256'] = run.digest(raw_rows[0]['response'])
            with self.assertRaises(ValueError): run.restore([safe], trees, [], raw_rows)

    def test_returned_raw_receipt_resumes_without_repeating_forward(self):
        safe = row(); trees = {safe['id']: plain(safe)}
        with TemporaryDirectory() as directory:
            folder = Path(directory); remote = Remote()
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw:
                journal = run.RequestJournal(remote, raw)
                collector = run.Collector(trees, {}, bank, journal)
                with patch.object(run, 'durable_line', side_effect=RuntimeError('Logical append failed')):
                    with self.assertRaises(RuntimeError): collector.one((safe, ''))
            receipts = run.protocol.strict_records(folder/'raw')
            self.assertEqual(len(receipts), 1)
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw:
                journal = run.RequestJournal(lambda *a: self.fail('No replacement inference'), raw, receipts)
                run.Collector(trees, {}, bank, journal).one((safe, ''))
            known, _ = run.restore([safe], trees, run.protocol.strict_records(folder/'bank'), receipts)
            self.assertEqual(len(known), 1); self.assertEqual(len(remote.calls), 1)
            with (folder/'bank').open('a') as stream: stream.write('{partial')
            with self.assertRaisesRegex(ValueError, 'Partial'): run.protocol.strict_records(folder/'bank')

    def test_complete192_grid_required_and_all_cached_nodes_align(self):
        rows, trees, data = synthetic_data()
        with TemporaryDirectory() as directory:
            folder = Path(directory); known = {}
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw:
                collector = run.Collector(trees, known, bank, run.RequestJournal(Remote(), raw))
                for safe in rows:
                    for path in run.PATHS: collector.one((safe, path))
            records, raw = run.protocol.strict_records(folder/'bank'), run.protocol.strict_records(folder/'raw')
            restored, coverage = run.restore(rows, trees, records, raw, True)
            self.assertEqual(coverage, {'expected_calls': 192, 'physical_calls': 192, 'logical_observations': 192})
            with self.assertRaises(ValueError): run.restore(rows, trees, records[:-1], raw, True)
            with self.assertRaises(ValueError): run.restore(rows, trees, records, raw[:-1], True)
            with self.assertRaises(ValueError): run.restore(rows, trees, records+records[:1], raw)
            aligned, indexes, _ = diagnostic.aligned_data(data, rows, trees, restored)
            self.assertEqual(indexes, list(range(192))); self.assertEqual(aligned['scoring'].shape, (192, 16))
            changed = copy.deepcopy(restored); changed[rows[0]['id'], '']['original_answer'] = 'J'
            with self.assertRaises(ValueError): diagnostic.aligned_data(data, rows, trees, changed)

    def test_refs_and_cloud_are_never_opened_before_gate_or_full_seal(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(data=Path(directory)/'data', refined_completion=Path(directory)/'missing', output=Path(directory))
            with patch.object(run.gates, 'public_train', side_effect=AssertionError('No inputs before revised A')):
                with self.assertRaises(ValueError): run.public_inputs(args)
            with patch.object(diagnostic.prior, 'load_inputs', side_effect=AssertionError('No labels before seal')):
                with self.assertRaises(FileNotFoundError): diagnostic.run(args)
                run.write(args.output/'complete.json', {'schema': run.SCHEMA, 'questions': 64})
                with self.assertRaises(ValueError): diagnostic.run(args)
            real_import = builtins.__import__
            def forbidden(name, *a, **kw):
                if name == 'modal': self.fail('No cloud before contract/gate')
                return real_import(name, *a, **kw)
            with patch('builtins.__import__', side_effect=forbidden):
                with self.assertRaises(ValueError): run.collect(args)

    def test_refined_gate_requires_complete_v4_oof_provenance_without_dev_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); data = root/'data'; (data/'splits').mkdir(parents=True)
            for split in ('train', 'dev'): (data/'splits'/f'{split}-questions.jsonl').write_text('not parsed private rows')
            folder = root/'development-refined-observer-v1'; folder.mkdir()
            verifier = root/'verifier-stream-v4'; verifier.mkdir()
            bindings = {'train.jsonl': 'metadata_sha256', 'oof.npz': 'oof_sha256',
                'contract.json': 'training_contract_sha256', 'training-summary.json': 'training_summary_sha256'}
            provenance = {'new_questions': 512, 'all_new_nodes_verified': 8704, 'decision_nodes_used': 1536,
                'development_test_question_overlap': 0, 'fold_recipe_sha256': {}}
            for name, key in bindings.items():
                (verifier/name).write_text('bound bytes'); provenance[key] = run.sha(verifier/name)
            for fold in ('0', '1', '2', 'full'):
                (verifier/fold).mkdir(); (verifier/fold/'recipe.json').write_text('bound recipe')
                provenance['fold_recipe_sha256'][fold] = run.sha(verifier/fold/'recipe.json')
            (folder/'oof-alignment.jsonl').write_text('bound train-only alignment')
            provenance['alignment_sha256'] = run.sha(folder/'oof-alignment.jsonl')
            run.write(folder/'oof-provenance.json', provenance)
            identity = {'test_used': False, 'verifier': str(verifier),
                'source_sha256': run.sha(Path(run.__file__).with_name('development_refined_observer.py')),
                'train_questions_sha256': run.sha(data/'splits/train-questions.jsonl'),
                'dev_questions_sha256': run.sha(data/'splits/dev-questions.jsonl')}
            run.write(folder/'identity.json', identity)
            summary = {'development_questions': 256, 'fitting_questions': 512, 'test_used': False,
                'oof_provenance_sha256': run.sha(folder/'oof-provenance.json')}
            run.write(folder/'summary.json', summary)
            args = SimpleNamespace(data=data, refined_completion=folder/'summary.json')
            self.assertIn(str((folder/'identity.json').resolve()), run.refined_gate(args))
            summary['development_questions'] = 255; run.write(folder/'summary.json', summary)
            with self.assertRaises(ValueError): run.refined_gate(args)
            (folder/'summary.json').write_text('{partial')
            with self.assertRaises(json.JSONDecodeError): run.refined_gate(args)

    def test_grouped_scaling_forecast_excludes_all_held_question_labels(self):
        _, _, data = synthetic_data(24)
        data['scoring'] = np.random.default_rng(13).normal(size=(72, 16))
        first, receipts, _ = diagnostic.crossfit(data, 'numeric_plus_scoring_C1')
        folds = np.asarray([diagnostic.core.fold_for(qid) for qid in data['ids']])
        changed = copy.deepcopy(data); changed['labels'][folds == 0] = (changed['labels'][folds == 0]+3) % 8
        second, changed_receipts, _ = diagnostic.crossfit(changed, 'numeric_plus_scoring_C1')
        np.testing.assert_allclose(first[folds == 0], second[folds == 0], atol=0, rtol=0)
        for field in ('scaler_mean_sha256', 'scaler_scale_sha256', 'classifier_coefficients_sha256', 'classifier_intercept_sha256'):
            self.assertEqual(receipts[0][field], changed_receipts[0][field])
        self.assertTrue(all(r['question_overlap'] == 0 and r['fit_question_weight'] == 3. for r in receipts))
        with diagnostic.prior.no_model_calls():
            with self.assertRaises(AssertionError): diagnostic.prior.perception.Observer.load(None)

    def test_coverage_denominator_keeps_censored_valid_answers_and_invalid_count(self):
        rows, _, data = synthetic_data(4)
        probabilities = np.ones((12, 8))/8
        known = {}
        for i, (qid, path) in enumerate(zip(data['ids'], data['paths'])):
            known[qid, path] = {'original_answer_valid': i != 0, 'current_letter_observed': i > 2,
                'observer_input_tokens': 100, 'observer_output_tokens': 1, 'observer_compute': 4*101/3.09,
                'latency_seconds': .2}
        reports = diagnostic.reports(data, dict(numeric_C1=probabilities, numeric_plus_scoring_C1=probabilities), known)
        coverage = reports['all']['coverage']
        self.assertEqual(coverage['originally_valid_answers'], 11)
        self.assertEqual(coverage['invalid_original_answers'], 1)
        self.assertEqual(coverage['valid_current_letter_observed'], 9)
        self.assertAlmostEqual(coverage['valid_current_letter_coverage'], 9/11)
        self.assertFalse(diagnostic.screen(reports)['passed'])
        self.assertEqual(reports['all']['observation_cost']['physical_calls'], 12)

    def test_source_closure_captures_diagnostic_not_test_and_exact_frozen_blob(self):
        sources = run.source_files()
        self.assertIn(str(Path(diagnostic.__file__).resolve()), sources)
        self.assertIn(str(Path(run.__file__).resolve()), sources)
        self.assertNotIn(str(Path(__file__).resolve()), sources)
        self.assertFalse(any('/.venv/' in path or '/test_' in path for path in sources))
        self.assertEqual(run.frozen_blob()['model_blob_sha256'], run.MODEL_BLOB)


if __name__ == '__main__':
    unittest.main()
