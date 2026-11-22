"""Synthetic blind DEV collection and postseal decoding tests only."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import development_complete_actions as run
from test_scoring_completion_availability import canonical
from test_scoring_observation import plain, Remote


def fixture():
    rows = []
    for subject in run.COUNTS:
        for i in range(64):
            safe = canonical(i, subject); safe['id'] = 'dev-'+safe['id']; rows.append(safe)
    return rows, {row['id']: plain(row) for row in rows}


class DevelopmentCompleteTests(unittest.TestCase):
    def test_gate_binds_fixed_training_fit_and_full16case_screen(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); model, train, data = root/'model', root/'train', root/'data'
            model.mkdir(); (train/'graded').mkdir(parents=True)
            (train/'complete.json').write_text('complete'); (train/'registration.json').write_text('registration')
            (model/'completion-A.pkl').write_bytes(b'fixed learned model')
            model_sha = run.sha(model/'completion-A.pkl')
            done = {'model_sha256': model_sha}; run.write(model/'complete.json', done)
            headroom = {'novel': 22, 'plain7_oracle': 96, 'with_complete_oracle': 118}
            record = {'paths': {'data': str(data), 'complete': str(train)}, 'regularization': .1, 'smoothing': 32.,
                'models': 1, 'state_count': 16, 'maximum_additional_calls': 2, 'complete_terminal': True,
                'paid_scoring_features_used': False, 'math_unchanged': True, 'neural_calls': 0, 'cloud_calls': 0,
                'development_or_test_records_read': False, 'training_headroom': headroom,
                'complete_grid_receipt': {'complete_sha256': run.sha(train/'complete.json')}}
            run.write(model/'registration.json', record)
            summary = {'schema': run.actions.SCHEMA, 'questions': 256, 'training_only': True, 'test_used': False,
                'controller_accuracy_claim': False, 'complete_sha256': run.sha(train/'complete.json'),
                'registration_sha256': run.sha(train/'registration.json'),
                'screen': {'criteria': run.actions.SCREEN, 'passed': True},
                'panels': {'all_science': {'original_seven_oracle_correct': 96, 'combined_ten_oracle_correct': 118,
                                         'novel_correct_beyond_original_seven': 22}}}
            run.write(train/'graded/summary.json', summary)
            args = SimpleNamespace(data=data, model=model)
            with patch.object(run.fitting, 'verify_complete', return_value=done):
                gate = run.model_gate(args)
                self.assertEqual(gate['model_sha256'], model_sha)
                summary['screen']['passed'] = False; run.write(train/'graded/summary.json', summary)
                with self.assertRaises(ValueError): run.model_gate(args)
                summary['screen']['passed'] = True; run.write(train/'graded/summary.json', summary)
                record['regularization'] = 1.; run.write(model/'registration.json', record)
                with self.assertRaises(ValueError): run.model_gate(args)

    def test_model_gate_precedes_even_public_dev_loading(self):
        with patch.object(run, 'model_gate', side_effect=ValueError('TRAIN incomplete')), \
                patch.object(run.actions.pilot.protocol, 'read_public_rows', side_effect=AssertionError('DEV opened early')):
            with self.assertRaisesRegex(ValueError, 'TRAIN incomplete'):
                run.public_inputs(SimpleNamespace(data=Path('/unused')))

    def test_registered_penalties_model_and_all_public_requests_cannot_change(self):
        rows, trees = fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory); args = SimpleNamespace(data=root/'data', model=root/'model', output=root/'output')
            evidence = {'model_gate': {'model_sha256': 'fixed-model'}}
            with patch.object(run, 'public_inputs', return_value=(rows, trees, evidence)):
                run.register(args)
                saved = run.read(args.output/'registration.json')
                self.assertEqual(saved['evaluation_penalties'], [0., .01, .02, .04, .08, .16, .24])
                self.assertEqual((saved['questions'], len(saved['requests'])), (128, 384))
                self.assertEqual(len(saved['original_baselines']), 5)
                self.assertEqual(len(saved['new_matched_controls']), 6)
                run.registered_view(args)
                saved['evaluation_penalties'].append(.32); run.write(args.output/'registration.json', saved)
                with self.assertRaises(ValueError): run.registered_view(args)

    def test_full384_grid_and_exact_physical_costs_no_teacher_or_neural(self):
        rows, trees = fixture()
        with TemporaryDirectory() as directory:
            output = Path(directory); remote = Remote(); known = {}
            with (output/'bank').open('a') as bank, (output/'raw').open('a') as raw:
                collector = run.Collector(trees, known, bank, run.actions.pilot.RequestJournal(remote, raw, slots=2))
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(collector.one, [(row, prefix) for row in rows for prefix in run.actions.PREFIXES]))
            physical = run.actions.pilot.protocol.strict_records(output/'raw')
            logical = run.actions.pilot.protocol.strict_records(output/'bank')
            restored, coverage = run.actions.restore(rows, trees, set(), logical, physical, True)
            self.assertEqual(coverage, {'expected_cells': 384, 'physical_calls': 384, 'logical_actions': 384,
                'reused_physical_calls': 0, 'new_physical_calls': 384})
            self.assertEqual(len(remote.calls), 384)
            self.assertEqual(sum(v['response']['input_tokens'] for v in restored.values()), 38400)
            self.assertTrue(all(v['response']['new_neural_calls'] == v['response']['review_calls'] == 0 for v in restored.values()))
            with self.assertRaises(ValueError): run.actions.restore(rows, trees, set(), logical[:-1], physical, True)
            with self.assertRaises(ValueError): run.actions.restore(rows, trees, set(), logical, physical[:-1], True)

    def test_crash_resume_uses_durable_token_once(self):
        rows, trees = fixture(); row = rows[0]
        with TemporaryDirectory() as directory:
            output = Path(directory); remote = Remote()
            with (output/'bank').open('a') as bank, (output/'raw').open('a') as raw:
                collector = run.Collector(trees, {}, bank, run.actions.pilot.RequestJournal(remote, raw))
                with patch.object(run.actions.pilot, 'durable_line', side_effect=RuntimeError('Logical append failed')):
                    with self.assertRaises(RuntimeError): collector.one((row, ''))
            physical = run.actions.pilot.protocol.strict_records(output/'raw')
            with (output/'bank').open('a') as bank, (output/'raw').open('a') as raw:
                journal = run.actions.pilot.RequestJournal(lambda *a: self.fail('No replacement generation'), raw, physical)
                run.Collector(trees, {}, bank, journal).one((row, ''))
            self.assertEqual(len(remote.calls), 1)

    def test_dev_label_decoding_only_after_complete_same_model_grid(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); output=root/'output'; output.mkdir(); data=root/'data'
            run.write(output/'registration.json', {'paths': {'data': str(data), 'model': str(root/'model'), 'output': str(output)}})
            with patch.object(run.actions.pilot.fixed, 'lines', side_effect=AssertionError('No reference decode')):
                with self.assertRaises(FileNotFoundError): run.development_data(data, output)
                with self.assertRaises(ValueError): run.development_data(root/'other-data', output)

    def test_public_loader_skips_invalid_json_reference_field(self):
        rows, _ = fixture()
        with TemporaryDirectory() as directory:
            path = Path(directory)/'questions.jsonl'
            path.write_text('\n'.join(json.dumps({k:v for k,v in row.items() if k!='answer'})[:-1]+
                ',"answer": PRIVATE_REFERENCE}' for row in rows)+'\n')
            self.assertEqual(run.actions.pilot.protocol.read_public_rows(path), rows)

    def test_source_closure_includes_fitter_protocol_and_excludes_tests(self):
        import sys
        sources = run.io.sources(sys.modules[run.__name__])
        self.assertIn(str(Path(run.__file__).resolve()), sources)
        self.assertIn(str(Path(run.fitting.__file__).resolve()), sources)
        self.assertIn(str(Path(run.actions.__file__).resolve()), sources)
        self.assertNotIn(str(Path(__file__).resolve()), sources)


if __name__ == '__main__':
    unittest.main()
