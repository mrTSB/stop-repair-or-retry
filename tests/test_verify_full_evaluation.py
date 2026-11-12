"""Offline verifier regressions; all controller responses are synthetic."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stop_repair_retry import llm_experiment as experiment

SCRIPT = experiment.ROOT / 'experiments/llm-controller/verify_full_evaluation.py'
spec = importlib.util.spec_from_file_location('portable_controller_verifier', SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
score_spec = importlib.util.spec_from_file_location('publication_scorer', experiment.DEFAULT_DATA / 'scoring.py')
scorer = importlib.util.module_from_spec(score_spec)
score_spec.loader.exec_module(scorer)


class PortableControllerVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        rows = experiment.load_dataset(experiment.DEFAULT_DATA, 'dev')
        self.rows = [next(r for r in rows if r['domain'] == d) for d in ['math', 'science']]
        config = experiment.load_config(experiment.ROOT / 'experiments/llm-controller/config.example.json')
        config.update(endpoint='https://offline.invalid', model='offline@revision', billing_mode='compute', compute_budget_confirmed=True)
        self.exp = experiment.fingerprint(config, experiment.DEFAULT_PROMPT.read_text(), experiment.DEFAULT_DATA, 'live')
        self.backend = {'reported_model': 'offline@revision', 'revision': 'revision'}
        manifest = {'experiment': self.exp, 'dataset': '/unavailable/original/path', 'split': 'dev',
                    'question_ids': [r['id'] for r in self.rows]}
        experiment.save(self.run / 'manifest.json', manifest)
        self.events = []
        details = []
        for row in self.rows:
            body = experiment.request_body(row, '', config, self.exp['prompt'])
            rid = experiment.digest([row['id'], ''])
            response = {'model': 'offline@revision', 'choices': [{'message': {'content': '{"action":"stop"}'}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 20, 'completion_tokens': 6, 'total_tokens': 26},
                        'measurement': {'model_revision': 'revision', 'formatted_prompt': '\n'.join(m['content'] for m in body['messages'])}}
            self.events += [{'type': 'request', 'request_id': rid, 'question_id': row['id'], 'path': '', 'body': body, 'created_utc': '2026-01-01T00:00:00+00:00'},
                            {'type': 'raw_response', 'request_id': rid, 'response': response, 'latency_seconds': .1, 'created_utc': '2026-01-01T00:00:01+00:00'},
                            {'type': 'response', 'request_id': rid, 'response': response, 'latency_seconds': .1, 'created_utc': '2026-01-01T00:00:01+00:00'}]
            node = row['tree']['']
            details.append({'id': row['id'], 'domain': row['domain'], 'policy': 'llm_controller', 'actions': ['stop'], 'path': '',
                            'answer': node['answer'], 'correct': int(node['correct']), 'initial_correct': int(node['correct']),
                            'solver_input_tokens': node['input_tokens'], 'solver_output_tokens': node['output_tokens'],
                            'controller_input_tokens': 20, 'controller_output_tokens': 6,
                            'tokens': node['input_tokens'] + node['output_tokens'] + 26, 'controller_calls': 1,
                            'invalid_actions': 0, 'controller_latency_seconds': .1})
        self.write_events()
        # Generate fixture summary via production arithmetic, without reading absolute paths or making calls.
        with patch.object(experiment, 'completed_records', return_value=(manifest, self.rows, details)), \
             patch.object(experiment, 'journal', return_value=({}, {'synthetic': {'reported_model': 'offline@revision'}}, set(), 0)):
            experiment.evaluate(self.run)

    def tearDown(self):
        self.temp.cleanup()

    def write_events(self):
        (self.run / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in self.events))

    def verify(self):
        return verifier.verify_split(self.run, self.rows, self.exp, self.backend, scorer)

    def test_portable_split_replay_ignores_original_absolute_path(self):
        result = self.verify()
        self.assertEqual(result['questions'], 2)
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['token_totals']['controller_input'], 40)

    def test_tampered_visited_request_rejected(self):
        self.events[0]['body']['messages'][1]['content'] += ' injected gold'
        self.write_events()
        with self.assertRaises(ValueError):
            self.verify()

    def test_changed_token_result_rejected(self):
        p = self.run / 'controller-results.json'
        details = json.loads(p.read_text()); details[0]['tokens'] += 1
        experiment.save(p, details)
        with self.assertRaises(ValueError):
            self.verify()

    def test_changed_bootstrap_endpoint_rejected(self):
        path = self.run / 'summary.json'
        summary = json.loads(path.read_text())
        summary['summaries'][0]['accuracy_ci95'][0] += .01
        experiment.save(path, summary)
        with self.assertRaises(ValueError):
            self.verify()

    def test_duplicate_request_rejected(self):
        self.events.insert(1, self.events[0])
        self.write_events()
        with self.assertRaises(ValueError):
            self.verify()

    def test_unresolved_journal_rejected(self):
        self.events.pop()
        self.write_events()
        with self.assertRaises(ValueError):
            self.verify()


if __name__ == '__main__':
    unittest.main()
