import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import evidence_first_review_runner as run
from test_active_textbook_review import Retriever, ActiveRemote
from test_strong_late_review import fixture


class EvidenceRemote(ActiveRemote):
    def __call__(self, endpoint, payload):
        result = super().__call__(endpoint, payload)
        cap = payload['options']['num_predict']
        if cap == 64:
            result['response'] = 'harmonic oscillator selection rules</search>'
        elif cap == 960:
            result['response'] = 'Use the retrieved governing principle to distinguish the options.'
        return result


class EvidenceFirstTests(unittest.TestCase):
    def test_full_cohort_replay_evidence_timing_and_billing(self):
        rows, trees, *_ = fixture(); remote = EvidenceRemote()
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with run.configured_runner(), patch.object(run.spec.retrieval, 'retriever', return_value=Retriever()), \
                    (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw, (folder/'attempts').open('a') as attempts:
                journal = run.Journal(remote, raw, attempts)
                collector = run.original.Collector(trees, {}, bank, journal)
                for row in rows: collector.one(row)
                records = run.protocol.strict_records(folder/'bank')
                physical = run.protocol.strict_records(folder/'raw')
                _, coverage = run.spec.restore(rows, trees, records, physical, complete=True)
                self.assertEqual(coverage['physical_calls'], 320)
                self.assertEqual(coverage['physical_teacher_phases'], 192)
                self.assertEqual([p['options']['num_predict'] for p in remote.calls[:5]], [64, 960, 512, 128, 128])
                self.assertNotIn('CURRENT SOLUTION', remote.calls[0]['prompt'])
                self.assertNotIn('PUBLIC FACT', remote.calls[0]['prompt'])
                self.assertIn('PUBLIC FACT', remote.calls[1]['prompt'])
                self.assertIn('PUBLIC FACT', remote.calls[2]['prompt'])
                teacher = records[0]['response']
                self.assertEqual(remote.calls[3]['prompt'], run.spec.previous.spec.short_repair(rows[0], trees[rows[0]['id']][''], teacher))
                run.attempts_checked(run.protocol.strict_records(folder/'attempts'), physical)
                billed = run.original.previous.trajectory([], teacher)
                self.assertEqual(billed['physical_review_calls'], 3)
                self.assertEqual(billed['review_input_tokens'], sum(c['input_tokens'] for c in teacher['calls']))
                tampered = copy.deepcopy(records); tampered[0]['response']['retrieval_results'][0]['passages'][0]['text'] = 'ALTERED'
                with self.assertRaises(ValueError): run.spec.restore(rows, trees, tampered, physical, complete=True)

    def test_all_three_phase_costs_are_mandatory(self):
        calls = [{'phase': phase, 'output_cap': cap, 'input_tokens': 10, 'output_tokens': 5}
                 for phase, cap in zip(run.spec.PHASES, (64, 960, 512))]
        review = {'calls': calls, 'physical_api_calls': 3, 'input_tokens': 30, 'output_tokens': 15, 'output_cap': 1536}
        run.spec.validate_review_charge(review)
        for changed in ({**review, 'input_tokens': 20}, {**review, 'calls': calls[1:]},
                        {**review, 'physical_api_calls': 2}, {**review, 'output_cap': 1472}):
            with self.assertRaises(ValueError): run.spec.validate_review_charge(changed)

    def test_partial_physical_attempt_is_not_automatically_retried(self):
        row = fixture()[0][0]
        try: run.spec.review(row, fixture()[1][row['id']], {})
        except run.spec.MissingPhysical as missing: payload = missing.payload
        key = run.original.old.digest({'endpoint': '/api/generate', 'payload': payload})
        attempt = {'schema': run.spec.SCHEMA, 'endpoint': '/api/generate', 'payload': payload,
                   'request_sha256': key, 'payload_sha256': run.original.old.digest(payload)}
        with self.assertRaises(ValueError): run.attempts_checked([attempt], [])


if __name__ == '__main__': unittest.main()
