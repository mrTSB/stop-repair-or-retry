import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import contrastive_review_runner as run
from test_science_mentor_expansion import Remote
from test_strong_late_review import fixture


class ContrastiveRemote(Remote):
    def __call__(self, endpoint, payload):
        result = super().__call__(endpoint, payload)
        if payload['options']['num_predict'] == 640:
            result['response'] = 'PAID_AUDIT_CONTENT: optionA violates the limiting case.'
        elif payload['options']['num_predict'] == 384:
            result['response'] = 'PAID_COUNTEREXAMPLE_CONTENT: optionB survives the check.'
        return result


class ContrastiveTests(unittest.TestCase):
    def test_full_cohort_prefix_visibility_and_all_physical_charges(self):
        rows, trees, *_ = fixture()
        remote = ContrastiveRemote(two_phase=True)
        with TemporaryDirectory() as tmp, run.configured_runner():
            folder = Path(tmp)
            with (folder/'bank').open('a') as bank, (folder/'raw').open('a') as raw, (folder/'attempts').open('a') as attempts:
                journal = run.infrastructure.Journal(remote, raw, attempts)
                collector = run.original.Collector(trees, {}, bank, journal)
                for row in rows: collector.one(row)
            records = run.protocol.strict_records(folder/'bank')
            physical = run.protocol.strict_records(folder/'raw')
            _, coverage = run.spec.restore(rows, trees, records, physical, complete=True)
            self.assertEqual(coverage['physical_calls'], 320)
            self.assertEqual(coverage['physical_teacher_phases'], 192)
            self.assertEqual([p['options']['num_predict'] for p in remote.calls[:5]], [640, 384, 512, 128, 128])
            self.assertNotIn('CURRENT SOLUTION', remote.calls[0]['prompt'])
            self.assertIn('PAID_AUDIT_CONTENT', remote.calls[1]['prompt'])
            self.assertNotIn('PAID_COUNTEREXAMPLE_CONTENT', remote.calls[1]['prompt'])
            self.assertIn('PAID_COUNTEREXAMPLE_CONTENT', remote.calls[2]['prompt'])
            teacher = records[0]['response']
            billed = run.original.previous.trajectory([], teacher)
            self.assertEqual(billed['physical_review_calls'], 3)
            self.assertEqual(billed['review_input_tokens'], sum(c['input_tokens'] for c in teacher['calls']))
            self.assertEqual(remote.calls[3]['prompt'], run.spec.previous.spec.short_repair(rows[0], trees[rows[0]['id']][''], teacher))
            run.infrastructure.attempts_checked(run.protocol.strict_records(folder/'attempts'), physical)
            tampered = copy.deepcopy(records)
            tampered[0]['response']['calls'][1]['text'] = 'UNPAID_ANALYSIS'
            with self.assertRaises(ValueError): run.spec.restore(rows, trees, tampered, physical, complete=True)
            with self.assertRaises(ValueError): run.spec.restore(rows, trees, records[:-1], physical, complete=True)

    def test_no_future_or_reference_values_change_first_request(self):
        row, trees = fixture()[0][0], fixture()[1]
        tree = trees[row['id']]
        with self.assertRaises(run.spec.MissingPhysical) as first:
            run.spec.review(row, tree, {})
        changed = copy.deepcopy(tree)
        for node in changed.values(): node['text'] = 'UNPAID_ALTERNATIVE_CONTENT'
        with self.assertRaises(run.spec.MissingPhysical) as second:
            run.spec.review(row, changed, {})
        self.assertEqual(first.exception.payload, second.exception.payload)
        with self.assertRaises(ValueError): run.spec.review({**row, 'answer': 'PRIVATE_LABEL'}, tree, {})

    def test_phase_omission_and_unresolved_attempt_fail_closed(self):
        calls = [{'phase': name, 'output_cap': cap, 'input_tokens': 10, 'output_tokens': 5}
                 for name, cap in zip(run.spec.PHASES, run.spec.CAPS)]
        teacher = {'calls': calls, 'physical_api_calls': 3, 'input_tokens': 30, 'output_tokens': 15, 'output_cap': 1536}
        run.spec.validate_review_charge(teacher)
        with self.assertRaises(ValueError): run.spec.validate_review_charge({**teacher, 'input_tokens': 20})
        with self.assertRaises(ValueError): run.spec.validate_review_charge({**teacher, 'calls': calls[:2]})
        row, trees = fixture()[0][0], fixture()[1]
        with self.assertRaises(run.spec.MissingPhysical) as missing: run.spec.review(row, trees[row['id']], {})
        payload = missing.exception.payload
        key = run.original.old.digest({'endpoint': '/api/generate', 'payload': payload})
        attempt = {'schema': run.spec.SCHEMA, 'endpoint': '/api/generate', 'payload': payload,
                   'request_sha256': key, 'payload_sha256': run.original.old.digest(payload)}
        with run.configured_runner(), self.assertRaises(ValueError):
            run.infrastructure.attempts_checked([attempt], [])


if __name__ == '__main__': unittest.main()
