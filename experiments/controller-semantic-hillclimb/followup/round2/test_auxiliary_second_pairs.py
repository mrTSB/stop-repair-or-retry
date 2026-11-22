from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import auxiliary_paired_contract as previous
import auxiliary_second_paired_contract as contract
import collect_auxiliary_pairs as old
import collect_auxiliary_second_pairs as run
from test_auxiliary_pairs import source_fixture, Remote, read_all
from test_scoring_completion_availability import canonical


class SecondPairedBlockTests(unittest.TestCase):
    def test_balanced_next_hash_block_is_disjoint_and_order_invariant(self):
        subjects = {f'{s}-{i}': s for s in contract.COUNTS for i in range(720)}
        first, second = previous.choose(subjects), contract.choose(subjects)
        self.assertFalse({r['id'] for r in first} & {r['id'] for r in second})
        self.assertEqual(Counter(r['subject'] for r in second), contract.COUNTS)
        self.assertEqual(second, contract.choose(dict(reversed(list(subjects.items())))))

    def test_new_source_projection_skips_reference_values_and_protected_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); args = source_fixture(root)
            subjects = {f'supergpqa-{s}-{i}': s for s in contract.COUNTS for i in range(720)}
            prior = root/'auxiliary-science512-pairs-v1'; prior.mkdir()
            run.io.write(prior/'registration.json', {'schema': previous.SCHEMA, 'selection': previous.choose(subjects), 'source_files': {}})
            run.io.jsonl(prior/'public-questions.jsonl', [])
            run.io.write(prior/'complete.json', {'registration_sha256': run.io.sha(prior/'registration.json'),
                'coverage': {'questions': 512, 'physical_calls': 5120}, 'test_used': False, 'review_calls': 0,
                'new_neural_calls': 0, 'digests': {n: run.io.sha(prior/n) for n in ('registration.json', 'public-questions.jsonl')}})
            original = json.loads
            def checked(value, *a, **kw):
                if isinstance(value, bytes) and b'PRIVATE_REFERENCE' in value: self.fail('Reference values decoded')
                return original(value, *a, **kw)
            with run.guarded.training_only_reads(), patch.object(contract.json, 'loads', side_effect=checked):
                rows, selected, receipt = contract.public_inputs(args)
            self.assertEqual(len(rows), 512); self.assertEqual(receipt['prior_paired_overlap'], 0)
            self.assertTrue(all(r['answer'] == '' for r in rows))

    def test_all_ten_requests_match_original_protocol_and_preserve_failed_answers(self):
        row = {**canonical(), 'id': 'new-q'}
        with TemporaryDirectory() as temporary:
            root = Path(temporary); remote = Remote(invalid=True)
            with (root/'raw').open('a') as raw, (root/'attempts').open('a') as attempts, (root/'bank').open('a') as bank:
                journal = run.Journal(remote, raw, attempts)
                collector = run.Collector({}, bank, journal, 'registered'); collector.one(row)
                tree = {}
                for path in contract.PATHS:
                    self.assertEqual(run.payload(row, path, tree), old.payload(row, path, tree))
                    tree[path] = collector.known[row['id'], path]['response']
            logical, raw, attempts = read_all(root)
            known, coverage = run.restore([row], logical, raw, attempts, 'registered', True)
            self.assertEqual(coverage['physical_calls'], 10); self.assertEqual(len(known), 10)
            self.assertEqual(len(remote.calls), 10)


if __name__ == '__main__': unittest.main()
