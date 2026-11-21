"""Regression checks for counterfactual alignment and independent-panel uncertainty."""

from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from medium_pomdp import merged_observations
from banks import canonical_review
from confirmation_common import BASELINES, margins, verify_fixed_prefix


class CounterfactualTests(unittest.TestCase):
    def test_fixed_baselines_allow_append_but_reject_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / 'test-observations.jsonl'
            original = b'{"fixed":1}\n'
            path.write_bytes(original)
            frozen = {'fixed_cache_bank_bytes': len(original),
                      'fixed_cache_bank_sha256': hashlib.sha256(original).hexdigest()}
            verify_fixed_prefix(frozen, folder)
            path.write_bytes(original + b'{"adaptive":2}\n')
            verify_fixed_prefix(frozen, folder)
            path.write_bytes(b'{"fixed":0}\n' + b'{"adaptive":2}\n')
            with self.assertRaises(ValueError):
                verify_fixed_prefix(frozen, folder)

    def test_replacement_keeps_logits_aligned_without_stale_duplicate(self):
        metadata = [{'id': 'a', 'kind': 'solver', 'path': ''},
                    {'id': 'a', 'kind': 'solver', 'path': 'repair/reviewed'},
                    {'id': 'b', 'kind': 'solver', 'path': ''}]
        old = {k: np.full((3, width), 0.) for k, width in (('state', 4), ('plain', 8), ('guided', 8))}
        for value in old.values():
            value[:, 0] = [10., 20., 30.]
        additional = [metadata[1], {'id': 'a', 'kind': 'late-review', 'path': 'repair'}]
        fresh = {k: np.full((2, width), 0.) for k, width in (('state', 4), ('plain', 8), ('guided', 8))}
        for value in fresh.values():
            value[:, 0] = [200., 400.]
        rows, values = merged_observations(metadata, old, additional, fresh)
        self.assertEqual([r['id'] for r in rows], ['a', 'b', 'a', 'a'])
        for value in values.values():
            np.testing.assert_array_equal(value[:, 0], [10., 30., 200., 400.])
        with self.assertRaises(ValueError):
            merged_observations(metadata, old, [additional[0], additional[0]], fresh)

    def test_aggregate_finish_is_derived_without_mutating_receipt(self):
        original = {'calls': [{'finish': 'length'}, {'finish': 'stop'}]}
        derived = canonical_review(original)
        self.assertEqual(derived['finish'], 'stop')
        self.assertNotIn('finish', original)
        with self.assertRaises(ValueError):
            canonical_review({'calls': []})


class BootstrapTests(unittest.TestCase):
    def test_source_mix_is_fixed_inside_each_paired_draw(self):
        rows = [{'id': f'{panel}-{subject}-{i}', 'subject': subject}
                for panel in ('same', 'transfer') for subject in ('math', 'chemistry', 'physics') for i in range(3)]
        panels = {r['id']: r['id'].split('-')[0] for r in rows}
        results = {}
        for name in (*BASELINES, 'A', 'B', 'C'):
            values = []
            for r in rows:
                # Each stratum is homogeneous; a valid stratified draw has
                # exactly the same aggregate even though source effects differ.
                success = 0 if name in BASELINES else int(panels[r['id']] == 'same')
                if name == 'C':
                    success = 1
                values.append({'correct': success, 'compute': 1. if name == 'C' else 2.})
            results[name] = values
        measured = margins(rows, results, panels, repetitions=200)
        for name in ('A_above_highest_baseline', 'B_above_highest_baseline', 'C_above_highest_A_B'):
            self.assertEqual(measured[name], {'point': .5, 'ci95': [.5, .5]})
        self.assertEqual(measured['C_compute_fraction_saved_vs_B'], {'point': .5, 'ci95': [.5, .5]})


if __name__ == '__main__':
    unittest.main()
