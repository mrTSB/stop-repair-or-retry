import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

import replay_new_source_mentor as replay


class DevelopmentImportTests(unittest.TestCase):
    def cache(self):
        return {('solver', 'synthetic', '', 'a' * 64, 'paid response'):
                (np.zeros(4), np.zeros(8), np.zeros(8), 12)}

    def test_completed_bank_rejects_changed_panel_partial_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = b'{"id":"synthetic-only","response":{}}\n'
            summary = {'test_used': False, 'development_questions': 256,
                       'bank_sha256': hashlib.sha256(data).hexdigest()}
            replay.write(folder / 'summary.json', summary)
            replay.write(folder / 'identity.json', {'test_used': False, 'dev_questions_sha256': 'panel'})
            bank = folder / 'dev-observations.jsonl'; bank.write_bytes(data)
            self.assertEqual(replay.completed_bank(folder, 'panel')[0], data)
            with self.assertRaises(ValueError):
                replay.completed_bank(folder, 'wrong-panel')
            bank.write_bytes(data.rstrip())
            with self.assertRaises(ValueError):
                replay.completed_bank(folder, 'panel')
            bank.write_bytes(data.replace(b'synthetic-only', b'changed'))
            with self.assertRaises(ValueError):
                replay.completed_bank(folder, 'panel')

    def test_resume_preserves_added_responses_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'replay'
            prefix, cache = b'{"answer":"wrong"}\n', pickle.dumps(self.cache())
            receipt = {'neural_cache_sha256': hashlib.sha256(cache).hexdigest(), 'test_used': False}
            replay.copy_prefix(out, prefix, cache, receipt)
            bank = out / 'dev-observations.jsonl'
            suffix = b'{"answer":"invalid"}\n'; bank.write_bytes(prefix + suffix)
            replay.copy_prefix(out, prefix, cache, receipt)
            self.assertEqual(bank.read_bytes(), prefix + suffix)
            with self.assertRaises(ValueError):
                replay.copy_prefix(out, prefix, cache, {**receipt, 'test_used': True})
            bank.write_bytes(prefix + b'{"partial":')
            with self.assertRaises(ValueError):
                replay.copy_prefix(out, prefix, cache, receipt)
            bank.write_bytes(b'{"answer":"corrected"}\n' + suffix)
            with self.assertRaises(ValueError):
                replay.copy_prefix(out, prefix, cache, receipt)

    def test_unregistered_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            original = out / 'dev-observations.jsonl'; original.write_bytes(b'keep')
            with self.assertRaises(ValueError):
                replay.copy_prefix(out, b'replace', b'cache', {'neural_cache_sha256': 'x'})
            self.assertEqual(original.read_bytes(), b'keep')

    def test_cache_values_and_resumed_imports_cannot_change(self):
        fixture = self.cache(); key = next(iter(fixture))
        bad = {key: (np.zeros(4), np.zeros(8), np.zeros(8), 0)}
        with self.assertRaises(ValueError):
            replay.cache_values(pickle.dumps(bad))
        bad = {key: (np.full(4, np.nan), np.zeros(8), np.zeros(8), 12)}
        with self.assertRaises(ValueError):
            replay.cache_values(pickle.dumps(bad))
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary); payload = pickle.dumps(fixture)
            receipt = {'neural_cache_sha256': hashlib.sha256(payload).hexdigest()}
            replay.copy_prefix(out, b'{}\n', payload, receipt)
            fixture[key] = (np.ones(4), np.zeros(8), np.zeros(8), 12)
            (out / 'private-neural-cache.pkl').write_bytes(pickle.dumps(fixture))
            with self.assertRaises(ValueError):
                replay.copy_prefix(out, b'{}\n', payload, receipt)

    def test_B_is_exact_accuracy_winner_and_cheapest_tie(self):
        ordinary = {'dev': {'accuracy': .8, 'mean_compute': 100.}}
        matched = {'dev': {'accuracy': .8, 'mean_compute': 200.}}
        identity = {'test_used': False, 'dev_questions_sha256': 'panel',
                    'source_sha256': replay.sha(replay.control.__file__),
                    'ordinary_B_source': 'synthetic', 'root_stop_rules': []}
        result = {'test_used': False, 'development_questions': 256,
                  'B': ordinary, 'ordinary_B': ordinary, 'matched_B': matched}
        replay.validate_b(result, identity, 'panel')
        for replacement in (matched, {'dev': {'accuracy': .81, 'mean_compute': 500.}}):
            with self.assertRaises(ValueError):
                replay.validate_b({**result, 'B': replacement}, identity, 'panel')
        with self.assertRaises(ValueError):
            replay.validate_b(result, {**identity, 'test_used': True}, 'panel')

    def test_final_summary_is_idempotently_sealed_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            replay.write(out / 'summary.json', {'bank_sha256': 'synthetic'})
            replay.write(out / 'identity.json', {'test_used': False})
            replay.write(out / 'cache-import.json', {'schema': replay.SCHEMA})
            replay.finalize(out)
            payload = (out / 'summary.json').read_bytes()
            marker = (out / 'replay-complete.json').read_bytes()
            replay.finalize(out)
            self.assertEqual(payload, (out / 'summary.json').read_bytes())
            self.assertEqual(marker, (out / 'replay-complete.json').read_bytes())
            self.assertFalse(replay.read(out / 'summary.json')['final_gap_claim'])
            replay.write(out / 'summary.json', {'bank_sha256': 'changed'})
            with self.assertRaises(ValueError):
                replay.finalize(out)


if __name__ == '__main__':
    unittest.main()
