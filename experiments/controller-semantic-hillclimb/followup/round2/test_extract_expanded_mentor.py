import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import extract_expanded_mentor as run
import expanded_mentor_io as io
import test_expanded_mentor_inputs as fixtures


def setup(directory):
    _, _, _, metadata, inputs, old, arrays, new_ids = fixtures.grid()
    reused, missing = run.inputs.alignment(metadata, inputs, old, arrays, new_ids)
    plan = run.inputs.extraction_plan(metadata, inputs, reused, missing, 'synthetic-recipe')
    bundle = {'metadata': metadata, 'inputs': inputs, 'cached_outputs': arrays,
        'reused_indexes': reused, 'missing_indexes': missing, 'identity': {'synthetic': True},
        'extraction_plan': plan}
    record = {'schema': run.SCHEMA, 'input_identity': bundle['identity'], 'source_files': {}, 'asset_files': {},
              'max_length': 512, 'device': 'cpu', 'runtime_recipe_sha256': 'synthetic-recipe',
              'chunk_size': run.CHUNK, 'batch_size': run.BATCH, 'output': str(directory.resolve())}
    io.write(directory/'registration.json', record); io.write(directory/'extraction-plan.json', plan)
    return bundle, record, io.sha(directory/'registration.json')


def values(entries):
    return {name: np.full((len(entries), width), 7.5, dtype=np.float32) for name, width in run.inputs.WIDTHS.items()}, len(entries)*512


class ExtractionTests(unittest.TestCase):
    def test_interrupted_resume_reuses_every_completed_chunk_and_parent_row(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary); bundle, record, sha = setup(output)
            calls = []
            def interrupted(entries):
                calls.append(entries)
                if len(calls) == 3: raise RuntimeError('synthetic interruption')
                return values(entries)
            with self.assertRaises(RuntimeError): run.extract_arrays(output, bundle, record, sha, interrupted)
            first = {p.name: io.sha(p) for p in (output/'chunks').iterdir()}
            resumed = []
            def forward(entries): resumed.append(entries); return values(entries)
            arrays, complete = run.extract_arrays(output, bundle, record, sha, forward)
            self.assertEqual(sum(map(len, resumed)), 960-64)
            self.assertTrue(all(io.sha(output/'chunks'/name) == value for name, value in first.items()))
            run.preserve_parent(bundle, arrays)
            self.assertEqual((complete['n'], complete['reused'], complete['new']), (5184, 4224, 960))
            def forbidden(entries): raise AssertionError('Completed extraction was repeated')
            final, _ = run.extract_arrays(output, bundle, record, sha, forbidden)
            for name in arrays: np.testing.assert_array_equal(final[name], arrays[name])
            attempts = [json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['attempted_rows'] for r in attempts), 992)

    def test_orphan_and_malformed_attempt_fail_before_forward(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary); bundle, record, sha = setup(output)
            (output/'chunks').mkdir()
            io.npz(output/'chunks/000.npz', values(bundle['inputs'][:32])[0])
            with self.assertRaisesRegex(ValueError, 'incomplete chunk'):
                run.extract_arrays(output, bundle, record, sha, lambda _: self.fail('unexpected forward'))
        with TemporaryDirectory() as temporary:
            output = Path(temporary); bundle, record, sha = setup(output)
            (output/'chunks').mkdir()
            (output/'chunks/000.npz.tmp').write_bytes(b'completed-but-not-renamed')
            with self.assertRaisesRegex(ValueError, 'temporary'):
                run.extract_arrays(output, bundle, record, sha, lambda _: self.fail('unexpected forward'))
            self.assertEqual((output/'chunks/000.npz.tmp').read_bytes(), b'completed-but-not-renamed')
        with TemporaryDirectory() as temporary:
            output = Path(temporary); bundle, record, sha = setup(output)
            (output/'attempts.jsonl').write_text('{"partial":')
            with self.assertRaisesRegex(ValueError, 'malformed'):
                run.extract_arrays(output, bundle, record, sha, lambda _: self.fail('unexpected forward'))

    def test_completed_grid_rejects_omitted_seal_and_rehashed_array_tampering(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary); bundle, record, sha = setup(output)
            arrays, complete = run.extract_arrays(output, bundle, record, sha, values)
            changed = copy.deepcopy(complete); del changed['digests']['chunks/000.json']
            io.write(output/'neural-complete.json', changed)
            with self.assertRaisesRegex(ValueError, 'omitted'):
                run.verify_complete(output, bundle, sha)
            changed = copy.deepcopy(complete)
            altered = {name: value.copy() for name, value in arrays.items()}
            altered['state'][bundle['missing_indexes'][0], 0] += 1
            io.npz(output/'neural.npz', altered)
            changed['digests']['neural.npz'] = io.sha(output/'neural.npz')
            semantic = io.read(output/'neural-receipt.json'); semantic['outputs_sha256'] = io.sha(output/'neural.npz')
            io.write(output/'neural-receipt.json', semantic)
            changed['digests']['neural-receipt.json'] = io.sha(output/'neural-receipt.json')
            io.write(output/'neural-complete.json', changed)
            with self.assertRaisesRegex(ValueError, 'sealed new chunks'):
                run.verify_complete(output, bundle, sha)

    def test_chunk_slots_and_semantic_receipt_are_validated_beyond_hashes(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary); bundle, record, sha = setup(output)
            _, complete = run.extract_arrays(output, bundle, record, sha, values)
            path = output/'chunks/000.json'; receipt = io.read(path); receipt['padded_token_slots'] -= 1
            io.write(path, receipt); complete['digests']['chunks/000.json'] = io.sha(path)
            io.write(output/'neural-complete.json', complete)
            with self.assertRaisesRegex(ValueError, 'row count'):
                run.verify_complete(output, bundle, sha)
            receipt['padded_token_slots'] += 1; io.write(path, receipt)
            complete['digests']['chunks/000.json'] = io.sha(path)
            semantic = io.read(output/'neural-receipt.json'); semantic['reference_answers_in_inputs'] = True
            io.write(output/'neural-receipt.json', semantic)
            complete['digests']['neural-receipt.json'] = io.sha(output/'neural-receipt.json')
            io.write(output/'neural-complete.json', complete)
            with self.assertRaisesRegex(ValueError, 'semantic'):
                run.verify_complete(output, bundle, sha)

    def test_parent_exact_bytes_and_output_dtype_enforced(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary); bundle, record, sha = setup(output)
            def bad(entries):
                result, slots = values(entries); result['state'] = result['state'].astype(np.float64)
                return result, slots
            with self.assertRaisesRegex(ValueError, 'dtype'):
                run.extract_arrays(output, bundle, record, sha, bad)
            self.assertFalse((output/'neural-complete.json').exists())
            result, _ = run.extract_arrays(output, bundle, record, sha, values)
            result['state'][next(iter(bundle['reused_indexes'])), 0] += 1
            with self.assertRaisesRegex(ValueError, 'parent'):
                run.preserve_parent(bundle, result)

    def test_dependency_closure_ignores_unrelated_imported_tests(self):
        import test_expanded_review_metadata
        paths = io.sources(run)
        self.assertIn(str(Path(run.__file__).resolve()), paths)
        self.assertIn(str(Path(run.inputs.__file__).resolve()), paths)
        self.assertNotIn(str(Path(test_expanded_review_metadata.__file__).resolve()), paths)
        self.assertFalse(any('/.venv/' in p or '/test_' in p for p in paths))

    def test_memory_gate_stops_before_any_new_forward_or_attempt(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary); bundle, record, sha = setup(output)
            def low(*args, **kwargs): return SimpleNamespace(stdout='System-wide memory free percentage: 24%')
            with patch.object(run.subprocess, 'run', side_effect=low):
                with self.assertRaisesRegex(RuntimeError, 'Memory headroom'):
                    run.extract_arrays(output, bundle, record, sha, lambda _: self.fail('unexpected forward'),
                                       lambda: run.memory_guard(output, True))
            self.assertFalse((output/'attempts.jsonl').exists())
            self.assertFalse((output/'neural-complete.json').exists())
            self.assertTrue((output/'memory.jsonl').exists())


if __name__ == '__main__': unittest.main()
