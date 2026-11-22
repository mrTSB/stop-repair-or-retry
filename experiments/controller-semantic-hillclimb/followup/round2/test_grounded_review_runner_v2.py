from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import grounded_review_runner_v2 as runner


class RunnerTests(unittest.TestCase):
    def test_nested_metadata_is_not_treated_as_a_path_and_hashes_still_enforced(self):
        io=runner.original.io
        with TemporaryDirectory() as tmp:
            root=Path(tmp);evidence=root/'evidence';evidence.write_text('pinned')
            capacity=root/'capacity.jsonl';capacity.write_text('{"min_containers":0,"max_containers":2}\n')
            io.write(root/'registration.json',{'schema':runner.original.paired.SCHEMA,'source_files':{},
                'input_evidence':{'files':{str(evidence):io.sha(evidence)},'auxiliary_encoder_features_allowed':True}})
            io.write(root/'complete.json',{'schema':runner.original.paired.SCHEMA,
                'registration_sha256':io.sha(root/'registration.json'),
                'coverage':{'physical_calls':5120,'logical_actions':5120,'physical_attempts':5120},
                'test_used':False,'scored':False,'digests':{'capacity.jsonl':io.sha(capacity)}})
            runner.require_priority(SimpleNamespace(priority=root))
            evidence.write_text('changed')
            with self.assertRaises(ValueError):runner.require_priority(SimpleNamespace(priority=root))

    def test_adapter_restores_original_functions_even_after_failure(self):
        old=(runner.original.registration,runner.original.require_priority)
        with self.assertRaises(RuntimeError):
            with runner.adapted_runner():
                self.assertIs(runner.original.registration,runner.registration)
                self.assertIs(runner.original.require_priority,runner.require_priority)
                raise RuntimeError('synthetic')
        self.assertEqual(old,(runner.original.registration,runner.original.require_priority))

    def test_new_registration_binds_adapter_and_preserves_no_inference_predecessor(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);failed=root/'grounded1024-root-review-pilot-v1';failed.mkdir()
            runner.original.io.write(failed/'registration.json',{'source_files':{},'input_evidence':{}})
            args=SimpleNamespace(data=root/'data')
            with patch.object(runner,'original_registration',return_value={'input_evidence':{}}):
                record=runner.registration(args)
                self.assertEqual(record['runner_schema'],runner.SCHEMA)
                self.assertIn(str(Path(runner.__file__).resolve()),record['source_files'])
                self.assertIn(str((failed/'registration.json').resolve()),record['input_evidence'])
                (failed/'attempts.jsonl').write_text('attempt')
                with self.assertRaises(ValueError):runner.registration(args)


if __name__=='__main__':unittest.main()
