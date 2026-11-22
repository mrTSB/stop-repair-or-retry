"""Regression for retained-summary identity mapping; inherited policy stays fixed."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import development_expanded_plain_a4_v2 as run


class ProvenanceTests(unittest.TestCase):
    def test_retained_outer_summary_and_resolved_identity_are_distinct_and_both_checked(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);outer=root/'retained';resolved=root/'actual';outer.mkdir();resolved.mkdir()
            for path,content in ((outer/'summary.json','outer-summary'),(outer/'identity.json','outer-identity'),
                (resolved/'summary.json','actual-summary'),(resolved/'identity.json','actual-identity'),(resolved/'model.pkl','actual-model')):
                path.write_text(content)
            ref={'summary':str(outer/'summary.json'),'summary_sha256':run.io.sha(outer/'summary.json'),
                'resolved':{'summary':str(resolved/'summary.json'),'summary_sha256':run.io.sha(resolved/'summary.json'),
                'identity_sha256':run.io.sha(resolved/'identity.json'),'model_path':str(resolved/'model.pkl'),
                'model_sha256':run.io.sha(resolved/'model.pkl')}}
            run.verify_a_references([ref])
            (outer/'summary.json').write_text('tampered-outer')
            with self.assertRaisesRegex(ValueError,'Registered evidence'):run.verify_a_references([ref])
            (outer/'summary.json').write_text('outer-summary');(resolved/'identity.json').write_text('tampered-inner')
            with self.assertRaisesRegex(ValueError,'Registered evidence'):run.verify_a_references([ref])

    def test_narrow_override_leaves_policy_and_cache_functions_identical_and_restores_globals(self):
        previous=(run.original.validate_registration,run.original.seal_payload,run.original.SCHEMA)
        functions=(run.original.replay_one,run.original.cache_only,run.original.aligned_cache,run.original.SETTINGS)
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with run.validation_override():
                self.assertIs(run.original.validate_registration,run.validate_registration)
                self.assertEqual(run.original.SCHEMA,run.SCHEMA)
                self.assertEqual(functions,(run.original.replay_one,run.original.cache_only,run.original.aligned_cache,run.original.SETTINGS))
                raise RuntimeError('synthetic')
        self.assertEqual(previous,(run.original.validate_registration,run.original.seal_payload,run.original.SCHEMA))

    def test_prior_failure_requires_no_policy_outputs(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            run.io.write(root/'registration.json',{'schema':run.original.SCHEMA,'source_files':{}})
            record=run.prior_attempt(root);self.assertEqual(record['policy_evaluations_before_failure'],0)
            (root/'search.json').write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError,'no policy outputs'):run.prior_attempt(root)


if __name__=='__main__':unittest.main()
