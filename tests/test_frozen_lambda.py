"""Offline rejection checks for the separately frozen lambda follow-up."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frozen_lambda',ROOT/'experiments/pomdp-controller/frozen_lambda.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class FreezeRejectionTests(unittest.TestCase):
    def test_altered_lambda_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            f.p.io.save(run/'freeze.json',{'source_hashes':{},'planning_lambda':.3})
            f.p.io.save(run/'status.json',{})
            with mock.patch.object(f,'source_hashes',return_value={}):
                with self.assertRaisesRegex(ValueError,'Frozen inputs changed'):f.verify(run)

    def test_changed_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);model={'fake':'model'};dev=[{'id':'dev'}]
            f.p.io.save(run/'model.json',model);f.p.io.save(run/'dev-evidence.json',dev)
            f.p.io.save(run/'test-paths.json',[{'path':'retry','solver_tokens':10}])
            f.p.io.save(run/'freeze.json',{'source_hashes':{},'planning_lambda':.248,
                'model_sha256':f.p.io.sha(run/'model.json'),'dev_evidence_sha256':f.p.io.sha(run/'dev-evidence.json'),
                'selected_candidate':{},'frozen_utc':'2026-11-11T00:00:00+00:00'})
            f.p.io.save(run/'status.json',{'status':'complete','questions':400,'test_started_utc':'2026-11-11T00:00:01+00:00','finished_utc':'2026-11-11T00:00:02+00:00'})
            with mock.patch.object(f,'source_hashes',return_value={}),mock.patch.object(f.p,'check_run',return_value=model),mock.patch.object(f,'verify_selection',return_value=({},dev)),mock.patch.object(f.p.io,'load_dataset',return_value=[]),mock.patch.object(f.p,'replay',return_value=[{'path':'repair','solver_tokens':10}]):
                with self.assertRaisesRegex(ValueError,'Test path/cost/belief reconstruction failed'):f.verify(run)

if __name__=='__main__':unittest.main()
