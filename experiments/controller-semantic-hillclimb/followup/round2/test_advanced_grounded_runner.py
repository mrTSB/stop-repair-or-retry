from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import advanced_grounded_runner as runner
from test_grounded_root_review import GroundedRemote
from test_strong_late_review import fixture


class AdvancedRunnerTests(unittest.TestCase):
    def test_only_identical_full_trajectories_are_reused_and_wrong_answers_retained(self):
        rows,trees,*_=fixture();original=runner.original
        with TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();new=root/'new';new.mkdir()
            with patch.object(runner.spec,'context',return_value='Original evidence'):
                with (old/'bank').open('a') as bank,(old/'raw').open('a') as raw,(old/'attempts').open('a') as attempts:
                    remote=GroundedRemote(True);collector=original.Collector(trees,{},bank,original.Journal(remote,raw,attempts))
                    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(collector.one,rows))
            raw=runner.protocol.strict_records(old/'raw');attempts=runner.protocol.strict_records(old/'attempts')
            same={r['id'] for r in rows[:32]}
            with patch.object(runner.spec,'context',side_effect=lambda q,*a:'Original evidence' if q in same else 'New lecture evidence'):
                logical,reused,reused_attempts,ids=runner.compatible(rows,trees,raw,attempts)
                self.assertEqual(set(ids),same);self.assertEqual(len(reused),128);self.assertEqual(len(logical),96)
                self.assertTrue(all(not v['response']['answer'] for v in logical if v['kind']=='solver'))
                for name,values in (('bank',logical),('raw',reused),('attempts',reused_attempts)):
                    with (new/name).open('a') as stream:
                        for value in values:original.paired.durable_line(stream,value)
                known,_=runner.spec.restore(rows,trees,logical,reused)
                with (new/'bank').open('a') as bank,(new/'raw').open('a') as out,(new/'attempts').open('a') as attempt_stream:
                    remote=GroundedRemote(True)
                    collector=original.Collector(trees,known,bank,original.Journal(remote,out,attempt_stream,reused,reused_attempts))
                    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(collector.one,rows))
                self.assertEqual(len(remote.calls),128)
                _,coverage=runner.spec.restore(rows,trees,runner.protocol.strict_records(new/'bank'),runner.protocol.strict_records(new/'raw'),True)
                self.assertEqual(coverage['physical_calls'],256);self.assertEqual(coverage['logical_reviews'],64)

    def test_configuration_restores_registered_protocol_and_runner(self):
        before=(runner.spec.CORPUS,runner.spec.EXTRA_BANKS,runner.original.registration,runner.original.require_priority)
        with self.assertRaises(RuntimeError):
            with runner.configured_runner():
                self.assertEqual(runner.spec.CORPUS,runner.CORPUS)
                self.assertIn(runner.PRIOR,runner.spec.EXTRA_BANKS)
                raise RuntimeError('synthetic')
        self.assertEqual(before,(runner.spec.CORPUS,runner.spec.EXTRA_BANKS,runner.original.registration,runner.original.require_priority))

    def test_reuse_seed_is_idempotent_and_detects_modified_prefix(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);record={'reuse_identity':'fixed'};runner.io.write(root/'registration.json',record)
            args=SimpleNamespace(output=root,registration_sha256=runner.io.sha(root/'registration.json'))
            with patch.object(runner,'registration',return_value=record),patch.object(runner,'reused_inputs',return_value=([{'node':1}],[{'raw':1}],[{'attempt':1}],['q'])):
                runner.seed_reused(args);before=(root/'raw-generations.jsonl').read_bytes()
                with (root/'raw-generations.jsonl').open('ab') as stream:stream.write(b'{"later":1}\n')
                runner.seed_reused(args);self.assertTrue((root/'raw-generations.jsonl').read_bytes().startswith(before))
                (root/'raw-generations.jsonl').write_bytes(b'changed\n')
                with self.assertRaises(ValueError):runner.seed_reused(args)


if __name__=='__main__':unittest.main()
