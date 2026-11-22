from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import development_completion_b as runner


class CompletionBDevelopmentTests(unittest.TestCase):
    def test_partial_or_changed_jsonl_is_preserved_not_overwritten(self):
        with TemporaryDirectory() as temporary:
            path=Path(temporary)/'private-candidate-00.jsonl';rows=[{'id':'q','correct':0}]
            runner.persist_jsonl(path,rows);complete=path.read_bytes()
            runner.persist_jsonl(path,rows)
            for corrupted in (complete[:-1],complete.replace(b'0',b'1')):
                path.write_bytes(corrupted)
                with self.assertRaisesRegex(ValueError,'partial earlier JSONL'):
                    runner.persist_jsonl(path,rows)
                self.assertEqual(path.read_bytes(),corrupted)

    def test_full256_by24_fixed_replay_retains_math_baselines_and_seals_resume(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);reference_path=root/'old-B.json'
            rows=[{'id':f'q{i}','question':'visible','domain':'math' if i<128 else 'science',
                   'subject':'math' if i<128 else ('chemistry' if i<192 else 'physics'),'answer':'A'} for i in range(256)]
            paths=['','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry',
                   'guided','guided/reuse','guided/retry','retry/reuse','complete','repair/complete','retry/complete']
            trees={r['id']:{p:{'answer':'A' if p.endswith('complete') else 'B','text':'paid',
                    'input_tokens':10,'output_tokens':1} for p in paths} for r in rows}
            judges={(r['id'],''):{'answer':'A','text':'paid review','action':'repair','p_correct':.5,
                'input_tokens':30,'output_tokens':2,'physical_api_calls':2,
                'calls':[{'input_tokens':10,'output_tokens':1},{'input_tokens':20,'output_tokens':1}]} for r in rows}
            old_settings={'root_stop':'action','second_stop':'never','second_action':'reuse'}
            old_items=[runner.policy.original.policy(r,trees[r['id']],judges,old_settings) for r in rows]
            old={'family':'B','protocol':'matched-retained-1024-512-128','settings':old_settings,
                 'dev':runner.policy.original.metrics(old_items)}
            baselines={n:runner.policy.original.statistics([runner.policy.original.fixed(r,trees[r['id']],*s) for r in rows])
                       for n,s in runner.policy.original.BASELINES.items()}
            reference={'B':old,'matched_B':old,'baselines':baselines,'development_questions':256,'test_used':False}
            runner.io.write(reference_path,reference)
            args=SimpleNamespace(data=root/'data',retained=root/'retained',complete=root/'complete',
                b_reference=reference_path,output=root/'output',mode='register')
            receipt={'dev_questions_sha256':'panel'}
            bundle=(rows,trees,judges,reference,receipt,{})
            with patch.object(runner,'inputs',return_value=bundle):
                runner.run(args)
                args.mode='run';args.registration_sha256=runner.io.sha(args.output/'registration.json')
                runner.run(args)
                summary=runner.io.read(args.output/'summary.json')
                self.assertEqual(summary['B']['dev']['correct'],128)
                self.assertEqual(summary['baselines'],baselines)
                self.assertEqual(summary['new_cloud_calls'],0);self.assertEqual(summary['new_neural_calls'],0)
                self.assertEqual(len(runner.io.read(args.output/'search.json')),24)
                before=runner.io.sha(args.output/'complete.json')
                with patch.object(runner.policy,'evaluate_panel',side_effect=AssertionError('sealed replay repeated')):
                    runner.run(args)
                self.assertEqual(runner.io.sha(args.output/'complete.json'),before)
                for name in ('private-candidate-00.jsonl','private-candidate-23.jsonl'):
                    records=[runner.io.json.loads(line) for line in (args.output/name).read_text().splitlines()]
                    self.assertEqual(records[:128],old_items[:128])


if __name__=='__main__':unittest.main()
