"""Synthetic route/composition tests; no datasets, model jobs or cloud calls."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import prospective_metadata_parallel as run
import test_metadata_paid_audit as fixtures
from test_metadata_review import make_adapter


def frozen(adapter):
    return {'selected':{'C':{'module':'metadata_review_hybrid','settings':{'penalty':.08},'protocol':'mixed-strong-root'}},
            'completed_A_B_registry':{'metadata_C_provenance':{'selected_adapter':{
                'state':adapter.state(),'receipt':adapter.receipt}}}}


class MetadataParallelTests(unittest.TestCase):
    def test_runtime_dispatch_is_explicit_and_plain_methods_delegate_unchanged(self):
        with TemporaryDirectory() as temporary:
            model,row,tree,reviews=fixtures.MetadataPaidAuditTest().fixture(Path(temporary))
            protocol=frozen(model.metadata_adapter)
            with patch.object(run.policy,'evaluate',return_value={'explicit':True}) as evaluate:
                fn=run.dispatch(protocol,{'C':model},'C')
                self.assertEqual(fn(row,tree,reviews),{'explicit':True})
                self.assertEqual(evaluate.call_args.args[-1],'C')
            protocol['selected']['C']['module']='hybrid_repairability'
            with self.assertRaisesRegex(ValueError,'explicit'):run.dispatch(protocol,{'C':model},'C')
            with patch.object(run,'ORIGINAL_DISPATCH',return_value='unchanged') as old:
                self.assertEqual(run.dispatch(protocol,{},'Stop'),'unchanged');old.assert_called_once()

    def test_same_attempt_timing_is_hashed_and_later_metadata_tampering_rejected(self):
        with TemporaryDirectory() as temporary:
            model,row,tree,reviews=fixtures.MetadataPaidAuditTest().fixture(Path(temporary));row['answer']=''
            protocol=frozen(model.metadata_adapter)
            fn=run.dispatch(protocol,{'C':model},'C')
            item,ledger=run.metadata_audit.replay_paid_inputs(fn,row,tree,reviews,'C',model)
            store=SimpleNamespace(view=lambda *args:(tree,reviews))
            initial={'id':row['id'],'subject':row['subject'],'method':'C','bank':'strong','item':item,'neural_receipts':ledger}
            with patch.object(run,'ORIGINAL_DECISION',return_value=initial):
                record=run.make_decision(protocol,{'C':model},'C',row,store,threading.Lock())
            with patch.object(run,'ORIGINAL_VERIFY_DECISION',return_value='final'):
                self.assertEqual(run.verify_decision(protocol,record,store,row),'final')
                changed=copy.deepcopy(record)
                changed['item']['metadata_observations'][0]['cpu_seconds']+=.1
                with self.assertRaises(ValueError):run.verify_decision(protocol,changed,store,row)
                changed=copy.deepcopy(record);changed['metadata_paid_audit']['metadata_calls']=0
                with self.assertRaises(ValueError):run.verify_decision(protocol,changed,store,row)

    def test_development_equality_exempts_only_nondeterministic_wall_time(self):
        expected={'accuracy':.8,'mean_compute':100.,'mean_metadata_calls':.5,'mean_metadata_cpu_ms':.01}
        actual={**expected,'mean_metadata_cpu_ms':4.}
        run.compare_development(actual,expected)
        for key in ('accuracy','mean_compute','mean_metadata_calls'):
            with self.assertRaises(ValueError):run.compare_development({**actual,key:actual[key]+.01},expected)
        with self.assertRaises(ValueError):run.compare_development({**actual,'mean_metadata_cpu_ms':float('nan')},expected)

    def test_protocol_requires_archived_new_sources_and_distinct_routes(self):
        with TemporaryDirectory() as temporary:
            study=Path(temporary);adapter=make_adapter();value=frozen(adapter)
            value['project_directory']=str(run.HERE.parents[3]);value['runtime_sources']={}
            for module in (run,run.frozen_metadata,run.metadata_audit,run.policy):
                path=Path(module.__file__).resolve();relative=str(path.relative_to(value['project_directory']))
                target=study/'runtime'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
                value['runtime_sources'][relative]=run.sha(path)
            with patch.object(run,'ORIGINAL_PAYLOAD',return_value={'source_sha256':{}}):
                payload=run.protocol_payload(value,study,'freeze')
                self.assertEqual(payload['schema'],run.SCHEMA)
                self.assertTrue(payload['grading_route'].startswith('prospective_metadata_parallel.py grade'))
                self.assertEqual(payload['metadata_adapter_sha256'],adapter.receipt['learned_state_sha256'])
                target.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'archived'):run.protocol_payload(value,study,'freeze')

    def test_whole_stage_hooks_restore_without_mutating_legacy_sources(self):
        original=(run.parallel.verify,run.parallel.make_decision,run.serial.dispatch,run.serial.replay_paid_inputs,run.grader.verify_decision)
        with run.installed():
            self.assertIs(run.serial.dispatch,run.dispatch)
            self.assertIs(run.serial.replay_paid_inputs,run.metadata_audit.replay_paid_inputs)
            self.assertIs(run.grader.verify_decision,run.verify_decision)
        self.assertEqual(original,(run.parallel.verify,run.parallel.make_decision,run.serial.dispatch,run.serial.replay_paid_inputs,run.grader.verify_decision))

    def test_grade_refuses_missing_metadata_seal_before_delegating_reference_reader(self):
        with TemporaryDirectory() as temporary:
            directory=Path(temporary);(directory/'collection').mkdir()
            run.ORIGINAL_ATOMIC_JSON(directory/'collection/complete.json',{})
            args=SimpleNamespace(study=directory,freeze_sha256='freeze',protocol_sha256='protocol')
            with patch.object(run.parallel,'verify_protocol'),patch.object(run,'verify_development'),patch.object(run.parallel,'grade') as old:
                with self.assertRaisesRegex(ValueError,'metadata-aware'):run.grade(args)
                old.assert_not_called()

    def test_complete_development_route_has_all2048_decisions_and_bound_receipts(self):
        with TemporaryDirectory() as temporary:
            directory=Path(temporary);data=directory/'data';data.mkdir()
            args=SimpleNamespace(study=directory,freeze_sha256='freeze',protocol_sha256='protocol',device='cpu')
            rows=[{'id':f'gsm8k-synthetic{i}','subject':'math','domain':'math','question':'Public?','answer':'1'} for i in range(256)]
            item={'path':'','compute':10.,'judge_physical_calls':0,'metadata_calls':0,
                  'metadata_cpu_seconds':0.,'review_neural_calls':0}
            expected={'n':256,'correct':256,'accuracy':1.,'mean_compute':10.,'mean_metadata_cpu_ms':99.}
            value={'data_directory':str(data),'development_baselines':{m:expected for m in run.serial.BASELINES},
                   'selected':{m:{'dev':expected} for m in ('A','B','C')}}
            def make(frozen,models,method,row,store,lock):
                self.assertEqual(row['answer'],'')
                return {'id':row['id'],'method':method,'item':dict(item)}
            def base_stats(values):
                return {'n':len(values),'correct':sum(v['correct'] for v in values),'accuracy':1.,'mean_compute':10.}
            with ExitStack() as stack:
                stack.enter_context(patch.object(run.parallel,'verify_protocol',return_value=value))
                stack.enter_context(patch.object(run.frozen_metadata,'load_models',return_value=(value,{})))
                stack.enter_context(patch.object(run.serial,'development_store',return_value=(rows,object())))
                stack.enter_context(patch.object(run,'make_decision',side_effect=make))
                stack.enter_context(patch.object(run,'verify_decision',return_value={'answer':'1'}))
                stack.enter_context(patch.object(run,'ORIGINAL_STATISTICS',side_effect=base_stats))
                stack.enter_context(patch('builtins.print'))
                run.audit_development(args)
            receipt=run.verify_development(args)
            self.assertEqual(receipt['method_question_audits'],2048)
            replay=json.loads((directory/'development-replay.json').read_bytes());self.assertEqual(len(replay),2048)
            self.assertTrue(all('correct' not in r['item'] for r in replay))
            (directory/'development-replay.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):run.verify_development(args)


if __name__=='__main__':unittest.main()
