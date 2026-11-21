from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file

import science_training_checkpoint as checkpoint
import train_science_listwise_resumable as trainer
import wait_science_listwise_resumable as waiter


class ResumableDriverTests(unittest.TestCase):
    def test_batch_order_exactly_matches_original_three_epoch_recipe(self):
        groups=[[4*q+i for i in range(4)] for q in range(1440)]
        for step in (0,1,49,719,720,721,1439,1440,2159,2160):
            epoch,batch=divmod(step,720)
            order=np.random.default_rng(trainer.original.SEED+epoch).permutation(1440)
            selected=[groups[i] for i in order[2*batch:2*batch+2]]
            value=trainer.next_position(step,720,groups)
            self.assertEqual(value,{'step':step,'epoch':epoch,'batch_in_epoch':batch,
                'next_group_indexes_sha256':hashlib.sha256(json.dumps(selected).encode()).hexdigest()})
        self.assertEqual(trainer.original.RECIPE['questions_per_batch'],2)
        self.assertEqual(trainer.original.RECIPE['epochs'],3)
        self.assertFalse(trainer.OPERATIONS['training_recipe_changed'])
        self.assertFalse(trainer.OPERATIONS['batch_shapes_changed'])

    def test_signal_handler_defers_first_interrupt_to_update_boundary(self):
        original=signal.getsignal(signal.SIGINT)
        with trainer.stop_at_boundary() as state:
            handler=signal.getsignal(signal.SIGINT);handler(signal.SIGINT,None)
            self.assertTrue(state['requested']);self.assertEqual(state['signal'],signal.SIGINT)
            with self.assertRaises(KeyboardInterrupt):handler(signal.SIGINT,None)
        self.assertIs(signal.getsignal(signal.SIGINT),original)

    def test_release_rejects_active_V4_or_low_memory_before_MPS_work(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);release=root/'release.json';output=root/'output'
            trainer.write(release,{'output':str(output.resolve()),'trainer_sha256':trainer.sha(trainer.__file__),
                                  'prerequisites':{},'protected_v4_pid':123456})
            args=SimpleNamespace(device='mps',release=release,output=output)
            with patch.object(trainer.os,'kill',return_value=None):
                with self.assertRaisesRegex(ValueError,'V4'):trainer.release_guard(args)
            with patch.object(trainer.os,'kill',side_effect=ProcessLookupError):
                with patch.object(trainer,'memory_snapshot',return_value={'system_free_percent':34}):
                    with self.assertRaisesRegex(ValueError,'35%'):trainer.release_guard(args)
                with patch.object(trainer,'memory_snapshot',return_value={'system_free_percent':50}):trainer.release_guard(args)

    def test_waiter_needs_all_archives_OOF_and_bound_refined_development(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);v4=root/'v4';dev=root/'dev';v4.mkdir();dev.mkdir()
            with patch.object(waiter,'alive',return_value=False):self.assertIsNone(waiter.prerequisites(v4,dev,1))
            trainer.write(v4/'contract.json',{'trainer_sha256':trainer.sha(waiter.refine_stream.__file__)})
            trainer.write(v4/'training-summary.json',{'fresh_data_used':False,'development_data_used':False,'question_grouped_folds':3})
            (v4/'oof.npz').write_bytes(b'synthetic OOF')
            for name in ('0','1','2','full'):
                folder=v4/name;folder.mkdir();(folder/'weights').write_bytes(b'synthetic weights')
                trainer.write(folder/'recipe.json',{'fold':int(name) if name!='full' else name,'weights':{'weights':trainer.sha(folder/'weights')}})
            trainer.write(dev/'identity.json',{'test_used':False,'verifier':str(v4.resolve()),
                'source_sha256':trainer.sha(waiter.development_refined_observer.__file__)})
            trainer.write(dev/'oof-provenance.json',{'oof_sha256':trainer.sha(v4/'oof.npz'),
                'training_contract_sha256':trainer.sha(v4/'contract.json'),'training_summary_sha256':trainer.sha(v4/'training-summary.json')})
            trainer.write(dev/'summary.json',{'test_used':False,'development_questions':256,
                'oof_provenance_sha256':trainer.sha(dev/'oof-provenance.json')})
            with patch.object(waiter,'alive',return_value=True):self.assertIsNone(waiter.prerequisites(v4,dev,1))
            with patch.object(waiter,'alive',return_value=False):
                result=waiter.prerequisites(v4,dev,1);self.assertIn(str((v4/'oof.npz').resolve()),result)
                (v4/'0/weights').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'weight'):waiter.prerequisites(v4,dev,1)

    def test_streamed_final_archive_matches_original_FP16_values_without_moving_model(self):
        class Config:
            def save_pretrained(self,directory):(directory/'config.json').write_text('{}')
        class Tokenizer:
            def save_pretrained(self,directory):(directory/'tokenizer.json').write_text('{}')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Linear(3,5);self.encoder.config=Config()
                self.register_buffer('numeric_mean',torch.arange(3,dtype=torch.float32))
                self.register_buffer('numeric_std',torch.ones(3))
                self.register_buffer('scalar',torch.tensor(.12345678))
                self.embedding=torch.nn.Embedding(12,3)
        with TemporaryDirectory() as temporary:
            model=Model();original={k:v.clone() for k,v in model.state_dict().items()}
            with patch.object(model,'cpu',side_effect=AssertionError('Live model must not move')):
                recipe=checkpoint.save_deployable(model,Tokenizer(),Path(temporary)/'full',{},chunk_bytes=24)
            self.assertLessEqual(recipe['streaming_archive']['maximum_tensor_cpu_copy_bytes'],24)
            parts={}
            for name,digest in recipe['weights'].items():
                self.assertEqual(checkpoint.sha(Path(temporary)/'full'/name),digest)
                parts.update(load_file(str(Path(temporary)/'full'/name)))
            for key,names in recipe['tensor_parts'].items():
                value=torch.cat([parts[name] for name in names],0) if len(names)>1 else parts[names[0]]
                expected=original[key] if key.startswith('numeric_') else original[key].half()
                self.assertTrue(torch.equal(value,expected));self.assertTrue(torch.equal(model.state_dict()[key],original[key]))

    def test_final_step_resume_archives_without_repeating_training(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'initial-checkpoint-copy').mkdir()
            trainer.write(root/'training-registration.json',{})
            np.savez(root/'training-encoded.npz',x=np.ones((1,1),dtype=np.int64))
            trainer.write(root/'training-encoded.json',{'sha256':trainer.sha(root/'training-encoded.npz'),'training_inputs_sha256':'synthetic'})
            recovery=root/'recovery';recovery.mkdir();(recovery/'LATEST.json').write_text('{}')
            (recovery/'manifest.json').write_text('{}');(root/'optimization-history.jsonl').touch()
            model=torch.nn.Linear(1,1);groups=[[i] for i in range(1440)]
            args=SimpleNamespace(output=root,initial=root,device='cpu',release=None)
            registration={'source_files':{},'training_inputs_sha256':'synthetic'}
            position=trainer.next_position(2160,720,groups)
            def restored(folder,model,optimizer,scheduler,*args):scheduler.last_epoch=2160;return position
            with ExitStack() as stack:
                stack.enter_context(patch.object(trainer.original.network,'load',return_value=(model,object(),{})))
                stack.enter_context(patch.object(trainer.original,'configure',return_value=([model.weight],[model.bias])))
                stack.enter_context(patch.object(trainer.checkpoint,'restore',side_effect=restored))
                stack.enter_context(patch.object(trainer.checkpoint,'load_manifest',return_value=(recovery,{'progress':position})))
                save=stack.enter_context(patch.object(trainer.checkpoint,'save_deployable'))
                final=stack.enter_context(patch.object(trainer,'final_calibration'))
                stack.enter_context(patch.object(model,'forward',side_effect=AssertionError('Do not repeat any completed update')))
                trainer.execute(args,{'initial_files':{}},[],groups,registration)
            save.assert_called_once();final.assert_called_once()
            self.assertEqual(final.call_args.args[-1]['step'],2160)


if __name__=='__main__':unittest.main()
