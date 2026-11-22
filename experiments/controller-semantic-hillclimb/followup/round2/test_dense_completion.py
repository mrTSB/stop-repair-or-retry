import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import dense_completion_probe as probe
import extract_dense_completion as driver


def arrays(n):
    return {**{name:np.zeros((n,width),np.float32) for name,width in
        (('state',4),('plain',8),('guided',8),('joint256',256))},
        'token_slots':np.full(n,12,np.int64)}


def bundle(n=64):
    return {'entries':[{'left':f'question{i}','right':'paid answer','numeric':[0.,0.]} for i in range(n)],
        'metadata':[{'id':f'q{i//3}','path':('','repair','retry')[i%3],
            'subject':'chemistry' if i<384 else 'physics','prompt_sha256':str(i),'input_sha256':str(i)} for i in range(n)],
        'old_raw20':np.zeros((n,20),np.float32)}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(hidden_size=4);self.calls=0
        self.embedding=nn.Embedding(8,4)
    def forward(self,input_ids,attention_mask,token_type_ids=None):
        self.calls+=1;return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def model_and_tensors(n=4):
    model=driver.network.Network(Encoder(),2)
    tensors={'input_ids':torch.ones((n,16),dtype=torch.long),
        'attention_mask':torch.tensor([[1]*12+[0]*4]*n), 'numeric':torch.zeros((n,2))}
    return model,tensors


class DenseProbeTests(unittest.TestCase):
    def test_hook_is_one_shared_forward_eval_and_detached(self):
        torch.manual_seed(238);model,tensors=model_and_tensors();model.train()
        result=probe.capture(model,tensors)
        self.assertEqual(model.encoder.calls,1);self.assertFalse(model.training)
        self.assertFalse(model.project._forward_hooks);self.assertTrue(all(p.grad is None for p in model.parameters()))
        with torch.inference_mode():
            for name,head in (('state',model.state),('plain',model.plain),('guided',model.guided)):
                np.testing.assert_array_equal(result[name],head(torch.tensor(result['joint256'])).numpy())
        second=probe.capture(model,tensors)
        for name in result:np.testing.assert_array_equal(result[name],second[name])
        np.testing.assert_array_equal(result['token_slots'],12)
        with patch.object(model,'forward',side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):probe.capture(model,tensors)
        self.assertFalse(model.project._forward_hooks)

    def test_full_logit_parity_bound_and_no_row_filtering(self):
        values=arrays(3);values['state'][0,0]=.0049
        passed,difference=probe.parity(values,np.zeros((3,20)))
        self.assertTrue(passed['all_logits_within_tolerance']);self.assertEqual(difference.shape,(3,20))
        values['guided'][2,7]=.006
        failed,difference=probe.parity(values,np.zeros((3,20)))
        self.assertFalse(failed['all_logits_within_tolerance']);self.assertEqual(failed['failed_logits'],1)
        self.assertEqual(failed['failed_rows'],1);self.assertEqual(failed['logits'],60)
        self.assertEqual(failed['difference_sha256'],probe.fixed.array_sha(difference))
        bad=arrays(3);bad['joint256'][0,0]=np.nan
        with self.assertRaises(ValueError):probe.validate(bad,3)

    def test_paid_features_and_joint_alignment_no_reference_or_future_inputs(self):
        rows=[{'id':f'q{i}','subject':'chemistry' if i<128 else 'physics','answer':'A'} for i in range(256)]
        paths=('','repair','retry','repair/repair','repair/retry','retry/repair','retry/retry',
            'complete','repair/complete','retry/complete')
        trees={r['id']:{p:{'answer':'A' if p.endswith('complete') else 'B','text':p,'prompt_sha256':p} for p in paths} for r in rows}
        metadata=[{'id':r['id'],'path':p,'prompt_sha256':p} for r in rows for p in ('','repair','retry')]
        values=arrays(768);values['joint256'][:,0]=np.arange(768)
        def observation(row,path,paid,neural):
            self.assertEqual(row['answer'],'');self.assertEqual(set(paid),{'',path})
            return np.concatenate([*neural[:3],np.zeros(110)])
        with patch.object(probe.fixed.core.plain,'observables',side_effect=observation):
            data=probe.table(rows,trees,metadata,values)
            changed=copy.deepcopy(trees);changed['q0']['repair/complete']['answer']='B'
            other=probe.table(rows,changed,metadata,values)
            for name in probe.VARIANTS:np.testing.assert_array_equal(data['features'][name],other['features'][name])
            self.assertNotEqual(data['labels'][1],other['labels'][1])
            malformed=copy.deepcopy(metadata);malformed[0]['prompt_sha256']='different'
            with self.assertRaisesRegex(ValueError,'prompt'):probe.table(rows,trees,malformed,values)
            with self.assertRaises(ValueError):probe.table(rows,trees,metadata[:-1],arrays(767))
        control,dense=(data['features'][name] for name in probe.VARIANTS)
        self.assertEqual(control.shape,(768,130));self.assertEqual(dense.shape,(768,386))
        np.testing.assert_array_equal(control,dense[:,:130]);np.testing.assert_array_equal(dense[:,130:],values['joint256'])

    def test_whole_question_and_fold_local_scaler_training(self):
        rng=np.random.default_rng(238)
        records=[{'id':f'q{i}','subject':'chemistry' if i<128 else 'physics','path':p,
            'fold':probe.fixed.refined.fold_for(f'q{i}')} for i in range(256) for p in ('','repair','retry')]
        x=rng.normal(size=(768,12));y=rng.integers(0,16,768)
        data={'records':records,'features':{'dense':x},'labels':y,'valid':np.ones(768,bool)}
        with threadpool_limits(limits=1):
            held,p,model,receipt=probe.fixed.outer_prediction(data,'dense',0)
            changed=copy.deepcopy(data);changed['labels'][held]=(y[held]+1)%16
            changed['features']['dense'][held]+=100
            held2,_,model2,_=probe.fixed.outer_prediction(changed,'dense',0)
        self.assertEqual(receipt['held_questions']*3,int(held.sum()))
        self.assertEqual(receipt['training_questions']+receipt['held_questions'],256)
        np.testing.assert_array_equal(held,held2)
        np.testing.assert_allclose(model[0].mean_,x[~held].mean(0))
        for key in ('mean_','scale_'):np.testing.assert_array_equal(getattr(model[0],key),getattr(model2[0],key))
        np.testing.assert_array_equal(model[1].coef_,model2[1].coef_)
        self.assertTrue(np.isfinite(p).all())

    def test_question_bootstrap_fixed_screen_and_scope(self):
        data={'records':[{'id':f'q{i}','subject':'chemistry' if i<128 else 'physics'} for i in range(256) for _ in range(3)],
            'labels':np.arange(768)%16}
        uniform=np.full((768,16),1/16);perfect=np.eye(16)[data['labels']]
        report=probe.diagnostics(data,dict(zip(probe.VARIANTS,(uniform,perfect))))
        self.assertTrue(report['screen_passed']);self.assertFalse(report['fresh_validation'])
        self.assertFalse(report['controller_accuracy_claim']);self.assertFalse(report['neural_refitting_or_nested_CV_claim'])
        for metric in report['comparisons']:self.assertLess(report['comparisons'][metric]['paired_question_CI95'][1],0)
        self.assertFalse(probe.diagnostics(data,dict.fromkeys(probe.VARIANTS,uniform))['screen_passed'])

    def test_protected_records_and_model_loading_guards(self):
        with TemporaryDirectory() as temporary,probe.fixed.training_only_reads():
            for name in ('dev-questions.jsonl','test-questions.jsonl'):
                with self.assertRaisesRegex(AssertionError,'DEV/TEST'):Path(temporary,name).read_text()
        with probe.fixed.guarded.no_model_calls():
            with self.assertRaises(AssertionError):probe.fixed.guarded.perception.Observer.load(None)


class DenseDurabilityTests(unittest.TestCase):
    def test_chunks_resume_without_any_forward_and_exact_costs(self):
        inputs=bundle();calls=[]
        with TemporaryDirectory() as temporary:
            output=Path(temporary)
            values,seals=driver.extract_arrays(output,inputs,'registration',lambda entries:(calls.append(len(entries)) or arrays(len(entries))),lambda:calls.append('guard'))
            self.assertEqual(calls,['guard',32,'guard',32]);self.assertEqual(len(seals),4)
            def forbidden(*args):self.fail('completed chunk was forwarded again')
            resumed,_=driver.extract_arrays(output,inputs,'registration',forbidden,forbidden)
            for name in values:np.testing.assert_array_equal(values[name],resumed[name])
            receipt=driver.io.read(output/'chunks/000.json');self.assertEqual(receipt['physical_padded_token_slots'],32*512)
            self.assertEqual(receipt['actual_unpadded_token_slots'],32*12)
            receipt['extra_encoder_passes_for_project256']=1;driver.io.write(output/'chunks/000.json',receipt)
            with self.assertRaisesRegex(ValueError,'semantics'):driver.extract_arrays(output,inputs,'registration',forbidden,forbidden)

    def test_partial_temporary_or_orphan_chunks_fail_before_inference(self):
        inputs=bundle()
        for filename in ('chunks/000.npz.tmp','chunks/000.npz','chunks/999.npz'):
            with self.subTest(filename=filename),TemporaryDirectory() as temporary:
                output=Path(temporary);(output/'chunks').mkdir();(output/filename).write_bytes(b'preserve')
                with self.assertRaises(ValueError):driver.extract_arrays(output,inputs,'registration',lambda x:self.fail('forward'),lambda:self.fail('guard'))
        with TemporaryDirectory() as temporary:
            output=Path(temporary);driver.io.append(output/'attempts.jsonl',{'identity':driver.chunk_identity(inputs,0,'registration'),'attempted_rows':32})
            with self.assertRaisesRegex(ValueError,'interrupted'):driver.extract_arrays(output,inputs,'registration',lambda x:self.fail('forward'),lambda:self.fail('guard'))
        with self.assertRaises(ValueError):driver.chunk_identity(inputs,-32,'registration')

    def test_guard_failure_has_no_attempt_and_wrong_parity_is_preserved(self):
        inputs=bundle(32)
        with TemporaryDirectory() as temporary:
            output=Path(temporary)
            with self.assertRaisesRegex(RuntimeError,'memory'):
                driver.extract_arrays(output,inputs,'registration',lambda x:self.fail('forward'),lambda:(_ for _ in ()).throw(RuntimeError('memory')))
            self.assertFalse((output/'attempts.jsonl').exists())
            bad=arrays(32);bad['state'][0,0]=1.
            with self.assertRaisesRegex(ValueError,'parity failed'):
                driver.extract_arrays(output,inputs,'registration',lambda x:bad,lambda:None)
            self.assertTrue((output/'chunks/000.npz').exists());self.assertTrue((output/'chunks/000.json').exists())
            with self.assertRaisesRegex(ValueError,'parity failed'):
                driver.extract_arrays(output,inputs,'registration',lambda x:self.fail('forward'),lambda:self.fail('guard'))

    def test_complete_grid_seals_recombine_no_work_and_reject_cost_tampering(self):
        inputs=bundle(768);model,_=model_and_tensors()
        with torch.no_grad():
            for parameter in model.parameters():parameter.zero_()
        inputs['recipe']={'parameters':sum(p.numel() for p in model.parameters())}
        with TemporaryDirectory() as temporary:
            output=Path(temporary)
            record={'schema':driver.SCHEMA,'output':str(output.resolve()),'source_files':{},'input_evidence':{},
                'encoder_directory':'unused','parameters':inputs['recipe']['parameters'],'device':'cpu','encoder_max_length':512,
                'batch_size':driver.BATCH,'chunk_size':driver.CHUNK,'metadata_sha256':probe.fixed.fingerprint(inputs['metadata']),
                'input_sha256':probe.fixed.fingerprint(inputs['entries']),'old_raw20_sha256':probe.fixed.array_sha(inputs['old_raw20'])}
            driver.io.write(output/'registration.json',record)
            args=SimpleNamespace(output=output,registration_sha256=driver.io.sha(output/'registration.json'))
            def encode(tokenizer,entries,length):
                self.assertEqual(length,512)
                n=len(entries);return {'input_ids':torch.ones((n,512),dtype=torch.long),
                    'attention_mask':torch.tensor([[1]*12+[0]*500]*n),'numeric':torch.zeros((n,2))}
            def guard(path,loaded):driver.io.append(path/'memory.jsonl',{'loaded':loaded})
            with (patch.object(driver.network,'load',return_value=(model,None,inputs['recipe'])) as load,
                patch.object(driver.network,'encode',side_effect=encode),patch.object(driver.resource,'memory_guard',side_effect=guard)):
                driver.extract(args,inputs,record)
                self.assertEqual(load.call_count,1);self.assertEqual(model.encoder.calls,192)
            with patch.object(driver.network,'load',side_effect=AssertionError('no reload')):
                driver.extract(args,inputs,record)
            _,complete=driver.verify_complete(output,args.registration_sha256,inputs)
            self.assertEqual(complete['physical_padded_token_slots'],768*512)
            complete['extra_project_encoder_passes']=1;driver.io.write(output/'neural-complete.json',complete)
            with self.assertRaisesRegex(ValueError,'cost/scope'):driver.verify_complete(output,args.registration_sha256,inputs)


if __name__=='__main__':unittest.main()
