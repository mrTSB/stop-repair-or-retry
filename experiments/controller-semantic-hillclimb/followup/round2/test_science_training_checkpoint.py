import copy
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import science_training_checkpoint as checkpoint


class Small(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(5,9),torch.nn.Dropout(.3),torch.nn.Linear(9,2))
    def forward(self,x):return self.layers(x)


def setup(seed):
    torch.set_num_threads(1);torch.random.default_generator.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=Small();optimizer=torch.optim.AdamW(model.parameters(),lr=.01,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0.,(11-step)/11))
    return model,optimizer,scheduler


def step(model,optimizer,scheduler,history,number):
    optimizer.zero_grad(set_to_none=True)
    x=torch.randn(3,5)+np.random.random()+random.random()
    y=torch.tensor([0,1,0]);loss=torch.nn.functional.cross_entropy(model(x),y)
    loss.backward();optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    with history.open('a') as stream:stream.write(json.dumps({'step':number,'loss':float(loss.detach())})+'\n')
    return float(loss.detach())


class CheckpointTests(unittest.TestCase):
    def test_exact_interrupted_resume_parity_all_rng_and_optimizer(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary);full_history=folder/'full.jsonl';full_history.touch()
            full,full_optimizer,full_scheduler=setup(31)
            expected=[step(full,full_optimizer,full_scheduler,full_history,n) for n in range(1,12)]
            history=folder/'history.jsonl';history.touch();model,optimizer,scheduler=setup(31)
            actual=[step(model,optimizer,scheduler,history,n) for n in range(1,5)]
            receipt=checkpoint.save(folder/'checkpoints',model,optimizer,scheduler,
                {'step':4,'epoch':0,'batch_in_epoch':4},{'recipe':'fixed'},'cpu',history,chunk_bytes=32)
            self.assertLessEqual(receipt['maximum_tensor_cpu_copy_bytes'],32)
            # Two completed but uncheckpointed updates are preserved as an
            # abandoned attempt, then replayed from the exact recovery state.
            for n in (5,6):step(model,optimizer,scheduler,history,n)
            resumed,resumed_optimizer,resumed_scheduler=setup(999)
            with patch.object(torch.mps,'get_rng_state',side_effect=AssertionError('CPU test must not touch MPS')):
                progress=checkpoint.restore(folder/'checkpoints',resumed,resumed_optimizer,resumed_scheduler,
                    {'recipe':'fixed'},'cpu',history)
            self.assertEqual(progress,{'step':4,'epoch':0,'batch_in_epoch':4})
            actual += [step(resumed,resumed_optimizer,resumed_scheduler,history,n) for n in range(5,12)]
            self.assertEqual(actual,expected)
            for key,value in full.state_dict().items():self.assertTrue(torch.equal(value,resumed.state_dict()[key]))
            self.assertEqual(full_scheduler.state_dict(),resumed_scheduler.state_dict())
            a,b=full_optimizer.state_dict(),resumed_optimizer.state_dict();self.assertEqual(a['param_groups'],b['param_groups'])
            for key,values in a['state'].items():
                for name,value in values.items():self.assertTrue(torch.equal(value,b['state'][key][name]))
            self.assertEqual(history.read_bytes(),full_history.read_bytes())
            self.assertEqual(len(list((folder/'checkpoints/abandoned-attempt-history').glob('*.jsonl'))),1)

    def test_manifest_tensor_and_binding_tampering_fail_before_restore(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);history=root/'history';history.touch();model,optimizer,scheduler=setup(1)
            checkpoint.save(root/'states',model,optimizer,scheduler,{'step':0},{'recipe':'a'},'cpu',history)
            with self.assertRaisesRegex(ValueError,'recipe'):
                checkpoint.load_manifest(root/'states',{'recipe':'b'},'cpu')
            directory,manifest=checkpoint.load_manifest(root/'states',{'recipe':'a'},'cpu')
            shard=directory/next(iter(manifest['files']));shard.write_bytes(b'tampered')
            before=copy.deepcopy(model.state_dict())
            with self.assertRaisesRegex(ValueError,'shard'):
                checkpoint.restore(root/'states',model,optimizer,scheduler,{'recipe':'a'},'cpu',history)
            for key,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))

    def test_partial_checkpoint_never_replaces_latest_and_keeps_two_complete(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);history=root/'history';history.touch();model,optimizer,scheduler=setup(2)
            checkpoint.save(root/'states',model,optimizer,scheduler,{'step':0},{},'cpu',history)
            old=(root/'states/LATEST.json').read_bytes()
            with patch.object(checkpoint.Writer,'tensor',side_effect=RuntimeError('simulated interruption')):
                with self.assertRaises(RuntimeError):checkpoint.save(root/'states',model,optimizer,scheduler,{'step':0},{},'cpu',history)
            self.assertEqual(old,(root/'states/LATEST.json').read_bytes())
            for n in (1,2,3):
                step(model,optimizer,scheduler,history,n)
                checkpoint.save(root/'states',model,optimizer,scheduler,{'step':n},{},'cpu',history)
            complete=[p for p in (root/'states').glob('step-*') if not p.name.endswith('.partial')]
            self.assertEqual(len(complete),2)
            self.assertEqual(len(list((root/'states/retired-manifests').glob('*.json'))),2)
            self.assertTrue(any((root/'states').glob('*.partial')))

    def test_mps_rng_api_is_saved_and_restored_without_actual_MPS_calls(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);history=root/'history';history.touch();model,optimizer,scheduler=setup(7)
            expected=torch.tensor([1,2,3],dtype=torch.uint8)
            with patch.object(torch.mps,'synchronize'),patch.object(torch.mps,'get_rng_state',return_value=expected) as get:
                checkpoint.save(root/'states',model,optimizer,scheduler,{'step':0},{},'mps',history)
            with patch.object(torch.mps,'set_rng_state') as setter:
                checkpoint.restore(root/'states',model,optimizer,scheduler,{},'mps',history)
            get.assert_called_once();self.assertTrue(torch.equal(setter.call_args.args[0],expected))

    def test_parameter_order_and_half_precision_training_state_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);history=root/'history';history.touch();model,optimizer,scheduler=setup(2)
            checkpoint.save(root/'states',model,optimizer,scheduler,{'step':0},{},'cpu',history)
            optimizer.param_groups[0]['params'].reverse()
            with self.assertRaisesRegex(ValueError,'order'):
                checkpoint.restore(root/'states',model,optimizer,scheduler,{},'cpu',history)
            model.half()
            with self.assertRaisesRegex(ValueError,'FP32'):
                checkpoint.save(root/'other',model,optimizer,scheduler,{'step':0},{},'cpu',history)


if __name__=='__main__':unittest.main()
