"""CPU-only tests for the bounded MPS operational wrapper; no MPS invocation."""
from pathlib import Path
import random
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import resume_science_cache_cleanup as run


class CleanupTests(unittest.TestCase):
    def test_cleanup_precedes_measurement_and_preserves_rng(self):
        events = []
        random.seed(4); np.random.seed(4); torch.manual_seed(4)
        before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        def measure(device):
            events.append('measure:'+device)
            return {'system_free_percent': 40, 'mps_driver_bytes': 100}
        with TemporaryDirectory() as directory, \
                patch.object(run.trainer, 'memory_snapshot', side_effect=measure), \
                patch.object(run.trainer.torch.mps, 'current_allocated_memory', return_value=10), \
                patch.object(run.trainer.torch.mps, 'driver_allocated_memory', return_value=20), \
                patch.object(run.trainer.torch.mps, 'synchronize', side_effect=lambda: events.append('sync')), \
                patch.object(run.trainer.torch.mps, 'empty_cache', side_effect=lambda: events.append('empty')), \
                patch.object(run.gc, 'collect', side_effect=lambda: events.append('gc')):
            original = run.trainer.memory_snapshot
            with run.operational_hooks(100, Path(directory)/'telemetry.jsonl'):
                self.assertEqual(run.trainer.memory_snapshot('mps')['system_free_percent'], 40)
                self.assertEqual(events, ['gc', 'sync', 'empty', 'sync', 'measure:mps'])
                events.clear(); run.trainer.memory_snapshot('cpu')
                self.assertEqual(events, ['measure:cpu'])
            self.assertIs(run.trainer.memory_snapshot, original)
        self.assertEqual(before[0], random.getstate())
        self.assertEqual(before[1][0], np.random.get_state()[0]); np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        self.assertEqual(before[1][2:], np.random.get_state()[2:]); torch.testing.assert_close(before[2], torch.get_rng_state(), rtol=0, atol=0)

    def test_stop_signal_occurs_only_after_atomic_target_checkpoint(self):
        events = []
        def saved(*args, **kwargs):
            events.append(('atomic-save', args[4]['step'])); return {'step': args[4]['step']}
        with patch.object(run.trainer.checkpoint, 'save', side_effect=saved), \
                patch.object(run.signal, 'getsignal', return_value=lambda *a: None), \
                patch.object(run.os, 'kill', side_effect=lambda pid, sig: events.append(('signal', sig))):
            original = run.trainer.checkpoint.save
            with run.operational_hooks(100, Path('unused')):
                self.assertEqual(run.trainer.checkpoint.save(None, None, None, None, {'step': 20}), {'step': 20})
                self.assertEqual(events, [('atomic-save', 20)])
                events.clear(); run.trainer.checkpoint.save(None, None, None, None, {'step': 100})
                self.assertEqual(events, [('atomic-save', 100), ('signal', signal.SIGTERM)])
            self.assertIs(run.trainer.checkpoint.save, original)

    def test_checkpoint_failure_or_uninstalled_handler_cannot_signal(self):
        with patch.object(run.trainer.checkpoint, 'save', side_effect=RuntimeError('Unsealed checkpoint')), \
                patch.object(run.os, 'kill') as kill:
            with run.operational_hooks(100, Path('unused')):
                with self.assertRaises(RuntimeError): run.trainer.checkpoint.save(None, None, None, None, {'step': 100})
            kill.assert_not_called()
        with patch.object(run.trainer.checkpoint, 'save', return_value={}), \
                patch.object(run.signal, 'getsignal', return_value=signal.SIG_DFL), patch.object(run.os, 'kill') as kill:
            with run.operational_hooks(100, Path('unused')):
                with self.assertRaises(ValueError): run.trainer.checkpoint.save(None, None, None, None, {'step': 100})
            kill.assert_not_called()

    def test_cpu_optimizer_trajectory_identical_with_operational_hook(self):
        def train(with_hook, target):
            torch.manual_seed(12)
            model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
            opt = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
            losses = []
            def steps():
                for step in range(20):
                    x = torch.randn(2, 3); y = torch.randn(2, 1)
                    loss = (model(x)-y).square().mean(); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                    losses.append(loss.item()); del loss, x, y
                    if with_hook and (step+1)%10 == 0: run.trainer.memory_snapshot('mps')
            if with_hook:
                with run.operational_hooks(100, target): steps()
            else: steps()
            return losses, model.state_dict(), opt.state_dict(), torch.get_rng_state()
        with TemporaryDirectory() as directory, patch.object(run.trainer, 'memory_snapshot', return_value={'system_free_percent': 90}), \
                patch.object(run.trainer.torch.mps, 'current_allocated_memory', return_value=0), \
                patch.object(run.trainer.torch.mps, 'driver_allocated_memory', return_value=0), \
                patch.object(run.trainer.torch.mps, 'synchronize'), patch.object(run.trainer.torch.mps, 'empty_cache'):
            left = train(False, None); right = train(True, Path(directory)/'telemetry.jsonl')
        self.assertEqual(left[0], right[0])
        for name in left[1]: torch.testing.assert_close(left[1][name], right[1][name], rtol=0, atol=0)
        for index, state in left[2]['state'].items():
            for name, value in state.items(): torch.testing.assert_close(value, right[2]['state'][index][name], rtol=0, atol=0)
        torch.testing.assert_close(left[3], right[3], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
