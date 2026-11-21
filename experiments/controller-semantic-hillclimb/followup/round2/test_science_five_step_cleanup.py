"""CPU-only arithmetic/RNG and exact recovery tests for five-step cleanup."""
from contextlib import ExitStack
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import resume_science_five_step_cleanup as run
from test_science_training_checkpoint import setup


def fake_mps(stack, events):
    stack.enter_context(patch.object(run.trainer.torch.mps, 'current_allocated_memory', return_value=10))
    stack.enter_context(patch.object(run.trainer.torch.mps, 'driver_allocated_memory', return_value=20))
    stack.enter_context(patch.object(run.trainer.torch.mps, 'synchronize', side_effect=lambda: events.append('sync')))
    stack.enter_context(patch.object(run.trainer.torch.mps, 'empty_cache', side_effect=lambda: events.append('empty')))


class FiveStepTests(unittest.TestCase):
    def test_history_boundary_order_interval_and_no_final_signal(self):
        events = []
        def append(path, value): events.append(('append', Path(path).name, value.get('step')))
        with ExitStack() as stack:
            fake_mps(stack, events)
            stack.enter_context(patch.object(run.trainer, 'append', side_effect=append))
            stack.enter_context(patch.object(run.gc, 'collect', side_effect=lambda: events.append('gc')))
            stack.enter_context(patch.object(run.os, 'kill', side_effect=AssertionError('No artificial final signal')))
            original = run.trainer.append
            with run.operational_hooks(Path('history'), Path('telemetry')):
                run.trainer.append(Path('history'), {'step': 104})
                self.assertEqual(events, [('append', 'history', 104)])
                events.clear(); run.trainer.append(Path('history'), {'step': 105})
                self.assertEqual(events, [('append', 'history', 105), 'gc', 'sync', 'empty', 'sync', ('append', 'telemetry', 105)])
                events.clear(); run.trainer.append(Path('unrelated'), {'step': 105})
                self.assertEqual(events, [('append', 'unrelated', 105)])
                run.trainer.append(Path('history'), {'step': 2160})
            self.assertIs(run.trainer.append, original)

    def test_pretelemetry_cleanup_cpu_noop_and_failed_append_has_no_cleanup(self):
        events = []
        with ExitStack() as stack, TemporaryDirectory() as directory:
            fake_mps(stack, events)
            stack.enter_context(patch.object(run.trainer, 'memory_snapshot', side_effect=lambda dev: events.append('measure:'+dev) or {'system_free_percent': 42}))
            original = run.trainer.memory_snapshot
            with run.operational_hooks(Path(directory)/'history', Path(directory)/'telemetry'):
                run.trainer.memory_snapshot('cpu'); self.assertEqual(events, ['measure:cpu'])
                events.clear(); run.trainer.memory_snapshot('mps')
                self.assertEqual(events, ['sync', 'empty', 'sync', 'measure:mps'])
            self.assertIs(run.trainer.memory_snapshot, original)
            with patch.object(run.trainer, 'append', side_effect=RuntimeError('Not durable')):
                events.clear()
                with run.operational_hooks(Path(directory)/'history', Path(directory)/'telemetry'):
                    with self.assertRaises(RuntimeError): run.trainer.append(Path(directory)/'history', {'step': 105})
                self.assertEqual(events, [])

    def test_exact_interrupted_resume_loss_parameters_optimizer_scheduler_and_rng(self):
        def update(model, optimizer, scheduler, history, number):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(3, 5) + np.random.random() + random.random()
            y = torch.tensor([0, 1, 0]); output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward(); optimizer.step(); scheduler.step()
            value = float(loss.detach())
            # Same frozen-trainer hook position: gradient/output references
            # still exist here, but backward and both updates are complete.
            run.trainer.append(history, {'step': number, 'loss': value})
            optimizer.zero_grad(set_to_none=True)
            return value
        with TemporaryDirectory() as directory, ExitStack() as stack:
            fake_mps(stack, [])
            root = Path(directory); full_history = root/'full'; full_history.touch()
            full, full_opt, full_sched = setup(31)
            expected = [update(full, full_opt, full_sched, full_history, n) for n in range(1, 12)]
            expected_rng = run.trainer.checkpoint.rng_state('cpu')
            history = root/'history'; history.touch(); model, opt, sched = setup(31)
            with run.operational_hooks(history, root/'telemetry'):
                actual = [update(model, opt, sched, history, n) for n in range(1, 6)]
                run.trainer.checkpoint.save(root/'checkpoints', model, opt, sched, {'step': 5}, {'same_recipe': True}, 'cpu', history, chunk_bytes=32)
                # Preserve two abandoned complete updates before exact recovery.
                for n in (6, 7): update(model, opt, sched, history, n)
                resumed, resumed_opt, resumed_sched = setup(999)
                position = run.trainer.checkpoint.restore(root/'checkpoints', resumed, resumed_opt, resumed_sched,
                    {'same_recipe': True}, 'cpu', history)
                self.assertEqual(position['step'], 5)
                actual += [update(resumed, resumed_opt, resumed_sched, history, n) for n in range(6, 12)]
            self.assertEqual(actual, expected); self.assertEqual(history.read_bytes(), full_history.read_bytes())
            for key, value in full.state_dict().items(): torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(full_sched.state_dict(), resumed_sched.state_dict())
            self.assertEqual(full_opt.state_dict()['param_groups'], resumed_opt.state_dict()['param_groups'])
            for key, values in full_opt.state_dict()['state'].items():
                for name, value in values.items(): torch.testing.assert_close(value, resumed_opt.state_dict()['state'][key][name], rtol=0, atol=0)
            actual_rng = run.trainer.checkpoint.rng_state('cpu')
            self.assertEqual(actual_rng['python'], expected_rng['python'])
            self.assertEqual(actual_rng['numpy'][0], expected_rng['numpy'][0]); np.testing.assert_array_equal(actual_rng['numpy'][1], expected_rng['numpy'][1])
            self.assertEqual(actual_rng['numpy'][2:], expected_rng['numpy'][2:])
            torch.testing.assert_close(actual_rng['torch_cpu'], expected_rng['torch_cpu'], rtol=0, atol=0)
            self.assertIsNone(actual_rng['torch_mps'])


if __name__ == '__main__': unittest.main()
