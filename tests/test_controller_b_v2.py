"""Offline checks for the Controller B v2 prompt and phase guards. No Modal, network or GPU."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

DRIVER = Path(__file__).resolve().parents[1] / 'experiments/llm-controller/modal_controller_b_v2.py'
spec = importlib.util.spec_from_file_location('modal_controller_b_v2', DRIVER)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)  # must not import modal at module level


class PromptTests(unittest.TestCase):
    def test_cost_inputs_are_plausible(self):
        solver, controller = v2.cost_inputs()
        for key in (('repair', 2), ('repair', 1), ('retry', 2), ('retry', 1)):
            self.assertGreater(solver[key], 100)
        self.assertGreater(solver[('repair', 2)], solver[('retry', 2)])  # repair re-reads the previous answer
        self.assertGreater(controller, 300)

    def test_cost_includes_follow_up_decision_only_with_two_remaining(self):
        solver, controller = v2.cost_inputs()
        table = v2.cost_table(0.248, solver, controller)
        self.assertEqual(table['repair2_tokens'], round(solver[('repair', 2)] + controller))
        self.assertEqual(table['retry1_tokens'], round(solver[('retry', 1)]))
        self.assertEqual(table['repair2_cost'], f"{0.248 * (solver[('repair', 2)] + controller) / 1000:.3f}")

    def test_rendered_prompt_is_complete_and_states_penalty(self):
        for penalty in v2.PENALTIES:
            with self.subTest(penalty=penalty):
                text, provenance = v2.render_prompt(penalty)
                self.assertNotIn('$', text)
                self.assertIn(f'In this task lambda is {penalty:g}.', text)
                self.assertIn('{"action":"stop"}', text)
                self.assertIn('rendered_values', provenance)

    def test_costs_scale_with_penalty(self):
        low, _ = v2.render_prompt(0.01)
        high, _ = v2.render_prompt(1.0)
        self.assertNotEqual(low, high)


class PhaseGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'run'

    def tearDown(self):
        self.tmp.cleanup()

    def guard(self, **overrides):
        args = dict(phase='dev', penalty=0.248, profile='me', env_profile='me', prior_reserved_usd=1.06,
                    output=self.out, execute=True, dry_run=False)
        args.update(overrides)
        return v2.check_launch(**args)

    def test_dev_needs_fresh_directory(self):
        self.assertGreater(self.guard(), 0)
        self.out.mkdir()
        with self.assertRaises(v2.sweep.GuardError):
            self.guard()

    def test_test_needs_frozen_dev_and_no_existing_test(self):
        with self.assertRaises(v2.sweep.GuardError):
            self.guard(phase='test')
        (self.out / 'dev').mkdir(parents=True)
        with self.assertRaises(v2.sweep.GuardError):
            self.guard(phase='test')
        (self.out / 'frozen.json').write_text('{}')
        self.assertGreater(self.guard(phase='test'), 0)
        (self.out / 'test').mkdir()
        with self.assertRaises(v2.sweep.GuardError):
            self.guard(phase='test')

    def test_penalty_profile_execute_and_cap(self):
        for bad in (dict(penalty=0.5), dict(execute=False), dict(env_profile='other'), dict(prior_reserved_usd=19.0)):
            with self.subTest(**bad), self.assertRaises(v2.sweep.GuardError):
                self.guard(**bad)
        self.assertGreater(self.guard(penalty=0.01), 0)


if __name__ == '__main__':
    unittest.main()
