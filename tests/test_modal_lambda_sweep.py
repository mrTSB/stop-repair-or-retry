"""Offline checks for the Controller B lambda-sweep driver's launch guards. No Modal, network or GPU."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

DRIVER = Path(__file__).resolve().parents[1] / 'experiments/llm-controller/modal_lambda_sweep.py'
spec = importlib.util.spec_from_file_location('modal_lambda_sweep', DRIVER)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)  # must not import modal at module level


class LaunchGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fresh = Path(self.tmp.name) / 'new-run'

    def tearDown(self):
        self.tmp.cleanup()

    def paid(self, **overrides):
        args = dict(penalty=0.248, profile='me', env_profile='me', prior_reserved_usd=0.0,
                    output=self.fresh, execute=True, dry_run=False)
        args.update(overrides)
        return sweep.check_launch(**args)

    def test_valid_paid_launch_returns_reservation_below_cap(self):
        reservation = self.paid()
        self.assertGreater(reservation, 0)
        self.assertLess(reservation, sweep.CAP_USD)

    def test_rejects_undeclared_penalty(self):
        for penalty in (0.01, 0.5, 0):
            with self.subTest(penalty=penalty), self.assertRaises(sweep.GuardError):
                self.paid(penalty=penalty)

    def test_requires_explicit_execute(self):
        with self.assertRaises(sweep.GuardError):
            self.paid(execute=False)

    def test_requires_matching_profile(self):
        for profile, env in (('', ''), ('me', None), ('me', 'gancone')):
            with self.subTest(profile=profile, env=env), self.assertRaises(sweep.GuardError):
                self.paid(profile=profile, env_profile=env)

    def test_enforces_cumulative_cap(self):
        just_fits = sweep.CAP_USD - sweep.RESERVATION_USD - 0.01
        self.assertEqual(self.paid(prior_reserved_usd=just_fits), sweep.RESERVATION_USD)
        for prior in (sweep.CAP_USD - sweep.RESERVATION_USD, sweep.CAP_USD, -1):
            with self.subTest(prior=prior), self.assertRaises(sweep.GuardError):
                self.paid(prior_reserved_usd=prior)

    def test_three_runs_fit_under_cap_from_zero(self):
        self.assertLess(3 * sweep.RESERVATION_USD, sweep.CAP_USD)

    def test_refuses_existing_output(self):
        self.fresh.mkdir()
        with self.assertRaises(sweep.GuardError):
            self.paid()

    def test_dry_run_must_write_under_runs(self):
        with self.assertRaises(sweep.GuardError):
            self.paid(execute=False, dry_run=True)
        inside = sweep.ROOT / 'runs' / 'definitely-not-existing-dry-run-dir'
        self.assertEqual(self.paid(execute=False, dry_run=True, profile='', env_profile=None, output=inside), 0.0)

    def test_dry_run_worker_matches_response_shape(self):
        import json
        worker = sweep.DryRunWorker('fake')
        body = {'messages': [{}, {'content': json.dumps({'remaining_solver_calls': 2})}]}
        response = worker(body)
        self.assertEqual(json.loads(response['choices'][0]['message']['content']), {'action': 'repair'})
        self.assertEqual(response['usage']['total_tokens'], response['usage']['prompt_tokens'] + response['usage']['completion_tokens'])


if __name__ == '__main__':
    unittest.main()
