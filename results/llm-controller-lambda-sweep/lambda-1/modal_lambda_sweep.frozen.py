"""Controller B at additional token penalties: bounded dev/freeze/test collection on private Modal RPC.

Reuses the unchanged GPU worker (`Controller`) from modal_full_evaluation.py: same Qwen2.5-14B-Instruct
revision, prompt, greedy decoding, 128-token cap, and A100-80GB resources. Only the declared penalty,
the Modal profile and the spending reservation differ. The original lambda 0.01 archive is not modified.

Penalties: 0.1 and 1 are on the predeclared grid in reward_design.json. 0.248 is NOT on that grid; it was
selected from Controller A's (POMDP) dev-only diagnostic and is included to compare A and B at the same penalty.

`--dry-run` exercises the complete dev -> freeze -> test pipeline offline with a deterministic fake worker:
no Modal import, no network, $0. Its output must stay under runs/ and is not LLM performance.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PENALTIES = (0.1, 0.248, 1.0)
PREDECLARED_GRID = (0, 0.001, 0.01, 0.1, 1)
CAP_USD = 20.0
WALL_LIMIT_SECONDS = 1800
MAX_CALLS = 920
# Same Modal rates as modal_full_evaluation.py (A100-80GB + 4 CPU + 48 GiB, USD/second). Recheck before paid runs.
RATE_USD_PER_SECOND = .000694 + 4 * .0000131 + 48 * .00000222
# Watchdog lifetime plus startup/call slack, plus a $1 fee margin (same formula as the original driver).
RESERVATION_USD = (WALL_LIMIT_SECONDS + 600 + 120) * RATE_USD_PER_SECOND + 1


class GuardError(SystemExit):
    pass


def check_launch(penalty, profile, env_profile, prior_reserved_usd, output, execute, dry_run):
    """Refuse to start unless every operational guard passes. Returns the new reservation in USD."""
    if penalty not in PENALTIES:
        raise GuardError(f'Penalty must be one of {PENALTIES}.')
    output = Path(output).resolve()
    if output.exists():
        raise GuardError('Refusing overwrite or automatic replay: output already exists.')
    if dry_run:
        if ROOT / 'runs' not in output.parents:
            raise GuardError('Dry-run output must be under runs/ (git-ignored).')
        return 0.0
    if not execute:
        raise GuardError('Paid collection requires explicit --execute.')
    if not profile or env_profile != profile:
        raise GuardError('MODAL_PROFILE must be set and equal --profile.')
    if not 0 <= prior_reserved_usd < CAP_USD or prior_reserved_usd + RESERVATION_USD >= CAP_USD:
        raise GuardError(f'Cumulative reservation would reach the ${CAP_USD:.0f} cap.')
    return RESERVATION_USD


class DryRunWorker:
    """Deterministic stand-in shaped like Controller.decide's response. Not LLM behavior."""

    def __init__(self, model):
        self.model = model

    def __call__(self, body):
        observation = json.loads(body['messages'][1]['content'])
        action = 'repair' if observation['remaining_solver_calls'] == 2 else 'stop'
        return {'id': 'dry-run', 'model': self.model,
                'choices': [{'message': {'content': json.dumps({'action': action})}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 8, 'total_tokens': 108}}


def collect(experiment, config, prompt, output, client):
    """Same sequence as the original full evaluation: all dev, evaluate, freeze unchanged settings, all test."""
    dev, test, freeze = output / 'dev', output / 'test', output / 'frozen.json'
    experiment.run(config, prompt, experiment.DEFAULT_DATA, 'dev', None, dev, 'live', client=client)
    experiment.evaluate(dev)
    experiment.freeze_run(dev, freeze)
    print('All 60 dev complete; settings frozen. Starting test without changing settings.', flush=True)
    experiment.run(config, prompt, experiment.DEFAULT_DATA, 'test', None, test, 'live', freeze=freeze, client=client)
    experiment.completed_records(test)


def start_watchdog(output, app_id, profile):
    """Independent process that stops the Modal app at the deadline, even if this driver is interrupted."""
    source = (
        'import subprocess,sys,time\n'
        'time.sleep(float(sys.argv[2]))\n'
        'for attempt in range(3):\n'
        '    try:\n'
        "        result=subprocess.run([sys.executable,'-m','modal','app','stop',sys.argv[1],'--profile',sys.argv[3],'--yes'],timeout=60)\n"
        '        if result.returncode==0:break\n'
        '    except subprocess.TimeoutExpired:pass\n'
        '    time.sleep(5)\n')
    with (output / 'watchdog.log').open('w') as log:
        return subprocess.Popen([sys.executable, '-c', source, app_id, str(WALL_LIMIT_SECONDS), profile],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def collect_on_modal(experiment, backend, config, prompt, output, profile, started):
    """Paid path: one temporary A100 app shared by dev and test, with request/time limits and a watchdog."""
    import modal
    shutil.copyfile(__file__, output / 'modal_lambda_sweep.frozen.py')
    shutil.copyfile(HERE / 'modal_full_evaluation.py', output / 'modal_full_evaluation.frozen.py')
    calls = 0
    watchdog = None
    with modal.enable_output(), backend.app.run():
        experiment.save(output / 'modal-app.json', {'app_id': backend.app.app_id, 'profile': profile, 'started_utc': experiment.now()})
        watchdog = start_watchdog(output, backend.app.app_id, profile)
        experiment.save(output / 'watchdog.json', {'pid': watchdog.pid, 'deadline_seconds': WALL_LIMIT_SECONDS,
                                                   'action': 'modal app stop --yes', 'independent_process': True})
        remote = backend.Controller()
        experiment.save(output / 'backend.json', remote.metadata.remote())

        def client(body):
            nonlocal calls
            if calls >= MAX_CALLS or time.monotonic() - started > WALL_LIMIT_SECONDS - 120:
                raise RuntimeError('Global decision or time budget exhausted.')
            calls += 1
            response = remote.decide.remote(body)
            if calls % 20 == 0:
                print(f'Controller decisions received: {calls}/{MAX_CALLS}', flush=True)
            return response
        collect(experiment, config, prompt, output, client)
    if watchdog is not None:
        watchdog.terminate()
        watchdog.wait(timeout=10)
    return calls


def main():
    sys.path.insert(0, str(ROOT / 'src'))
    from stop_repair_retry import llm_experiment as experiment
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--penalty', required=True, type=float)
    p.add_argument('--output', required=True)
    p.add_argument('--profile', default='')
    p.add_argument('--prior-reserved-usd', type=float, default=0.0)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true', help='paid Modal collection')
    mode.add_argument('--dry-run', action='store_true', help='offline fake worker, no network, $0')
    args = p.parse_args()
    reservation = check_launch(args.penalty, args.profile, os.environ.get('MODAL_PROFILE'), args.prior_reserved_usd,
                               args.output, args.execute, args.dry_run)
    output = Path(args.output).resolve()
    backend = None
    if args.dry_run:
        model_id, profile = 'dry-run-fake-worker', 'none'
    else:
        sys.path.insert(0, str(HERE))
        import modal_full_evaluation as backend  # the unchanged GPU worker; imports modal
        model_id, profile = backend.MODEL_ID, args.profile
    config = experiment.load_config(ROOT / 'experiments/llm-controller/config.example.json')
    config.update(endpoint=f'https://modal.com/apps/{profile}/main/cs238-llm-controller-full-evaluation', model=model_id,
                  penalty=args.penalty, response_format='none', billing_mode='compute', compute_budget_confirmed=True,
                  max_requests=800, max_run_seconds=1200, timeout_seconds=120)
    prompt = experiment.DEFAULT_PROMPT.read_text()
    output.mkdir(parents=True)
    experiment.save(output / 'config.json', config)
    experiment.save(output / 'protocol.json', {
        'created_utc': experiment.now(), 'controller': 'Controller B (LLM), proposal naming',
        'penalty': args.penalty, 'penalty_on_predeclared_grid': args.penalty in PREDECLARED_GRID,
        'penalty_note': ('On the predeclared grid in reward_design.json.' if args.penalty in PREDECLARED_GRID
                         else 'Not predeclared: selected from Controller A (POMDP) dev-only diagnostic to compare A and B at one penalty.'),
        'sequence': ['all 60 dev', 'evaluate dev', 'freeze unchanged settings', 'all 400 test', 'evaluate test'],
        'model': model_id, 'dry_run': args.dry_run,
        'driver_sha256': experiment.sha(__file__), 'worker_sha256': experiment.sha(HERE / 'modal_full_evaluation.py'),
        'prompt_sha256': experiment.sha(experiment.DEFAULT_PROMPT),
        'changes_from_lambda_0.01_run': 'Penalty (shown to the controller and used in utility), Modal profile and spending reservation only. Same worker code, weights, prompt, decoding, parser and visited-history inputs.',
        'max_decisions': MAX_CALLS, 'wall_watchdog_seconds': WALL_LIMIT_SECONDS,
        'prior_reserved_usd': args.prior_reserved_usd, 'new_reservation_usd': reservation,
        'cumulative_reserved_usd': args.prior_reserved_usd + reservation, 'user_cap_usd': CAP_USD,
        'rate_usd_per_second': RATE_USD_PER_SECOND,
        'split_note': 'Existing fixed-baseline and controller test outcomes have been viewed previously; not an unseen confirmation set.',
    })
    started = time.monotonic()
    if args.dry_run:
        collect(experiment, config, prompt, output, DryRunWorker(model_id))
        calls = None
    else:
        calls = collect_on_modal(experiment, backend, config, prompt, output, profile, started)
    experiment.save(output / 'completion.json', {'app_context_closed': True, 'controller_calls': calls, 'dry_run': args.dry_run,
                                                 'client_wall_seconds': time.monotonic() - started, 'finished_utc': experiment.now()})
    experiment.evaluate(output / 'test')
    print(json.dumps({'status': 'complete', 'penalty': args.penalty, 'controller_calls': calls, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
