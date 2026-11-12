"""Controller B v2: cost-explicit prompt, collected in two phases (dev -> freeze, then test) on private Modal RPC.

Differences from v1 are confined to the system prompt (prompt_v2_template.txt): it states lambda, the
estimated additional cost of repair/retry in tokens and objective units, the controller's own per-decision
cost, and two-sided guidance about stopping vs continuing. The GPU worker, model revision, decoding,
request schema, parser, replay and scoring are unchanged (worker: modal_full_evaluation.py).

Cost estimates use only: mean solver tokens per action and depth on the 100 training trees, and mean
controller tokens per decision from v1's archived dev run. No test data inform the prompt.

Phases: `--phase dev` collects all 60 dev questions, evaluates them and freezes. Review dev before running
`--phase test` on the same output directory, which collects all 400 test questions under that freeze.
`--dry-run` runs a phase offline with a fake worker under runs/ ($0, not LLM behavior).
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import statistics
from string import Template
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import modal_lambda_sweep as sweep  # guards, watchdog and fake worker; does not import modal

PENALTIES = (0.01, 0.1, 0.248, 1.0)
TEMPLATE = HERE / 'prompt_v2_template.txt'
TRAIN_TREES = ROOT / 'results/cached-tree-dataset/data/train-trees.jsonl.gz'
V1_DEV_EVENTS = ROOT / 'results/llm-controller-full-evaluation/dev/events.jsonl'


def cost_inputs():
    """Mean solver tokens per action/depth (train only) and mean controller tokens per decision (v1 dev)."""
    rows = [json.loads(line) for line in gzip.open(TRAIN_TREES, 'rt', encoding='utf-8')]
    tokens = lambda node: int(node['input_tokens']) + int(node['output_tokens'])
    solver = {}
    for action in ('repair', 'retry'):
        solver[(action, 2)] = statistics.mean(tokens(r['tree'][action]) for r in rows)
        solver[(action, 1)] = statistics.mean(tokens(r['tree'][p + '/' + action]) for r in rows for p in ('repair', 'retry'))
    usage = [json.loads(line)['response'] for line in V1_DEV_EVENTS.read_text().splitlines() if json.loads(line)['type'] == 'response']
    usage = [(json.loads(u) if isinstance(u, str) else u)['usage'] for u in usage]
    controller = statistics.mean(u['prompt_tokens'] + u['completion_tokens'] for u in usage)
    return solver, controller


def cost_table(penalty, solver, controller):
    """Additional tokens if chosen now: the solver call, plus one more controller decision if a call would remain."""
    table = {}
    for remaining in (2, 1):
        for action in ('repair', 'retry'):
            extra = solver[(action, remaining)] + (controller if remaining == 2 else 0)
            table[f'{action}{remaining}_tokens'] = round(extra)
            table[f'{action}{remaining}_cost'] = f'{penalty * extra / 1000:.3f}'
    return table


def render_prompt(penalty):
    solver, controller = cost_inputs()
    values = cost_table(penalty, solver, controller)
    values.update(lam=f'{penalty:g}', controller_tokens=round(controller))
    text = Template(TEMPLATE.read_text()).substitute(values)
    provenance = {'solver_mean_tokens_train': {f'{a}_with_{r}_remaining': v for (a, r), v in solver.items()},
                  'controller_mean_tokens_per_decision_v1_dev': controller, 'rendered_values': values}
    return text, provenance


def check_launch(phase, penalty, profile, env_profile, prior_reserved_usd, output, execute, dry_run):
    """Phase-aware guard. Returns the new reservation in USD."""
    if penalty not in PENALTIES:
        raise sweep.GuardError(f'Penalty must be one of {PENALTIES}.')
    output = Path(output).resolve()
    if phase == 'dev' and output.exists():
        raise sweep.GuardError('Dev phase needs a fresh output directory.')
    if phase == 'test' and not ((output / 'frozen.json').exists() and (output / 'dev').exists() and not (output / 'test').exists()):
        raise sweep.GuardError('Test phase needs a completed, frozen dev phase in this directory and no existing test.')
    if dry_run:
        if ROOT / 'runs' not in output.parents:
            raise sweep.GuardError('Dry-run output must be under runs/ (git-ignored).')
        return 0.0
    if not execute:
        raise sweep.GuardError('Paid collection requires explicit --execute.')
    if not profile or env_profile != profile:
        raise sweep.GuardError('MODAL_PROFILE must be set and equal --profile.')
    if not 0 <= prior_reserved_usd < sweep.CAP_USD or prior_reserved_usd + sweep.RESERVATION_USD >= sweep.CAP_USD:
        raise sweep.GuardError(f'Cumulative reservation would reach the ${sweep.CAP_USD:.0f} cap.')
    return sweep.RESERVATION_USD


def run_phase(experiment, phase, config, prompt, output, client):
    if phase == 'dev':
        experiment.run(config, prompt, experiment.DEFAULT_DATA, 'dev', None, output / 'dev', 'live', client=client)
        experiment.evaluate(output / 'dev')
        experiment.freeze_run(output / 'dev', output / 'frozen.json')
    else:
        experiment.run(config, prompt, experiment.DEFAULT_DATA, 'test', None, output / 'test', 'live', freeze=output / 'frozen.json', client=client)
        experiment.completed_records(output / 'test')


def on_modal(experiment, backend, phase, config, prompt, output, profile, started):
    import modal
    calls = 0
    with modal.enable_output(), backend.app.run():
        experiment.save(output / f'modal-app-{phase}.json', {'app_id': backend.app.app_id, 'profile': profile, 'started_utc': experiment.now()})
        watchdog = sweep.start_watchdog(output, backend.app.app_id, profile)
        (output / 'watchdog.log').rename(output / f'watchdog-{phase}.log')
        remote = backend.Controller()
        experiment.save(output / f'backend-{phase}.json', remote.metadata.remote())

        def client(body):
            nonlocal calls
            if calls >= sweep.MAX_CALLS or time.monotonic() - started > sweep.WALL_LIMIT_SECONDS - 120:
                raise RuntimeError('Global decision or time budget exhausted.')
            calls += 1
            response = remote.decide.remote(body)
            if calls % 20 == 0:
                print(f'Controller decisions received: {calls}', flush=True)
            return response
        run_phase(experiment, phase, config, prompt, output, client)
    watchdog.terminate()
    watchdog.wait(timeout=10)
    return calls


def main():
    sys.path.insert(0, str(ROOT / 'src'))
    from stop_repair_retry import llm_experiment as experiment
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--phase', required=True, choices=('dev', 'test'))
    p.add_argument('--penalty', required=True, type=float)
    p.add_argument('--output', required=True)
    p.add_argument('--profile', default='')
    p.add_argument('--prior-reserved-usd', type=float, default=0.0)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true', help='paid Modal collection')
    mode.add_argument('--dry-run', action='store_true', help='offline fake worker, no network, $0')
    args = p.parse_args()
    reservation = check_launch(args.phase, args.penalty, args.profile, os.environ.get('MODAL_PROFILE'),
                               args.prior_reserved_usd, args.output, args.execute, args.dry_run)
    output = Path(args.output).resolve()
    backend = None
    if args.dry_run:
        model_id, profile = 'dry-run-fake-worker', 'none'
    else:
        import modal_full_evaluation as backend  # unchanged GPU worker; imports modal
        model_id, profile = backend.MODEL_ID, args.profile
    prompt, provenance = render_prompt(args.penalty)
    if args.phase == 'dev':
        config = experiment.load_config(ROOT / 'experiments/llm-controller/config.example.json')
        config.update(endpoint=f'https://modal.com/apps/{profile}/main/cs238-llm-controller-full-evaluation', model=model_id,
                      penalty=args.penalty, response_format='none', billing_mode='compute', compute_budget_confirmed=True,
                      max_requests=800, max_run_seconds=1200, timeout_seconds=120)
        output.mkdir(parents=True)
        experiment.save(output / 'config.json', config)
        (output / 'prompt.txt').write_text(prompt)
        experiment.save(output / 'protocol.json', {
            'created_utc': experiment.now(), 'controller': 'Controller B v2 (LLM), proposal naming', 'penalty': args.penalty,
            'model': model_id, 'dry_run': args.dry_run,
            'changes_from_v1': ['States lambda and the estimated additional cost of repair/retry in tokens and objective units, by remaining calls',
                                "States the controller's own per-decision token cost",
                                'Two-sided stop/continue guidance replaces the one-sided "do not stop solely because the solver sounds confident" nudge'],
            'unchanged_from_v1': 'GPU worker, model revision, decoding, 128-token cap, request schema, action parser, replay and scoring',
            'cost_provenance': provenance,
            'cost_sources': 'Solver tokens: training trees only. Controller tokens: v1 archived dev run (dev is the tuning split). No test data.',
            'sequence': ['phase dev: all 60 dev, evaluate, freeze', 'review dev', 'phase test: all 400 test under the freeze, evaluate'],
            'driver_sha256': experiment.sha(__file__), 'template_sha256': experiment.sha(TEMPLATE),
            'rendered_prompt_sha256': experiment.sha(output / 'prompt.txt'),
            'worker_sha256': experiment.sha(HERE / 'modal_full_evaluation.py'),
            'split_note': 'Existing test outcomes have been viewed previously; not an unseen confirmation set.',
        })
    else:
        config = experiment.read_json(output / 'config.json')
        if prompt != (output / 'prompt.txt').read_text():
            raise sweep.GuardError('Rendered prompt differs from the dev phase; refusing test.')
    experiment.save(output / f'reservation-{args.phase}.json', {'prior_reserved_usd': args.prior_reserved_usd, 'new_reservation_usd': reservation,
                                                               'cumulative_reserved_usd': args.prior_reserved_usd + reservation, 'user_cap_usd': sweep.CAP_USD})
    started = time.monotonic()
    if args.dry_run:
        run_phase(experiment, args.phase, config, prompt, output, sweep.DryRunWorker(model_id))
        calls = None
    else:
        calls = on_modal(experiment, backend, args.phase, config, prompt, output, profile, started)
    if args.phase == 'test':
        experiment.evaluate(output / 'test')
    experiment.save(output / f'completion-{args.phase}.json', {'controller_calls': calls, 'dry_run': args.dry_run,
                                                              'client_wall_seconds': time.monotonic() - started, 'finished_utc': experiment.now()})
    print(json.dumps({'status': 'complete', 'phase': args.phase, 'penalty': args.penalty, 'controller_calls': calls, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
