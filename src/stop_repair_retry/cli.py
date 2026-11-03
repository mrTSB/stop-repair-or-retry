import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from .core import Model, read_jsonl, write_jsonl
from .policies import TabularQ, majority, simulate, tune

POLICIES = ('stop', 'repair', 'retry', 'schedule', 'threshold', 'one_step', 'supervised', 'pomdp', 'judge_full', 'judge_compact', 'hybrid', 'majority')


def bootstrap_interval(values, seed=0, repetitions=2000):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(repetitions))
    return [means[int(.025 * repetitions)], means[min(repetitions - 1, int(.975 * repetitions))]]


def evaluate(args):
    train, dev, test = (read_jsonl(p) for p in (args.train, args.dev, args.test))
    if not all((train, dev, test)): raise ValueError('Each split must be nonempty')
    ids = [set(str(r['id']) for r in group) for group in (train, dev, test)]
    if any(len(ids[i]) != len(group) for i, group in enumerate((train, dev, test))) or ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError('Question ids must be unique within and disjoint across splits')
    policies = args.policies.split(',')
    if set(policies) - set(POLICIES): raise ValueError('Unknown policies: ' + str(set(policies) - set(POLICIES)))
    if set(policies) & {'judge_full', 'judge_compact', 'hybrid'}:
        for row in dev + test:
            if row.get('judge_penalty') != args.penalty:
                raise ValueError('Judge data must be generated with the same --penalty as evaluation')
            if any('judges' not in node for path, node in row['tree'].items() if not path or len(path.split('/')) < 2):
                raise ValueError('Judge policies require precomputed judge calls on every decision node')
    model, qmodel = Model.fit(train), TabularQ(train)
    details, summaries = [], []
    all_results = {}
    for policy in policies:
        setting = tune(dev, policy, model, qmodel, args.penalty)
        results = [majority(row) if policy == 'majority' else simulate(row, policy, model, qmodel, args.penalty, setting) for row in test]
        all_results[policy] = results
        for result in results:
            details.append({'policy': policy, **result})
        mean = lambda key: statistics.mean(r[key] for r in results)
        summaries.append({'policy': policy, 'setting_selected_on_dev': setting, 'n': len(results), 'accuracy': mean('correct'), 'accuracy_ci_95': bootstrap_interval([r['correct'] for r in results]), 'mean_tokens': mean('tokens'), 'mean_input_tokens': mean('input_tokens'), 'mean_output_tokens': mean('output_tokens'), 'mean_solver_latency_seconds': mean('solver_latency_seconds'), 'mean_judge_latency_seconds': mean('judge_latency_seconds'), 'mean_judge_calls': mean('judge_calls'), 'wrong_to_right': statistics.mean(int(not r['initial_correct'] and r['correct']) for r in results), 'right_to_wrong': statistics.mean(int(r['initial_correct'] and not r['correct']) for r in results), 'mean_utility': statistics.mean(r['correct'] - args.penalty * r['tokens'] / 1000 for r in results)})
    if 'stop' in all_results:
        for summary in summaries:
            diffs = [r['correct'] - baseline['correct'] for r, baseline in zip(all_results[summary['policy']], all_results['stop'])]
            summary['paired_accuracy_diff_vs_stop'] = statistics.mean(diffs)
            summary['paired_diff_ci_95'] = bootstrap_interval(diffs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'per_question.jsonl', details)
    (out / 'summary.json').write_text(json.dumps({'penalty': args.penalty, 'results': summaries}, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Run cost-aware reasoning controller experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate', help='Generate all branches with a local MLX model')
    gen.add_argument('--input', required=True, help='Question JSONL: id, question, answer')
    gen.add_argument('--output', required=True)
    gen.add_argument('--model', required=True, help='MLX-compatible Hugging Face model or local directory')
    gen.add_argument('--backend', choices=('mlx', 'ollama'), default='mlx', help='Use MLX directly or an already running local Ollama server')
    gen.add_argument('--judge', action='store_true', help='Precompute full and compact judge calls with the same model')
    gen.add_argument('--penalty', type=float, default=.01)
    gen.add_argument('--max-tokens', type=int, default=512)
    gen.add_argument('--temperature', type=float, default=.7)
    gen.add_argument('--depth', type=int, choices=(1, 2), default=2, help='Decision-tree depth: 1 for one stop/repair/retry checkpoint; 2 for adaptive two-decision trajectories')
    gen.add_argument('--seed', type=int, default=0)
    gen.add_argument('--resume', action='store_true', help='Resume a matching, incrementally saved output')
    ev = sub.add_parser('evaluate', help='Fit on train, tune on dev, evaluate on test')
    for arg in ('train', 'dev', 'test', 'output'): ev.add_argument('--' + arg, required=True)
    ev.add_argument('--penalty', type=float, default=.01)
    ev.add_argument('--policies', default='stop,repair,retry,schedule,threshold,one_step,supervised,pomdp,majority')
    args = parser.parse_args()
    if args.command == 'generate':
        from . import generate
        from .generate import MLXBackend, OllamaBackend, generate_rows
        backend_type = MLXBackend if args.backend == 'mlx' else OllamaBackend
        config = {'backend': args.backend, 'model': args.model, 'max_tokens': args.max_tokens, 'temperature': args.temperature, 'seed': args.seed, 'depth': args.depth, 'judge': args.judge, 'penalty': args.penalty if args.judge else None, 'generator_sha256': hashlib.sha256(Path(generate.__file__).read_bytes()).hexdigest()}
        backend = backend_type(args.model, args.max_tokens, args.temperature, args.seed)
        generate_rows(read_jsonl(args.input), backend, args.output, backend if args.judge else None, args.penalty, args.seed, config, args.resume, args.depth)
    else: evaluate(args)

if __name__ == '__main__': main()
