"""Summarize a completed baseline evaluation without changing its outcomes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    root = Path(args.run_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    result = json.loads((root / 'results' / 'summary.json').read_text())
    lines = ['# GSM8K baseline results', '', f"Dataset: {manifest['dataset']}; split seed {manifest['seed']}; {manifest['counts']['train']} training, {manifest['counts']['dev']} development, {manifest['counts']['test']} test questions. The test set excludes the inspected pilot questions.", '', f"Primary cost penalty: {result['penalty']} per 1,000 total input and output tokens. All policies use paired cached solver trajectories. Intervals are question-level bootstrap intervals; their validity assumes sampled questions are representative and does not include model-seed or benchmark contamination uncertainty.", '', '| Policy | Accuracy | 95% CI | Mean tokens | Utility | Accuracy difference vs stop (95% CI) |', '|---|---:|---:|---:|---:|---:|']
    for row in result['results']:
        ci = row['accuracy_ci_95']
        diff = row.get('paired_accuracy_diff_vs_stop', 0)
        dci = row.get('paired_diff_ci_95', [0, 0])
        lines.append(f"| {row['policy']} | {row['accuracy']:.1%} | [{ci[0]:.1%}, {ci[1]:.1%}] | {row['mean_tokens']:.0f} | {row['mean_utility']:.4f} | {diff:+.1%} [{dci[0]:+.1%}, {dci[1]:+.1%}] |")
    lines += ['', 'The full-history and compact judges use the solver model. Their prompts are conditioned on the primary cost penalty; the additional penalty-grid outputs include only non-judge policies. The full cached tree has one initial plus six solver-branch calls per question, and dev/test have six judge calls per question. These offline data-generation costs are separate from the per-policy inference costs in the table.', '', '## Response quality']
    for split in ('train', 'dev', 'test'):
        d = json.loads((root / f'{split}-diagnostics.json').read_text())
        lines.append(f"- {split}: {d['solver_calls']} solver calls ({d['invalid_solver_calls']} invalid); {d['judge_calls']} judge calls ({d['invalid_judge_calls']} invalid); {d['cached_generation_seconds']:.1f} seconds summed generation time.")
    lines += ['', 'An output with recoverable fields may still be wrong, and numeric grading is not a proof of mathematical equivalence. Review examples and scorer edge cases before research claims. This single-model, single-seed study does not establish transfer to other models or datasets. CoRefine and ARR are not reproduced.', '', 'Per-question outcomes and action paths are in results/per_question.jsonl; full summaries, paired intervals, tuned settings, and solver/judge timings are in results/summary.json. Source data hashes and selection seeds are in manifest.json.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(root / 'report.md')

if __name__ == '__main__': main()
