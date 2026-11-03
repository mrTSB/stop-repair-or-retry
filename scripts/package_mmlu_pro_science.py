#!/usr/bin/env python3
"""Package a compact MMLU-Pro science report without prompts or generations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mmlu-pro-science/qwen35-9b-one-step"))
    parser.add_argument("--prepared-dir", type=Path, default=Path(".cache/mmlu-pro/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/mmlu-pro-science"))
    args = parser.parse_args()

    evaluation_dir = args.run_dir / "evaluation"
    summary = json.loads((evaluation_dir / "summary.json").read_text())
    details = read_jsonl(evaluation_dir / "per_question.jsonl")
    test_rows = read_jsonl(args.run_dir / "test-trajectories.jsonl")
    split_manifest = json.loads((args.prepared_dir / "manifest.json").read_text())
    config = test_rows[0].get("generation_config", {}) if test_rows else {}

    by_policy_category: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in details:
        category = row["id"].split("-")[2]
        by_policy_category[row["policy"]][category].append(int(row["correct"]))
    per_policy_question: dict[str, dict[str, int]] = defaultdict(dict)
    for row in details:
        per_policy_question[row["policy"]][row["id"]] = int(row["correct"])
    repair_tokens = next(result["mean_tokens"] for result in summary["results"] if result["policy"] == "repair")
    for result in summary["results"]:
        policy = result["policy"]
        paired = [
            per_policy_question[policy][question_id] - per_policy_question["repair"][question_id]
            for question_id in per_policy_question["repair"]
        ]
        from stop_repair_retry.cli import bootstrap_interval
        result["paired_accuracy_diff_vs_repair"] = sum(paired) / len(paired)
        result["paired_diff_ci_95_vs_repair"] = bootstrap_interval(paired, seed=20261102)
        result["mean_token_savings_vs_repair"] = (repair_tokens - result["mean_tokens"]) / repair_tokens
    category_accuracy = {
        policy: {category: sum(values) / len(values) for category, values in sorted(categories.items())}
        for policy, categories in by_policy_category.items()
    }
    packaged = {
        "dataset": split_manifest,
        "generation": config,
        "one_decision_per_question": all(int(row.get("max_depth", 2)) == 1 for row in test_rows),
        "evaluation": summary,
        "test_accuracy_by_category": category_accuracy,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(packaged, indent=2) + "\n")
    table = [
        "| Policy | Accuracy (95% CI) | Paired change vs stop (95% CI) | Mean tokens |",
        "|---|---:|---:|---:|",
    ]
    for result in summary["results"]:
        acc_low, acc_high = result["accuracy_ci_95"]
        diff = result.get("paired_accuracy_diff_vs_stop")
        diff_ci = result.get("paired_diff_ci_95")
        if diff is None:
            diff_text = "—"
        else:
            diff_text = f"{100 * diff:+.1f} pp [{100 * diff_ci[0]:+.1f}, {100 * diff_ci[1]:+.1f}]"
        table.append(
            f"| {result['policy']} | {100 * result['accuracy']:.1f}% "
            f"[{100 * acc_low:.1f}, {100 * acc_high:.1f}] | {diff_text} | {result['mean_tokens']:.0f} |"
        )
    category_lines = ["| Policy | Biology | Chemistry | Physics |", "|---|---:|---:|---:|"]
    for policy, categories in sorted(category_accuracy.items()):
        cells = [f"{100 * categories.get(category, float('nan')):.1f}%" for category in ("biology", "chemistry", "physics")]
        category_lines.append(f"| {policy} | " + " | ".join(cells) + " |")

    report = "\n".join([
        "# MMLU-Pro science baseline",
        "",
        f"Model: `{config.get('model', 'unknown')}` via `{config.get('backend', 'unknown')}`. ",
        f"Held-out test size: {len(test_rows)} questions (biology, chemistry, physics).",
        "",
        "This is a local science-subset experiment, not an official MMLU-Pro leaderboard score. "
        "The source release has a large public test split but only 70 validation questions; "
        "we formed disjoint study-specific controller-train/dev/test partitions from the official test file. "
        "The model saw the question and options only; MMLU-Pro reference chain-of-thought was excluded.",
        "",
        "## Stop, repair, and retry results",
        "",
        *table,
        "",
        "## Accuracy by subject",
        "",
        *category_lines,
        "",
        "## Cost-aware controllers vs fixed repair",
        "",
        "| Policy | Paired accuracy change vs repair (95% CI) | Mean token savings vs repair |",
        "|---|---:|---:|",
        *[
            f"| {result['policy']} | {100*result['paired_accuracy_diff_vs_repair']:+.1f} pp "
            f"[{100*result['paired_diff_ci_95_vs_repair'][0]:+.1f}, {100*result['paired_diff_ci_95_vs_repair'][1]:+.1f}] | "
            f"{100*result['mean_token_savings_vs_repair']:.1f}% |"
            for result in summary["results"] if result["policy"] in ("threshold", "supervised", "one_step", "pomdp")
        ],
        "",
        "## Method",
        "",
        f"The one-checkpoint tree generated {config.get('depth', 'unknown')} decision level(s) at temperature "
        f"{config.get('temperature', 'unknown')} with a {config.get('max_tokens', 'unknown')}-token output limit. "
        "Train/development/test questions are disjoint by stable question ID and balanced across the three subjects. "
        "Confidence intervals use the repository's per-question bootstrap; differences are paired against stop.",
        "",
        "The result is conditional on the model, prompt, and public benchmark. It does not measure multimodal "
        "science reasoning, experimental design in an interactive simulator, or general scientific discovery.",
        "",
    ])
    (args.output_dir / "report.md").write_text(report)
    (args.output_dir / "README.md").write_text(
        "Compact MMLU-Pro science aggregates and methodology. Per-question details and full trajectories remain in ignored `runs/`; prompts and generated reasoning are not copied here.\n"
    )
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
