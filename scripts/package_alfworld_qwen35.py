#!/usr/bin/env python3
"""Package fixed-policy ALFWorld results from a local Qwen3.5 9B run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from stop_repair_retry.cli import bootstrap_interval


BRANCHES = ("", "repair", "retry")
LABELS = {"": "stop", "repair": "repair", "retry": "retry"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def result_for(row: dict, branch: str) -> dict:
    root = row["tree"][""]
    node = root if not branch else row["tree"][branch]
    total = {key: int(root.get(key, 0)) for key in ("input_tokens", "output_tokens", "env_steps", "invalid_actions")}
    if branch:
        for key in total:
            total[key] += int(node.get(key, 0))
    return {
        "id": row["id"],
        "task_family": row.get("task_family", row.get("task", "unknown")),
        "success": int(float(node.get("score", 0)) >= 1.0),
        "initial_success": int(float(root.get("score", 0)) >= 1.0),
        "retry_reset_match": bool(row.get("retry_reset_match", True)),
        "tokens": total["input_tokens"] + total["output_tokens"],
        **total,
    }


def summarize(rows: list[dict], seed: int) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    results = {branch: [result_for(row, branch) for row in rows] for branch in BRANCHES}
    summary = {}
    for branch, values in results.items():
        key = LABELS[branch]
        successes = [value["success"] for value in values]
        summary[key] = {
            "n": len(values),
            "successes": sum(successes),
            "success_rate": statistics.mean(successes),
            "success_rate_bootstrap_95": bootstrap_interval(successes, seed=seed),
            "mean_tokens": statistics.mean(value["tokens"] for value in values),
            "mean_actions": statistics.mean(value["env_steps"] for value in values),
            "mean_invalid_actions": statistics.mean(value["invalid_actions"] for value in values),
        }
        if branch:
            diffs = [value["success"] - baseline["success"] for value, baseline in zip(values, results[""])]
            summary[key]["paired_difference_vs_stop"] = statistics.mean(diffs)
            summary[key]["paired_difference_bootstrap_95"] = bootstrap_interval(diffs, seed=seed)
            summary[key]["repair_or_retry_wins"] = sum(diff == 1 for diff in diffs)
            summary[key]["repair_or_retry_losses"] = sum(diff == -1 for diff in diffs)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=Path("runs/alfworld-qwen35-9b-extended/test-trajectories.jsonl"))
    parser.add_argument("--pilot", type=Path, default=Path("runs/alfworld-qwen35-9b-pilot/test-trajectories.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/alfworld-qwen35-9b"))
    args = parser.parse_args()

    rows = read_rows(args.trajectories)
    pilot = read_rows(args.pilot)
    if not rows or not pilot:
        raise ValueError("Both the 40-task extension and 10-task exploratory pilot must be present")
    test_ids = {row["id"] for row in rows}
    pilot_ids = {row["id"] for row in pilot}
    if test_ids & pilot_ids:
        raise ValueError("The exploratory pilot and follow-up test IDs must be disjoint")
    if any(row.get("generation_config", {}).get("model") != "qwen3.5:9b" for row in rows + pilot):
        raise ValueError("Expected the local qwen3.5:9b model")

    results, summary = summarize(rows, seed=20261191)
    pilot_results, pilot_summary = summarize(pilot, seed=20261101)

    family_results: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for branch, values in results.items():
        for value in values:
            family_results[value["task_family"]][LABELS[branch]].append(value["success"])
    family_summary = {
        family: {
            "n": len(by_policy["stop"]),
            **{policy: {"successes": sum(items), "n": len(items)} for policy, items in sorted(by_policy.items())},
        }
        for family, by_policy in sorted(family_results.items())
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_question.jsonl").write_text(
        "".join(json.dumps({"policy": LABELS[branch], **value}) + "\n" for branch, values in results.items() for value in values)
    )
    (args.output / "pilot-per-question.jsonl").write_text(
        "".join(json.dumps({"policy": LABELS[branch], **value}) + "\n" for branch, values in pilot_results.items() for value in values)
    )
    manifest = {
        "benchmark": "ALFWorld",
        "model": "qwen3.5:9b",
        "backend": "Ollama local; model served from Mac GPU",
        "test_n": len(rows),
        "selection_seed": rows[0]["generation_config"]["selection_seed"],
        "generation_config": rows[0]["generation_config"],
        "test_ids_sha256": hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest(),
        "trajectory_sha256": digest(args.trajectories),
        "pilot_n": len(pilot),
        "pilot_selection_seed": pilot[0]["generation_config"]["selection_seed"],
        "pilot_ids_sha256": hashlib.sha256("\n".join(sorted(pilot_ids)).encode()).hexdigest(),
        "pilot_trajectory_sha256": digest(args.pilot),
        "pilot_metrics": pilot_summary,
        "pilot_and_followup_disjoint": True,
        "retry_reset_matches": sum(bool(row.get("retry_reset_match", True)) for row in rows),
        "metrics": summary,
        "task_families": family_summary,
        "interpretation": "Fixed-policy follow-up only. The 10-task pilot is exploratory and reported separately; the 40-task follow-up was selected to be disjoint after the pilot. This is not a train/dev/test adaptive-controller evaluation.",
    }
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# ALFWorld Qwen3.5 9B fixed-policy follow-up",
        "",
        f"This follow-up evaluates {len(rows)} tasks with local Qwen3.5 9B through Ollama. The separate {len(pilot)}-task pilot used a different selection seed; the two sets are disjoint. The pilot was exploratory. The 40-task sample was added after inspecting that pilot and is also exploratory, not preregistered.",
        "",
        f"All {manifest['retry_reset_matches']}/{len(rows)} retries reset to the same task instance and matched the initial observation. Exact success is binary ALFWorld completion. The fixed checkpoint is {rows[0]['generation_config']['checkpoint_steps']} actions into a {rows[0]['generation_config']['episode_steps']}-action attempt.",
        "",
        "| Policy | Exact success (95% bootstrap CI) | Paired change vs stop (95% bootstrap CI) | Mean actions | Mean tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in ("stop", "repair", "retry"):
        metric = summary[policy]
        lo, hi = metric["success_rate_bootstrap_95"]
        paired = "—"
        if policy != "stop":
            change = metric["paired_difference_vs_stop"]
            plo, phi = metric["paired_difference_bootstrap_95"]
            paired = (
                f"{100*change:+.1f} pp [{100*plo:+.1f}, {100*phi:+.1f}] "
                f"(wins/losses {metric['repair_or_retry_wins']}/{metric['repair_or_retry_losses']})"
            )
        lines.append(
            f"| {policy} | {100*metric['success_rate']:.1f}% [{100*lo:.1f}, {100*hi:.1f}] "
            f"({metric['successes']}/{metric['n']}) | {paired} | {metric['mean_actions']:.1f} | {metric['mean_tokens']:.0f} |"
        )
    lines.extend(["", "## Exploratory 10-task pilot", "", "This pilot is separate from the follow-up and is too small for stable estimates.", "", "| Policy | Exact success (95% bootstrap CI) | Paired change vs stop | Mean actions |", "|---|---:|---:|---:|"])
    for policy in ("stop", "repair", "retry"):
        metric = pilot_summary[policy]
        lo, hi = metric["success_rate_bootstrap_95"]
        paired = "—" if policy == "stop" else f"{100*metric['paired_difference_vs_stop']:+.1f} pp"
        lines.append(f"| {policy} | {100*metric['success_rate']:.1f}% [{100*lo:.1f}, {100*hi:.1f}] ({metric['successes']}/{metric['n']}) | {paired} | {metric['mean_actions']:.1f} |")
    lines.extend(["", "## Success by ALFWorld task family", "", "| Family | N | Stop | Repair | Retry |", "|---|---:|---:|---:|---:|"])
    for family, values in family_summary.items():
        lines.append(f"| {family} | {values['n']} | {values['stop']['successes']}/{values['n']} | {values['repair']['successes']}/{values['n']} | {values['retry']['successes']}/{values['n']} |")
    lines.extend([
        "",
        "## Limits",
        "",
        "This follow-up contains 40 test tasks and evaluates fixed stop, repair, and retry branches only; it does not include controller train/dev splits or support claims about adaptive policies. The sample was chosen after a separate exploratory pilot, so treat these results as promising but exploratory. The underlying ALFWorld test set and task distribution are public.",
        "",
        "The earlier full 3B study used a different 134-task selection and had low absolute completion. These 9B results should not be pooled with it: model, sample, and backend differ. See `summary.json` and `per_question.jsonl` for hashes, costs, per-task outcomes, and paired intervals. Raw trajectories remain under ignored `runs/`.",
        "",
    ])
    (args.output / "report.md").write_text("\n".join(lines))
    (args.output / "README.md").write_text(
        "Aggregate fixed-policy ALFWorld follow-up for local Qwen3.5 9B. See `report.md` for methods and limitations; `summary.json` and `per_question.jsonl` contain compact outcomes without prompts, observations, or generations. Raw trajectories remain in ignored `runs/`.\n"
    )
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
