#!/usr/bin/env python3
"""Package interactive benchmark aggregates without observations or prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package(run_dir: Path, output: Path, benchmarks: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels = {"scienceworld": "ScienceWorld", "alfworld": "ALFWorld"}
    report = ["# Interactive benchmark results", "", "Local MLX/Metal results for " + " and ".join(labels[b] for b in benchmarks) + ".", ""]
    manifest = {"benchmarks": {}, "runner_sha256": digest(Path("src/stop_repair_retry/interactive.py"))}
    for benchmark in benchmarks:
        source = run_dir / benchmark
        destination = output / benchmark
        destination.mkdir(parents=True, exist_ok=True)
        split_info = {}
        for split in ("train", "dev", "test"):
            trajectories = source / f"{split}-trajectories.jsonl"
            if not trajectories.exists():
                trajectories = source / f"{split}.jsonl"
            rows = [json.loads(line) for line in trajectories.read_text().splitlines() if line.strip()]
            if not rows:
                raise FileNotFoundError(f"Missing completed trajectories: {trajectories}")
            configs = {json.dumps(row.get("generation_config"), sort_keys=True) for row in rows}
            if len(configs) != 1:
                raise ValueError(f"Mixed generation configs in {trajectories}")
            config = json.loads(next(iter(configs)))
            split_info[split] = {
                "n": len(rows), "sha256": digest(trajectories), "config": config,
                "retry_reset_match_n": sum(bool(row.get("retry_reset_match", True)) for row in rows),
            }
        evaluation = source / "evaluation"
        for name in ("summary.json", "paired-vs-stop.json", "per_question.jsonl"):
            shutil.copyfile(evaluation / name, destination / name)
        manifest["benchmarks"][benchmark] = split_info
        summary = json.loads((evaluation / "summary.json").read_text())
        data = {result["policy"]: result for result in summary["results"]}
        test_rows = [json.loads(line) for line in (source / "test-trajectories.jsonl").read_text().splitlines() if line.strip()]
        family_by_id = {row["id"]: row.get("task_family", row.get("task", "unknown")) for row in test_rows}
        per_question = [json.loads(line) for line in (evaluation / "per_question.jsonl").read_text().splitlines() if line.strip()]
        from collections import defaultdict
        family_results = defaultdict(lambda: defaultdict(list))
        for item in per_question:
            family_results[family_by_id[item["id"]]][item["policy"]].append(item["correct"])
        report.extend([
            f"## {benchmark}",
            "",
            f"Train/dev/test counts: {split_info['train']['n']}/{split_info['dev']['n']}/{split_info['test']['n']}.",
            f"Retry reset matches: {split_info['test']['retry_reset_match_n']}/{split_info['test']['n']}; paired comparisons exclude mismatches.",
            f"Attempt budget: {split_info['test']['config']['episode_steps']} actions; controller checkpoint: {split_info['test']['config']['checkpoint_steps']} actions.",
            "",
            "| Policy | Success (95% CI) | Mean tokens | Mean actions | Mean invalid actions |",
            "|---|---:|---:|---:|---:|",
        ])
        for policy in ("stop", "repair", "retry", "tuned_fixed", "threshold", "one_step", "supervised", "judge_full", "judge_compact", "hybrid"):
            r = data[policy]
            low, high = r["accuracy_ci_95"]
            report.append(f"| {policy} | {r['accuracy']:.3f} [{low:.3f}, {high:.3f}] | {r['mean_tokens']:.1f} | {r['mean_env_steps']:.1f} | {r['mean_invalid_actions']:.2f} |")
        report.extend(["", "Success is exact benchmark completion. For ALFWorld, reward is binary completion (0 or 1).", "", "### Paired difference from stop", "", "| Policy | Success difference (95% CI) |", "|---|---:|"])
        paired = json.loads((evaluation / "paired-vs-stop.json").read_text())
        for item in paired:
            if item["policy"] in ("repair", "retry", "tuned_fixed", "threshold", "one_step", "supervised", "judge_full", "judge_compact", "hybrid"):
                lo, hi = item["accuracy_difference_ci_95"]
                report.append(f"| {item['policy']} | {item['accuracy_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        report.extend(["", "### Success by task family", "", "| Task family | N | Stop | Repair | Retry |", "|---|---:|---:|---:|---:|"])
        for family, policies in sorted(family_results.items()):
            n = len(policies["stop"])
            report.append("| " + family + f" | {n} | {sum(policies['stop'])}/{n} | {sum(policies['repair'])}/{n} | {sum(policies['retry'])}/{n} |")
        report.append("")
    manifest["limitations"] = [
        "Pilot outputs with fewer than 100 test episodes are smoke tests, not publishable estimates.",
        "Results use a small local 3B model and should not be interpreted as benchmark-level agent performance.",
        "Interactive evaluation makes one controller choice per episode at the configured checkpoint; multi-step scheduling and finite-horizon POMDP policies are not reported as distinct methods.",
    ]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report.extend([
        "## Method and limitations", "",
        "At a fixed checkpoint, stop submits current progress, repair continues the live attempt, and retry restarts the same task instance. All controller policies use cached paired branch outcomes. Fitted models use train trajectories; tuned settings use dev; test is held out. Confidence intervals and per-task results are in the JSON artifacts.",
        "", "Per-question artifacts retain task IDs, controller actions, scores, and costs; they exclude free-form observations, prompts, and generations. Raw trajectories remain in the local `runs/` directory.", "",
        "These runs support pipeline validation and feasibility analysis. The local model has limited task success; larger test samples alone will not make this a useful workshop result. Interpret results only for the benchmark(s) listed above.", "",
    ])
    (output / "report.md").write_text("\n".join(report))
    (output / "README.md").write_text(
        "# Interactive benchmark results\n\n"
        "Contains aggregate evaluations for: " + ", ".join(labels[b] for b in benchmarks) + ". `report.md` summarizes metrics and limits; `manifest.json` records hashes and split configurations. Per-question results omit observations and model text. Raw local trajectories remain in the matching `runs/` directory.\n"
    )


def package_scienceworld_gate(trajectories: Path, output: Path) -> None:
    rows = [json.loads(line) for line in trajectories.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"ScienceWorld gate file is empty: {trajectories}")
    destination = output / "scienceworld-feasibility.json"
    branches = {}
    for branch in ("", "repair", "retry"):
        scores = [float(row["tree"][branch]["score"]) for row in rows]
        branches[branch or "stop"] = {
            "exact_successes": sum(score >= 1 for score in scores),
            "n": len(scores),
            "mean_reward": sum(scores) / len(scores),
        }
    from collections import defaultdict
    by_family = defaultdict(list)
    for row in rows:
        by_family[row.get("task_family", row.get("task", "unknown"))].append(row)
    family_scores = {}
    for family, family_rows in sorted(by_family.items()):
        family_scores[family] = {"n": len(family_rows)}
        for branch in ("", "repair", "retry"):
            scores = [float(row["tree"][branch]["score"]) for row in family_rows]
            family_scores[family][branch or "stop"] = {
                "exact_successes": sum(score >= 1 for score in scores),
                "mean_reward": sum(scores) / len(scores),
            }
    summary = {
        "benchmark": "ScienceWorld",
        "n": len(rows),
        "task_ids": [row["id"] for row in rows],
        "task_families": [row.get("task_family", row.get("task", "unknown")) for row in rows],
        "retry_reset_match_n": sum(bool(row.get("retry_reset_match", True)) for row in rows),
        "generation_config": rows[0].get("generation_config"),
        "branches": branches,
        "family_scores": family_scores,
        "trajectory_sha256": digest(trajectories),
        "interpretation": "Feasibility gate only; five tasks and zero exact completions do not support a full baseline comparison.",
    }
    (output / destination.name).write_text(json.dumps(summary, indent=2) + "\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["scienceworld_feasibility_gate"] = {
        "n": len(rows), "trajectory_sha256": summary["trajectory_sha256"],
        "retry_reset_match_n": summary["retry_reset_match_n"],
        "config": summary["generation_config"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report_path = output / "report.md"
    report = report_path.read_text() if report_path.exists() else "# Interactive benchmark results\n"
    report += "\n## ScienceWorld feasibility gate\n\n"
    report += f"Current-code gate: {len(rows)} held-out tasks, 30-action attempt budget, 6-action checkpoint. Exact completions were stop {branches['stop']['exact_successes']}/{len(rows)}, repair {branches['repair']['exact_successes']}/{len(rows)}, and retry {branches['retry']['exact_successes']}/{len(rows)}. Mean normalized progress was {branches['stop']['mean_reward']:.3f}, {branches['repair']['mean_reward']:.3f}, and {branches['retry']['mean_reward']:.3f}, respectively. Retry resets matched the initial observations on {summary['retry_reset_match_n']}/{len(rows)} tasks.\n\n"
    report += "This gate did not pass: no branch completed a task, and reset mismatches make two task pairs unsuitable for paired claims. The full ScienceWorld run was therefore not launched with this 3B agent. See `scienceworld-feasibility.json` for task IDs, family scores, configuration, and the trajectory hash.\n"
    report_path.write_text(report)
    readme = output / "README.md"
    if readme.exists():
        content = readme.read_text()
        if "scienceworld-feasibility.json" not in content:
            readme.write_text(content.rstrip() + " It also includes a five-task ScienceWorld feasibility gate; see `scienceworld-feasibility.json`.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/interactive"))
    parser.add_argument("--output", type=Path, default=Path("results/interactive"))
    parser.add_argument("--benchmarks", nargs="+", choices=("scienceworld", "alfworld"), default=("scienceworld", "alfworld"))
    parser.add_argument("--scienceworld-gate", type=Path)
    args = parser.parse_args()
    package(args.run_dir, args.output, args.benchmarks)
    if args.scienceworld_gate:
        package_scienceworld_gate(args.scienceworld_gate, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
