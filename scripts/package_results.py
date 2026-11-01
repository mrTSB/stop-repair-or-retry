#!/usr/bin/env python3
"""Collect the final GSM8K aggregate artifacts into a shareable results bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_json(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_outcomes(source: Path, destination: Path) -> None:
    """Keep paired metrics and actions while omitting free-form model answers."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as input_file, destination.open("w") as output_file:
        for line in input_file:
            row = json.loads(line)
            row.pop("answer", None)
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")


def package(run_dir: Path, destination: Path) -> None:
    required = [
        run_dir / "manifest.json",
        run_dir / "methodology.json",
        run_dir / "report.md",
        run_dir / "results" / "summary.json",
        run_dir / "results" / "paired-vs-repair.json",
        run_dir / "results" / "paired-vs-one_step.json",
        run_dir / "results" / "per_question.jsonl",
        run_dir / "results-seed-104" / "summary.json",
        run_dir / "results-seed-104" / "paired-seed-comparison.json",
        run_dir / "results-seed-104" / "per_question.jsonl",
    ]
    required.extend(run_dir / f"{split}-diagnostics.json" for split in ("train", "dev", "test"))
    required.extend(run_dir / f"results-{penalty}" / "summary.json" for penalty in ("0.001", "0.003", "0.03", "0.1", "0.3"))
    required.extend(run_dir / f"results-train-{size}" / "summary.json" for size in (50, 150))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed-run artifacts:\n" + "\n".join(missing))

    destination.mkdir(parents=True, exist_ok=True)
    copy_json(run_dir / "manifest.json", destination / "manifest.json")
    copy_json(run_dir / "methodology.json", destination / "methodology.json")
    copy_json(run_dir / "results" / "summary.json", destination / "primary" / "summary.json")
    copy_json(run_dir / "results" / "paired-vs-repair.json", destination / "primary" / "paired-vs-repair.json")
    copy_json(run_dir / "results" / "paired-vs-one_step.json", destination / "primary" / "paired-vs-one_step.json")
    copy_outcomes(run_dir / "results" / "per_question.jsonl", destination / "primary" / "per_question.jsonl")

    for penalty in ("0.001", "0.003", "0.03", "0.1", "0.3"):
        copy_json(
            run_dir / f"results-{penalty}" / "summary.json",
            destination / "penalty-sweep" / f"penalty-{penalty}.json",
        )
    for size in (50, 150):
        copy_json(
            run_dir / f"results-train-{size}" / "summary.json",
            destination / "training-size" / f"train-{size}.json",
        )
    copy_json(
        run_dir / "results-seed-104" / "summary.json",
        destination / "repeat-seed" / "summary.json",
    )
    copy_json(
        run_dir / "results-seed-104" / "paired-seed-comparison.json",
        destination / "repeat-seed" / "paired-seed-comparison.json",
    )
    copy_outcomes(
        run_dir / "results-seed-104" / "per_question.jsonl",
        destination / "repeat-seed" / "per_question.jsonl",
    )
    for split in ("train", "dev", "test"):
        copy_json(run_dir / f"{split}-diagnostics.json", destination / "diagnostics" / f"{split}.json")

    report = (run_dir / "report.md").read_text()
    report = report.replace(
        "Per-question outcomes and action paths are in results/per_question.jsonl; full summaries, paired intervals, tuned settings, and solver/judge timings are in results/summary.json. Source data hashes and selection seeds are in manifest.json.",
        "Aggregate summaries, paired comparisons, cost sweeps, training-size checks, repeat-seed analysis, and response diagnostics are included beside this report. Per-question outputs and raw trajectories remain in the local run directory and are not part of this aggregate bundle. Source hashes and selection seeds are in manifest.json and methodology.json.",
    )
    (destination / "report.md").write_text(report)

    (destination / "README.md").write_text(
        "# GSM8K experiment results\n\n"
        "This folder contains results for the frozen local MLX experiment. "
        "Per-question files contain correctness, actions, and cost metrics, with free-form model answers removed. "
        "GSM8K question text, references, and raw generations are excluded.\n\n"
        "- `report.md`: primary results and study limitations.\n"
        "- `manifest.json`, `methodology.json`: split hashes, seeds, and frozen code hashes.\n"
        "- `primary/`: primary summary, paired comparisons, and sanitized per-question outcomes.\n"
        "- `penalty-sweep/`: aggregate summaries at five additional token penalties.\n"
        "- `training-size/`: aggregate summaries using 50 and 150 training questions.\n"
        "- `repeat-seed/`: the 50-question repeat-seed summary, paired comparison, and sanitized outcomes.\n"
        "- `diagnostics/`: aggregate response-format and confidence diagnostics by split.\n\n"
        "Rebuild this folder from local run artifacts with `uv run --no-sync python scripts/package_results.py`.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/gsm8k-large"))
    parser.add_argument("--output", type=Path, default=Path("results/gsm8k-large"))
    args = parser.parse_args()
    package(args.run_dir, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
