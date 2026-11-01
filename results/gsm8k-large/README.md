# GSM8K experiment results

This folder contains results for the frozen local MLX experiment. Per-question files contain correctness, actions, and cost metrics, with free-form model answers removed. GSM8K question text, references, and raw generations are excluded.

- `report.md`: primary results and study limitations.
- `manifest.json`, `methodology.json`: split hashes, seeds, and frozen code hashes.
- `primary/`: primary summary, paired comparisons, and sanitized per-question outcomes.
- `penalty-sweep/`: aggregate summaries at five additional token penalties.
- `training-size/`: aggregate summaries using 50 and 150 training questions.
- `repeat-seed/`: the 50-question repeat-seed summary, paired comparison, and sanitized outcomes.
- `diagnostics/`: aggregate response-format and confidence diagnostics by split.

Rebuild this folder from local run artifacts with `uv run --no-sync python scripts/package_results.py`.
