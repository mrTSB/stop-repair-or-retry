#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${1:-runs/gsm8k-large}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/codex-uv-cache}"

for file in "$DATA_DIR/results/summary.json" "$DATA_DIR/train-diagnostics.json" "$DATA_DIR/dev-diagnostics.json" "$DATA_DIR/test-diagnostics.json"; do
    if [ ! -f "$file" ]; then
        printf 'Missing completed-run artifact: %s\n' "$file" >&2
        exit 1
    fi
done

scripts/evaluate_training_sizes.sh "$DATA_DIR"
uv run --no-sync --offline python -m stop_repair_retry.compare "$DATA_DIR/results" --baseline repair
uv run --no-sync --offline python -m stop_repair_retry.compare "$DATA_DIR/results" --baseline one_step
uv run --no-sync --offline python -m stop_repair_retry.report "$DATA_DIR"
scripts/evaluate_seed_stability.sh "$DATA_DIR"
uv run --no-sync --offline python -m stop_repair_retry.compare_seeds "$DATA_DIR/results" "$DATA_DIR/results-seed-104"
uv run --no-sync --offline python scripts/package_results.py --run-dir "$DATA_DIR"
printf 'Final results bundle: results/gsm8k-large/report.md\n'
