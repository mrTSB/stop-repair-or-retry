#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/codex-uv-cache}"
DATA_DIR="${1:-runs/gsm8k-large}"
for n in 50 150; do
    head -n "$n" "$DATA_DIR/train-trajectories.jsonl" > "$DATA_DIR/train-first-$n.jsonl"
    uv run --no-sync --offline srr evaluate --train "$DATA_DIR/train-first-$n.jsonl" --dev "$DATA_DIR/dev-trajectories.jsonl" --test "$DATA_DIR/test-trajectories.jsonl" --output "$DATA_DIR/results-train-$n" --penalty 0.01 --policies stop,repair,retry,schedule,threshold,one_step,supervised,pomdp,judge_full,judge_compact,hybrid,majority > "$DATA_DIR/evaluation-train-$n.log"
done
