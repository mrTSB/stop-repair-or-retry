#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/codex-uv-cache}"
MODEL="$HF_HOME/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit/snapshots/4f83f8f146fdf28b512a06562b671d7af4fab457"
DATA_DIR="${1:-runs/gsm8k-large}"

uv run --no-sync --offline srr generate --model "$MODEL" --input "$DATA_DIR/train.jsonl" --output "$DATA_DIR/train-trajectories.jsonl" --max-tokens 384 --seed 101 --resume
uv run --no-sync --offline srr generate --model "$MODEL" --input "$DATA_DIR/dev.jsonl" --output "$DATA_DIR/dev-trajectories.jsonl" --max-tokens 384 --seed 102 --judge --resume
uv run --no-sync --offline srr generate --model "$MODEL" --input "$DATA_DIR/test.jsonl" --output "$DATA_DIR/test-trajectories.jsonl" --max-tokens 384 --seed 103 --judge --resume
uv run --no-sync --offline srr evaluate --train "$DATA_DIR/train-trajectories.jsonl" --dev "$DATA_DIR/dev-trajectories.jsonl" --test "$DATA_DIR/test-trajectories.jsonl" --output "$DATA_DIR/results" --penalty 0.01 --policies stop,repair,retry,schedule,threshold,one_step,supervised,pomdp,judge_full,judge_compact,hybrid,majority > "$DATA_DIR/evaluation.log"
for penalty in 0.001 0.003 0.03 0.1 0.3; do
    uv run --no-sync --offline srr evaluate --train "$DATA_DIR/train-trajectories.jsonl" --dev "$DATA_DIR/dev-trajectories.jsonl" --test "$DATA_DIR/test-trajectories.jsonl" --output "$DATA_DIR/results-$penalty" --penalty "$penalty" > "$DATA_DIR/evaluation-$penalty.log"
done
for split in train dev test; do
    uv run --no-sync --offline python -m stop_repair_retry.diagnose "$DATA_DIR/$split-trajectories.jsonl" > "$DATA_DIR/$split-diagnostics.json"
done
printf 'Finished: %s\n' "$DATA_DIR/results/summary.json"
