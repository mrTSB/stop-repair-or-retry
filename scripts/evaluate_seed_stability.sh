#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${1:-runs/gsm8k-large}"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/codex-uv-cache}"
MODEL="$HF_HOME/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit/snapshots/4f83f8f146fdf28b512a06562b671d7af4fab457"
head -n 50 "$DATA_DIR/test.jsonl" > "$DATA_DIR/test-first-50.jsonl"
uv run --no-sync --offline srr generate --model "$MODEL" --input "$DATA_DIR/test-first-50.jsonl" --output "$DATA_DIR/test-seed-104.jsonl" --max-tokens 384 --seed 104 --judge --resume
uv run --no-sync --offline srr evaluate --train "$DATA_DIR/train-trajectories.jsonl" --dev "$DATA_DIR/dev-trajectories.jsonl" --test "$DATA_DIR/test-seed-104.jsonl" --output "$DATA_DIR/results-seed-104" --penalty 0.01 --policies stop,repair,retry,schedule,threshold,one_step,supervised,pomdp,judge_full,judge_compact,hybrid,majority > "$DATA_DIR/evaluation-seed-104.log"
