#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/codex-uv-cache}"
UV="${UV:-uv}"
MODEL="${MODEL:-qwen3.5:9b}"
ROOT="${1:-runs/mmlu-pro-science/qwen35-9b-one-step}"

"$UV" run --offline --no-sync python scripts/prepare_mmlu_pro_science.py
for split in train dev test; do
  args=(generate --input ".cache/mmlu-pro/prepared/$split.jsonl"
    --output "$ROOT/$split-trajectories.jsonl" --backend ollama --model "$MODEL"
    --max-tokens 256 --temperature 0.2 --depth 1 --seed 20261102 --resume)
  "$UV" run --offline --no-sync python -m stop_repair_retry.cli "${args[@]}"
done
"$UV" run --offline --no-sync python -m stop_repair_retry.cli evaluate \
  --train "$ROOT/train-trajectories.jsonl" \
  --dev "$ROOT/dev-trajectories.jsonl" \
  --test "$ROOT/test-trajectories.jsonl" \
  --output "$ROOT/evaluation" \
  --policies stop,repair,retry,schedule,threshold,one_step,supervised,pomdp
"$UV" run --offline --no-sync python scripts/package_mmlu_pro_science.py --run-dir "$ROOT"
printf 'MMLU-Pro science evaluation written to %s\n' "$ROOT/evaluation/summary.json"
