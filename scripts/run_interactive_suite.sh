#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/codex-uv-cache}"
export HF_HOME="$PWD/.cache/huggingface"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$PWD/.cache/alfworld}"
export HF_HUB_OFFLINE=1
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21}"
export PATH="$JAVA_HOME/bin:$PATH"
UV="${UV:-uv}"
ROOT="${1:-runs/interactive}"
TRAIN_SIZE="${TRAIN_SIZE:-100}"
DEV_SIZE="${DEV_SIZE:-50}"
SCIENCEWORLD_TEST_SIZE="${SCIENCEWORLD_TEST_SIZE:-500}"
ALFWORLD_TEST_SIZE="${ALFWORLD_TEST_SIZE:-500}"
SCIENCEWORLD_EPISODE_STEPS="${SCIENCEWORLD_EPISODE_STEPS:-30}"
ALFWORLD_EPISODE_STEPS="${ALFWORLD_EPISODE_STEPS:-50}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-6}"
ALFWORLD_DATA="${ALFWORLD_DATA:-$PWD/.cache/alfworld}"
BENCHMARKS="${BENCHMARKS:-alfworld}"
MODEL="${MODEL:-$HF_HOME/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit/snapshots/4f83f8f146fdf28b512a06562b671d7af4fab457}"

for benchmark in $BENCHMARKS; do
  for split in train dev test; do
    output="$ROOT/$benchmark/$split-trajectories.jsonl"
    size="$TRAIN_SIZE"
    episode_steps="$SCIENCEWORLD_EPISODE_STEPS"
    if [[ "$split" == dev ]]; then size="$DEV_SIZE"; fi
    if [[ "$split" == test && "$benchmark" == scienceworld ]]; then size="$SCIENCEWORLD_TEST_SIZE"; fi
    if [[ "$split" == test && "$benchmark" == alfworld ]]; then size="$ALFWORLD_TEST_SIZE"; fi
    if [[ "$benchmark" == alfworld ]]; then episode_steps="$ALFWORLD_EPISODE_STEPS"; fi
    args=(--benchmark "$benchmark" --split "$split" --model "$MODEL"
      --size "$size" --episode-steps "$episode_steps"
      --checkpoint-steps "$CHECKPOINT_STEPS"
      --output "$output" --alfworld-data "$ALFWORLD_DATA" --resume)
    if [[ "$split" != train ]]; then args+=(--judge); fi
    "$UV" run --offline --no-sync python -m stop_repair_retry.interactive generate "${args[@]}"
  done
  "$UV" run --offline --no-sync python -m stop_repair_retry.interactive evaluate \
    --train "$ROOT/$benchmark/train-trajectories.jsonl" \
    --dev "$ROOT/$benchmark/dev-trajectories.jsonl" \
    --test "$ROOT/$benchmark/test-trajectories.jsonl" \
    --output "$ROOT/$benchmark/evaluation"
done

"$UV" run --offline --no-sync python scripts/package_interactive_results.py --run-dir "$ROOT" --output "${RESULTS_DIR:-results/interactive}" --benchmarks $BENCHMARKS

printf 'Finished interactive benchmark suite under %s\n' "$ROOT"
