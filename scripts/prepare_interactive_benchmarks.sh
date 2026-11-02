#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/codex-uv-cache}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$PWD/.cache/alfworld}"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@21}"
export PATH="$JAVA_HOME/bin:$PATH"

uv sync --extra mlx --extra scienceworld --extra alfworld
java -version
if [[ ! -f "$ALFWORLD_DATA/logic/alfred.pddl" ]]; then
  uv run alfworld-download --data "$ALFWORLD_DATA"
fi
printf 'Prepared ScienceWorld and ALFWorld at %s\n' "$ALFWORLD_DATA"
