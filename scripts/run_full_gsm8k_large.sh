#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${1:-runs/gsm8k-large}"
scripts/run_gsm8k_large.sh "$DATA_DIR"
scripts/finish_gsm8k_large.sh "$DATA_DIR"
