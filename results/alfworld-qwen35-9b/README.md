Aggregate fixed-policy ALFWorld follow-up for local Qwen3.5 9B. See `report.md` for methods and limitations; `summary.json` and `per_question.jsonl` contain compact outcomes without prompts, observations, or generations. Raw trajectories remain in ignored `runs/`.

Reproduce the 40-task follow-up with the same local setup:

```sh
PYTHONPATH=src UV_CACHE_DIR=/tmp/codex-uv-cache uv run --offline --no-sync python -m stop_repair_retry.interactive generate \
  --benchmark alfworld --split test --size 40 --backend ollama --model qwen3.5:9b \
  --seed 101 --selection-seed 20261191 --checkpoint-steps 6 --episode-steps 50 \
  --max-tokens 128 --temperature 0.7 \
  --output runs/alfworld-qwen35-9b-extended/test-trajectories.jsonl \
  --alfworld-data .cache/alfworld --resume
```

Then package the report with `PYTHONPATH=src uv run --offline --no-sync python scripts/package_alfworld_qwen35.py`.
