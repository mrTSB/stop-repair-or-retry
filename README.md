# Stop, Repair, or Retry

A reproducible starting point for the CS238 proposal: generate paired solver trajectories, fit lightweight controllers on training questions, select policy parameters on development questions, and evaluate on held-out questions.

Aggregated benchmark reports are indexed in [`results/benchmark-baselines/README.md`](results/benchmark-baselines/README.md).

The [cached math/science dataset and characterization](results/cached-tree-dataset/README.md) includes 560 complete solver trees, explicit 100/60/400 train/dev/test splits, and five fixed baseline results. It is a standalone dataset contribution, with no new controller claims.

Controller naming follows the project proposal: **Controller A** is the two-state POMDP planner and **Controller B** is the prompted LLM controller. Frozen archives under `results/pomdp-*`, `experiments/pomdp-controller/DESIGN.md`, and code identifiers (for example `llm_controller_A`, `controller_A_archive_sha256`) predate this convention and use the swapped labels; they are left unchanged because they are hash-locked by the freeze records.

## Controller B: LLM controller experiment

The [LLM controller workflow](experiments/llm-controller/README.md) replays the published cached dataset and compares the five fixed baselines. The [full frozen evaluation](results/llm-controller-full-evaluation/README.md) includes all 60 dev and 400 test questions, real Qwen2.5-14B-Instruct decisions on private Modal compute, exact raw evidence, token accounting, and independent offline verification. On test, the LLM controller achieved 44.25% accuracy at 1,616 mean tokens, compared with 52.75% at 833 tokens for repair once. No POMDP was fitted or evaluated. Older benchmark reports above are separate historical artifacts and were not used as controller evidence in this experiment.

## Controller A: two-state POMDP milestone

The [Controller A decision record](experiments/pomdp-controller/DESIGN.md) separates user-selected scope from implementation defaults and deferred final-project work. The milestone uses two correctness states, confidence/validity observations, pooled dynamics, and two-step planning. A separate four-state extension is investigated below; it does not replace the frozen milestone.

The [frozen two-state results](results/pomdp-two-state/README.md) use train-only fitting, full dev checks, and cached test evaluation with no new inference spending. At the primary penalty, the model selects repair twice on every question and exactly matches that baseline (54.75% test accuracy, 1,358 mean tokens). This is not an adaptive-controller advantage. No one-step ablation was added.

A [fine train/dev lambda diagnostic](results/pomdp-lambda-diagnostic/report.md) finds eight policies: near lambda 0.248 the POMDP reduces dev tokens by 34.5% while keeping 53.33% accuracy, but remains below repair once. This follow-up does not change the frozen primary experiment or re-evaluate test.

A [separate frozen lambda 0.248 test experiment](results/pomdp-lambda-0248/README.md) reports 53.00% accuracy at 913 mean tokens, compared with Controller B at 44.25%/1,616 and repair once at 52.75%/833. It preserves the original primary experiment and documents dev selection and prior test exposure.

A [known-context train/dev diagnostic](results/pomdp-context-diagnostic/report.md) investigates further improvement without changing frozen results. It highlights diminishing repair returns and a limitation in smoothing unseen confidence categories; the candidate has not been run on test.

A [question-level calibration cross-validation diagnostic](results/pomdp-calibration-cv/report.md) tests unsupported-observation handling without new features or test evaluation. It reproduces the repair-twice dev result and documents why calibration alone has not improved accuracy.

A [four-state latent-model diagnostic](results/pomdp-latent-state-cv/report.md) compares extra hidden states with a matched two-state control on question-grouped training folds. It finds no accuracy improvement, retains the existing observations, and adds no new test result.

## Hardware and credentials

On an Apple Silicon Mac, local inference uses [MLX LM](https://github.com/ml-explore/mlx-lm) and Metal. **No API key, cloud GPU, or paid API is required** for a locally available, openly accessible MLX-compatible model. A first download normally needs internet; gated models may require a Hugging Face account/token and acceptance of the model license. Choose a model that fits unified memory along with its context/KV cache and operating system. Quantized models reduce memory use but can change model behavior. Full experiments can take considerable time: each question has seven solver calls (one initial and six branch calls), plus six judge calls if `--judge` is enabled. The original solver-generation workflow below uses local backends; the separate prepared LLM-controller workflow above supports a configurable hosted endpoint.

The pure Python evaluation and tests run on CPU without MLX or model weights. Python package management and execution use `uv`. A local Ollama server is also supported with `--backend ollama --model qwen3.5:4b`; Ollama's installed models are separate from MLX's model cache. The Ollama backend needs access to the local server at `127.0.0.1:11434` and requests JSON mode and disabled model thinking.

## Setup

```sh
uv sync --extra mlx --extra data
uv run --no-sync srr --help
```

For offline evaluation on a machine without MLX, use `PYTHONPATH=src uv run --no-project --offline python -m stop_repair_retry.cli ...`.

## Data and generation

Prepare **disjoint** train, development, and test JSONL files before generating. Each line is a question with a stable, unique string ID and a short reference answer:

```json
{"id":"train-001","question":"What is 2 + 2?","answer":"4"}
```

A tiny smoke dataset is in `data/`; it is synthetic and **not** suitable for research results. Replace it with appropriately licensed data and a validated answer evaluator.

For GSM8K, download or locate the official `openai/gsm8k` main train and test parquet files, then prepare disjoint reproducible subsets:

```sh
uv run --no-sync python -m stop_repair_retry.gsm8k --train-parquet /path/to/train.parquet --test-parquet /path/to/test.parquet --output runs/gsm8k --train-size 500 --dev-size 200 --test-size 500 --seed 42
```

The command uses separate subsets of the official training split for controller training and development and samples only the official test split for testing. It records source file hashes, seed, and sizes in `manifest.json`. GSM8K's numerical gold answers are extracted after `####`; the scorer supports exact integer, decimal, and simple fraction equivalence, but does not guess a number from prose. Audit extraction and grading on a sample before using results.

```sh
uv run --no-sync srr generate --input data/train.jsonl --output runs/train.jsonl --model mlx-community/Qwen2.5-3B-Instruct-4bit --seed 1
uv run --no-sync srr generate --input data/dev.jsonl --output runs/dev.jsonl --model mlx-community/Qwen2.5-3B-Instruct-4bit --seed 2
uv run --no-sync srr generate --input data/test.jsonl --output runs/test.jsonl --model mlx-community/Qwen2.5-3B-Instruct-4bit --seed 3
uv run --no-sync srr evaluate --train runs/train.jsonl --dev runs/dev.jsonl --test runs/test.jsonl --output runs/results --penalty 0.01
```

Select a model compatible with your installed `mlx-lm` version and hardware. `generate` saves complete questions atomically. To continue an interrupted job, repeat the identical command with `--resume`; it checks the question order and generation configuration before proceeding. Each question receives a deterministic seed derived from its ID and the run seed. Existing output cannot be overwritten accidentally. A retry uses the original prompt with a new stochastic generation; repair sees the immediate previous solution. Use a positive temperature for independent samples. All methods replay the same cached branches for paired comparisons. To study generation variability, produce additional complete datasets using independent seeds and analyze them as repeated samples, clustered by question.

The local pilot stored the model under this project's `.cache/huggingface`; prefix generation commands with `HF_HOME="$PWD/.cache/huggingface"` to reuse that copy. On restricted environments, the local GPU or local Ollama server may require sandbox access. Loading model weights happens outside the measured per-call latencies.

## Science benchmark

The hard science baseline uses the official [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) release, filtered to biology, chemistry, and physics. The converter samples disjoint controller-train, development, and held-out test question IDs (default 150/60/300, balanced across the three subjects); it excludes the dataset's reference chain-of-thought from every model prompt. Since the official release provides a large test split but only 70 validation items, these are study-specific partitions of its public test questions, not an official MMLU-Pro leaderboard score. The manifest records the source hash, filtering, and counts.

Download the official Parquet files once, then prepare the study splits and run the local one-checkpoint stop/repair/retry comparison:

```sh
uv run --no-project --python 3.10 --with huggingface_hub hf download TIGER-Lab/MMLU-Pro --repo-type dataset data/test-00000-of-00001.parquet --local-dir .cache/mmlu-pro
uv run --offline --no-sync python scripts/prepare_mmlu_pro_science.py
UV_CACHE_DIR=/tmp/codex-uv-cache bash scripts/run_mmlu_pro_science.sh
```

The runner uses an already running local Ollama model (`qwen3.5:9b` by default), no API key, and saves trajectories under ignored `runs/mmlu-pro-science/`. Set `MODEL` to another local Ollama model and pass an alternate run directory to reproduce with another model. MMLU-Pro is a knowledge benchmark, so it complements the interactive agent benchmarks rather than replacing them.

The complete local large GSM8K workflow uses `scripts/run_full_gsm8k_large.sh`. It resumes the main run, evaluates all implemented controllers, runs the training-size and repeat-seed checks, and packages results. Use `scripts/run_gsm8k_large.sh` to resume only generation and the main evaluation, or `scripts/finish_gsm8k_large.sh` to rerun postprocessing after a completed main run. The test split excludes questions inspected in the pilot. Preserve the data and model snapshot paths and the generator source unchanged when resuming.

The published aggregate bundle is in `results/gsm8k-large/`. It includes the report, summaries, paired comparisons, diagnostics, data hashes, and frozen code hashes. The full local inputs, references, trajectories, logs, and per-question outputs stay under ignored `runs/`. Rebuild the bundle after a completed run with `uv run --no-sync python scripts/package_results.py`.

Optional local judge: add `--judge --penalty 0.01` to **dev and test** generation (and optionally train), then include `judge_full,judge_compact,hybrid` in `--policies`. Judge calls use the same model as the solver, not a separately loaded model. Full judges see question and history; compact judges see confidence, calls used, previous action, and whether the answer changed. Each judge sees the requested penalty and its actual generated tokens and latency are charged even when it stops. To change the penalty, regenerate judge calls; they are conditioned on it. A local judge is not automatically a strong or accurate decision maker.

The model is asked to emit JSON with `reasoning`, `answer`, and `confidence` (`low`, `medium`, `high`). A fallback recovers explicitly named scalar fields from some malformed JSON; responses without a recoverable answer and confidence are scored wrong, and malformed judge responses fall back to stop. Raw text and `valid` flags are retained for auditing. Numeric answers use exact rational matching, allow a percent sign as in GSM8K percentage answers, simple arithmetic expressions, and up to four trailing unit words; the scorer does **not** recognize general mathematical equivalence or convert units. Validate the scorer for each benchmark before drawing conclusions. No test-time controller receives gold answers.

## Controllers and evaluation

- `stop`, `repair`, `retry`: fixed actions until stopped or budget exhausted.
- `schedule`: best two-action fixed schedule selected by development utility.
- `threshold`: development-tuned confidence threshold and repair/retry choice.
- `one_step`: the learned two-state model with one-step lookahead.
- `pomdp`: finite-horizon two-state POMDP, action-specific confidence and transition distributions with Laplace smoothing, and state-conditioned expected token cost.
- `supervised`: smoothed tabular one-step action-value regression using observable compact history features and paired labels. It is a small supervised comparator, not a neural model or full-horizon Q-learning.
- `judge_full`, `judge_compact`: model-selected action with full or compact information.
- `hybrid`: development-tuned gap gate followed by the full judge, charging its cost. This gate is a heuristic, not a proven value-of-information policy.
- `majority`: retain three independent initial/retry/retry candidates and majority vote; ties favor the first candidate. It always uses all three calls and is a distinct output-selection protocol.

The output includes per-question trajectories and a summary with end-to-end input/output tokens, solver and judge latency, action results, mean utility, question-bootstrap accuracy intervals, and paired accuracy differences versus stop. All accuracy-cost points are for the requested penalty and chosen policies; run a grid of penalties (regenerating judge data when needed) for curves. Token counts are not FLOPs or dollar cost, and latencies are cached single-run wall times, not concurrent-serving benchmarks. Training and data-generation costs are not included in per-question inference cost and must be reported separately. Inspect action-specific transition rates and calibration separately for research use. The stored trajectories can support those diagnostics.

The closest published CoRefine method is **not implemented or claimed as reproduced**: it requires its own full-trace confidence features, trained controller, and prompts. A direct comparison requires the authors' implementation or a faithful independently validated reproduction. This package also does not implement the separate fixed-point/replay method ARR.

## Tests

```sh
PYTHONPATH=src uv run --no-project --offline python -m unittest discover -s tests -v
```

Use `uv run --no-sync python -m stop_repair_retry.diagnose runs/gsm8k/train-trajectories.jsonl` to inspect confidence bins, repair/retry outcome transitions, and malformed outputs. The pilot samples are too small for a publication claim; preregister meaningful sample sizes and repeat seeds for the full study.

Run the unit suite with `uv run --no-sync --offline python -m unittest discover -s tests -v`.

If an audited parser improvement requires reprocessing an existing cache, preserve the original and write a separate copy with `uv run --no-sync python -m stop_repair_retry.reparse INPUT.jsonl OUTPUT.jsonl`. Do not tune parsing against held-out test answers. Use the same frozen parser across every split.
