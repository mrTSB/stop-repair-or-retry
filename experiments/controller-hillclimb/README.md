# Accuracy and compute hillclimb

The experiment uses the 100 training, 60 development, and 400 test questions in the published [cached tree dataset](../../results/cached-tree-dataset/README.md). Each of the seven solver nodes stays with its question. The runner uses the published scoring labels for the original solver and applies the same answer scorer to new answers.

## Controller variants on the original trees

- **A** fits ridge regressions for paired changes in correctness and solver cost. It compares the remaining stop, repair, and retry plans and replans after each observed answer. Candidate features include validity, reported confidence, answer agreement, response length, and arithmetic consistency.
- **B** adds one independent verifier call. The verifier produces a short arithmetic expression for math or one token of multiple-choice evidence for science. It contributes an agreement signal and token probabilities to the planner. The controller submits a visited answer from the original Qwen2.5 tree.
- **C** applies a development-selected gate to the action-value margin from the planner without verifier evidence. The gate determines whether to acquire and reuse verifier evidence. The search considers domain restrictions and an optional early stop when math answers agree.

Training uses five domain-balanced folds grouped by question. The search prioritizes a positive accuracy margin over both fixed repair baselines and lower compute than repair once on both out-of-fold training predictions and development. If that region is empty, the next priority is a positive accuracy margin and lower compute than repair twice. The frozen run records each selected configuration and its fitted weights before replaying test.

## Compact-first variant

The separate response cache adds a Qwen3.5-4B answer with a short scratch-work prompt and a 128-token ceiling. The controller starts with that answer. It can submit it, request a short independent answer from the same model, or retry with the original Qwen2.5 solver. After a Qwen2.5 retry, it can submit, repair, or retry within the two-extra-call budget. The shorter response records retain the exact model digest, prompt hash, token counts, and raw output. The paired comparison includes compact-only, fixed continuation, and rules that continue only when the compact answer fails to parse.

The compact-first search uses the original training and development question splits. Its first criterion is more correct answers than compact-only and lower compute than the cheapest fixed continuation on both development and out-of-fold training. It freezes the selected gate and regressions before test scoring.

`--short-first` starts with the expression or one-token option prompt and makes the compact scratch-work response available as a retry. It adds the available token-probability features to the planner and searches for a higher-accuracy policy within the original submit-immediately compute budget. Both orders use the same separate response records.

## Compute accounting

Each replay charges every visited solver and verifier call, including their full input and generated output token counts. The compute proxy is `2 × model parameters × input and output tokens`, summed by model and reported in Qwen2.5-3B-equivalent tokens. The denominator uses 3.09B parameters and the verifier/compact models use nominal 2B and 4B weights. This proxy approximates model arithmetic and omits attention, caching, precision, and CPU overhead. It does not measure serving latency or GPU energy. Offline observation collection and training are outside the per-question inference totals. See the [solver model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [2B model card](https://huggingface.co/Qwen/Qwen3.5-2B), [4B model card](https://huggingface.co/Qwen/Qwen3.5-4B), and [Ollama usage fields](https://docs.ollama.com/api/generate).

## Reproduction

From the repository root, unpack the archived observations and run both workflows. All replay stages run on CPU. The reference environment used Python 3.12 or newer and NumPy 2.5.4.

```sh
sig=runs/controller-hillclimb-replay/signals
mkdir -p "$sig"
for artifact in results/controller-hillclimb/data/*.jsonl.gz; do
  name=$(basename "$artifact" .gz)
  gzip -dc "$artifact" > "$sig/$name"
done

uv run --python 3.12 --with numpy==2.5.4 --no-project python experiments/controller-hillclimb/hillclimb.py \
  tune --signals "$sig" --output runs/controller-hillclimb-replay/original
uv run --python 3.12 --with numpy==2.5.4 --no-project python experiments/controller-hillclimb/hillclimb.py \
  test --signals "$sig" --output runs/controller-hillclimb-replay/original
uv run --python 3.12 --with numpy==2.5.4 --no-project python experiments/controller-hillclimb/cascade.py \
  tune --signals "$sig" --output runs/controller-hillclimb-replay/compact
uv run --python 3.12 --with numpy==2.5.4 --no-project python experiments/controller-hillclimb/cascade.py \
  test --signals "$sig" --output runs/controller-hillclimb-replay/compact
uv run --python 3.12 --with numpy==2.5.4 --no-project python experiments/controller-hillclimb/cascade.py \
  tune --short-first --signals "$sig" --output runs/controller-hillclimb-replay/short
uv run --python 3.12 --with numpy==2.5.4 --no-project python experiments/controller-hillclimb/cascade.py \
  test --short-first --signals "$sig" --output runs/controller-hillclimb-replay/short
```

`collect.py` and `compact.py` can regenerate observations with a local Ollama server and the matching weights. Regeneration can vary with the inference backend. The shipped sidecars are the inputs to the reported replay. The [results](../../results/controller-hillclimb/README.md) include paired question bootstrap intervals conditional on these model responses.
