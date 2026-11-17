# Learned retry gate for the compact solver

The principal run keeps the exact Qwen3.5-4B weights, compact prompt, 128-token generation limit, and one-token science or expression math retry prompt used in the earlier experiment. The baseline buys the short retry whenever the original compact parser returns an empty answer. The gate learns how to allocate a fraction of that retry budget across a batch and when to retain the compact answer after observing the retry.

The two learned decisions use paired outcomes. The answer comparator fits the accuracy difference between observed candidates. The query model predicts the gain from a retry evaluated by an out-of-fold comparator. Candidate features include answer validity, exact arithmetic checks, question subject, trace length, and uncertainty language. Predictions about an unvisited answer use the observed first trace. The allocator charges every executed input and output token. The initial call is mandatory for `compact_first`. The `routed_first` ablation learns whether to start with the existing short format using question features.

The fit uses 100 previous training questions plus 180 newly sampled question pairs. Five question-level folds guide selection with the 60 previous development questions. The configuration, fits, and source hashes are frozen before scoring the previous 400-question and new disjoint 180-question test sets. Both new splits are balanced between GSM8K and MMLU-Pro science. The sampling manifest records source hashes, excludes previously used IDs and normalized stems, and includes the exact question files. [Results](../../../results/controller-hillclimb/retry-gate/README.md)

## Replay

The archived data are sufficient for an offline replay with Python 3.12 and NumPy 2.5.4. Use a fresh output directory. The published comparison reads the Qwen3.5 responses from the earlier hillclimb archive.

```sh
data=results/controller-hillclimb/retry-gate/data
out=runs/retry-gate-reproduction
for phase in tune published fresh; do
  uv run --python 3.12 --with numpy==2.5.4 --no-project python \
    experiments/controller-hillclimb/retry-gate/evaluate.py "$phase" --data "$data" --output "$out"
done
```

The development-only token-confidence pilot compares the same fitting code on a matched 160-question training sample and 60 development questions. It captures probabilities already emitted by the compact decoding pass. Its replay uses the archived probabilities and runs entirely on CPU.

```sh
uv run --python 3.12 --with numpy==2.5.4 --no-project python \
  experiments/controller-hillclimb/retry-gate/token-confidence/pilot.py \
  --data "$data" --signals "$data/pilots/confidence" --output "$out/confidence.json"
```
