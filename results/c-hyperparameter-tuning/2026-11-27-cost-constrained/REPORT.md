# Frozen-architecture tuning: 2026-11-27 cost-constrained search

## Finding

| Policy | Correct / 256 | Accuracy | Mean compute proxy |
| --- | ---: | ---: | ---: |
| Frozen A | 153 | 59.765625% | 1087.2748356677234 |
| Frozen B | 179 | 69.921875% | 3858.2777103559874 |
| Frozen C | 177 | 69.140625% | 3728.5754404183053 |
| Tuned C | **178** | **69.53125%** | **3330.645424500263** |

Tuned C gains one correct answer (+0.390625 percentage points) and uses 10.67244% less compute than original C. It remains one correct answer below B while using 13.67533% less compute. Relative to original C, one question becomes correct and none becomes incorrect; 52 questions change their path or compute bill. This is an exploratory selection result on the same repeatedly reused development panel.

## Exact changes

| Existing subject expert | Compute penalty | Review threshold | Existing ensemble prior |
| --- | ---: | ---: | --- |
| Math (`smooth32-review1`) | 0.005 | 0 | unchanged |
| Chemistry (`smooth8-review1`) | 0.0005 | 0 | unchanged |
| Physics (existing two-parent Bayesian ensemble) | 0.01 | 0.06 | 0.1 / 0.9, preserving parent order |

No trained weights, tokenizer, observation features, model classes, dynamics arrays, parent models, routing structure, maximum three solver calls, maximum one logical root review, or billing formulas changed. The serialized-model audit reverses only these settings/prior weights and recovers a byte-identical serialization of the original model. A, B and all five baselines are unchanged reference results from the independently verified replay.

## Search and selection

The source is [`cost_constrained.py`](../../../experiments/c-hyperparameter-tuning/cost_constrained.py). [Registration](registration.json) records its SHA-256, original model SHA-256, objective and complete grids before scoring.

- Penalties: `0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1`.
- Thresholds: `0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.2, 0.25, 0.3`.
- Physics first-parent prior weights: `0.1, 0.2, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9`; second weight is the complement.
- 180 math settings, 180 chemistry settings and 1,440 physics settings, plus the original setting for each subject: 1,803 subject candidates total. These are not 1,803 independent whole-panel experiments.
- Select the most correct answers under original C's mean compute ceiling; break ties by lower compute. Subject composition uses dynamic programming and is independently checked by enumerating correctness/cost frontier combinations.
- The alternative ceiling of B's compute selects the same result.

For each penalty/prior, the original runtime is evaluated at threshold zero and a threshold of one million (no review). Threshold does not change either branch's internal planning. Its exact comparison `information_value > threshold + 1e-9` selects between these outcomes for the sweep. The chosen configuration is then run directly through the original router; the independent audit reloads the actual saved pickle in another process.

## Sensitivity and limits

Four subject-stratified folds use RNG seed `20261127`. Each fold selects parameters using the other three development folds, under their original-C mean compute ceiling, then scores the held fold.

| Held fold | Selected policy correct / 64 | Original C correct / 64 |
| --- | ---: | ---: |
| 0 | 41 | 43 |
| 1 | 43 | 43 |
| 2 | 43 | 43 |
| 3 | 44 | 48 |
| Total | **171/256** | **177/256** |

Mean held-fold compute is 3186.6421703394367. This does not support a robust accuracy gain. Historical tuning already reused these development questions, so these folds are a selection-sensitivity diagnostic, **not fresh cross-validation or independent confirmation**. The reference itself was selected on the whole panel. Do not characterize either this diagnostic or the full-panel 178 as reserved-test performance. No significance or generalization claim is made.

## Verification

[Independent audit](audit.json): 256 direct saved-checkpoint replays and 256 label-blinded paid-observation replays passed. Per-question correctness, solver/reviewer tokens, unique observer component charges, total compute and action budgets were checked independently of selection. Reversing settings and prior weights exactly restores the original serialized model. All 17,904 installed frozen input files (4,864,173,829 bytes) matched their manifest before and after. The earlier complete-archive verification covered all 38,826 files (102,537,523,932 bytes); this run uses the verified local cached-replay subset.

No model forwards, inference credentials, cloud jobs or new paid compute were used. The no-test-read and no-model-call guards stayed active during the experiment and audit. Integrity checks may hash archived labels; that is not model fitting, selection or grading.

## Artifacts and reproduction

[summary.json](summary.json), [per-question.json](per-question.json), [changed-questions.json](changed-questions.json), [candidate-metrics.json](candidate-metrics.json), [candidate-outcomes.json.gz](candidate-outcomes.json.gz), [hybrid.pkl](hybrid.pkl), [audit.json](audit.json), [complete.json](complete.json), logs and input-integrity receipts are all included. Candidate outcomes are compressed losslessly; [packaging.json](packaging.json) records compressed and decompressed hashes. `complete.json` refers to the original uncompressed filenames.

Use the full frozen runtime and exact hashed dependencies described in [the release guide](../../reproduction/README.md). Install Python 3.12.11 and dependencies in a separate environment; do not execute the archived `.venv`. Native macOS arm64 matched the archived strict numerical cache fingerprints. If macOS offloads files in Documents, use a non-offloaded environment and restore archived bytes only from verified archive parts. The run restored 1,175 offloaded files from local archive tars, checking original SHA-256 values.

From the extracted runtime root, copy the experiment sources into a new extension directory:

```bash
mkdir -p runs/extensions
cp /path/to/checkout/experiments/c-hyperparameter-tuning/cost_constrained.py runs/extensions/cost_constrained.py
cp /path/to/checkout/experiments/c-hyperparameter-tuning/audit.py runs/extensions/audit_tuning.py
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
/path/to/pinned-venv/bin/python scripts/reproduction/run.py \
  runs/extensions/cost_constrained.py /absolute/path/to/new-output
/path/to/pinned-venv/bin/python scripts/reproduction/run.py \
  runs/extensions/audit_tuning.py /absolute/path/to/new-output
```

Always use a new output directory. To audit the published artifacts instead, copy this result directory elsewhere and decompress `candidate-outcomes.json.gz` to `candidate-outcomes.json` there before invoking the audit. Do not rewrite the frozen release's source or inputs.
