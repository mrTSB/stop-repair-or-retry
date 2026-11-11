# Separate frozen experiment: POMDP at lambda 0.248

The user explicitly selected lambda 0.248 after the train/dev diagnostic. This experiment freezes that choice and the unchanged two-state training model before one replay of all 400 test questions. It does not replace the original lambda0.01 milestone or its predeclared sweep. No additional model fitting, prompt changes, solver generation, or paid inference occurred.

## Test results

| Method | Correct | Accuracy | Mean total tokens | Utility at λ = 0.248 | Reference utility at λ = 0.01 |
|---|---:|---:|---:|---:|---:|
| POMDP, λ = 0.248 | 212/400 | 53.00% | 913.49 | 0.30346 | 0.52087 |
| Submit initial | 145/400 | 36.25% | 340.11 | 0.27815 | 0.35910 |
| Repair once | 211/400 | 52.75% | 832.95 | 0.32093 | 0.51917 |
| Repair twice | 219/400 | 54.75% | 1,358.30 | 0.21064 | 0.53392 |
| Retry once | 153/400 | 38.25% | 679.56 | 0.21397 | 0.37570 |
| Retry twice | 161/400 | 40.25% | 1,020.66 | 0.14938 | 0.39229 |
| Controller A: LLM | 177/400 | 44.25% | 1,615.97 | 0.04174 | 0.42634 |

The new POMDP gets 35 more questions correct than Controller A while using about 43.5% fewer tokens. Relative to repair once, it gets **one more question correct** (212 versus 211), at about 80.54 additional tokens per question. Repair once has higher utility at the selected penalty 0.248. The slight accuracy difference is not convincing evidence of superiority.

The new policy gets seven fewer questions correct than repair twice (212 versus 219), while using about 32.7% fewer tokens. It therefore trades accuracy for cost; it has **not exceeded 54.75% test accuracy**. The original lambda0.01 POMDP exactly matched repair twice.

Controller A remains the original frozen policy prompted with lambda0.01. Its lambda0.248 utility here reweights its existing outcomes; it is not a newly tuned LLM policy. Both utility columns use the same actual executed token totals. Do not choose whichever penalty makes a preferred method look best.

## Actions and domains

Actions: 332 repair-then-stop episodes, 61 repair-twice episodes, and seven immediate stops. No retry was selected. This is adaptive behavior across observed episodes, but adaptive behavior alone does not establish a benefit over a fixed baseline.

| Domain | POMDP accuracy | POMDP tokens | LLM accuracy | LLM tokens | Repair-once accuracy | Repair-once tokens |
|---|---:|---:|---:|---:|---:|---:|
| math | 66.00% | 705.88 | 52.50% | 1334.09 | 65.50% | 652.18 |
| science | 40.00% | 1121.09 | 36.00% | 1897.85 | 40.00% | 1013.72 |

## Paired uncertainty

Differences below are POMDP minus comparator. Accuracy differences are percentage points. Intervals use 2,000 paired question bootstrap resamples, conditional on this fixed model/cache. They exclude parameter-selection and model-fitting uncertainty, and are not corrected for six comparisons.

| Comparator | Accuracy difference | 95% interval | Utility difference at λ = 0.248 | 95% interval |
|---|---:|---|---:|---|
| Submit initial | +16.75 | [+12.50, +20.75] | +0.02530 | [-0.01679, +0.06636] |
| Repair once | +0.25 | [+0.00, +0.75] | -0.01747 | [-0.02433, -0.00927] |
| Repair twice | -1.75 | [-3.25, -0.25] | +0.09281 | [+0.07581, +0.10843] |
| Retry once | +14.75 | [+9.50, +20.00] | +0.08949 | [+0.03501, +0.14241] |
| Retry twice | +12.75 | [+7.75, +17.50] | +0.15408 | [+0.10440, +0.20132] |
| Controller A: LLM | +8.75 | [+5.75, +12.00] | +0.26172 | [+0.22951, +0.29622] |

The machine-readable summary also retains paired utility intervals at the common lambda0.01 reference.

## Selection, freeze, and interpretation

The earlier diagnostic examined 602 penalties on train/dev and highlighted the policy with highest dev accuracy, then lowest tokens, then lowest lambda. Lambda0.248 achieved 32/60 correct (53.33%) and 820.55 mean tokens on dev. The user then explicitly selected it for this separate test experiment. This does not change the original cost preference retroactively.

Earlier baseline, Controller A, and original POMDP test results had already been inspected before this follow-up. The fine sweep did not evaluate test, and this run uses the fixed selected value; nevertheless, this is an exploratory follow-up on an already viewed test set, not an untouched confirmation claim. Further tuning should use train/dev and would require another distinct experiment.

The runner verifies the saved dev selection and reproduces all dev paths, writes a durable freeze with source/model/dev hashes, checks those hashes, and only then loads the test split. The original two-state model is reproduced from the same 100 training trees and is not refitted on dev/test. Confidence/validity observations, pooled dynamics, smoothing, planning horizon, and tie handling remain unchanged.

Freeze timestamp: `2026-11-11T16:17:27.747659+00:00`. Test started: `2026-11-11T16:17:27.748817+00:00`. Test finished: `2026-11-11T16:17:32.727649+00:00`.

The archive contains exact beliefs/action values, paths, scores, tokens, model and dev evidence, source hashes, and the selected-value freeze. CPU time is recorded separately from tokens. There were zero new paid inference calls. The frozen original model, Controller A archive, milestone results, and diagnostic results are preserved.
