# Frozen POMDP lambda 0.248 follow-up

Read [report.md](report.md) for the full comparison against Controller A and five baselines. This separate experiment uses the unchanged training-fitted two-state model and the dev-selected lambda0.248. The original primary experiment remains lambda0.01.

Offline reproduction/verification at this repository revision:

```sh
python3 -B experiments/pomdp-controller/frozen_lambda.py verify --run results/pomdp-lambda-0248
```

`freeze.json` records the explicit user choice, selected dev evidence, source hashes and timestamp before test loading. `model.json` and `dev-evidence.json` are exact copies of the original fitted model and selected dev paths. `test-paths.json` contains every executed decision with beliefs/action values and final score/cost. `summary.json` includes all seven methods, domains, accuracy intervals and paired comparisons at both penalties. Controller A was not rerun at lambda0.248: that utility is a reweighting of its existing lambda0.01-prompted policy.

`validation.json` records the independent audit. `artifact-sha256.json` covers every archived file except itself. No original artifact or frozen model was overwritten; no paid inference was used. Previously viewed test results and dev-selection uncertainty limit confirmatory claims.

A fresh offline collection can be made with the `collect` command and a new output directory, but it is unnecessary to reproduce deterministic paths; use `verify` instead. Existing output directories are refused. Changes to source, model or development evidence invalidate the frozen run and must be versioned separately.
