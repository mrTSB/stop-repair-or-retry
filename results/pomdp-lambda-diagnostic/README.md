# Exploratory train/dev lambda sweep

Read [report.md](report.md). This is separate from the frozen two-state milestone: no new test evaluation or new model fitting occurred.

Reproduction uses the frozen core/model and a fresh output directory:

```sh
python3 -B experiments/pomdp-controller/lambda_sweep.py --output runs/fine-lambda-v1
python3 -B experiments/pomdp-controller/policy_regions.py --output runs/lambda-regions-v1
```

Both are offline. The fine-grid protocol declares the grid and the dev-only candidate rule before evaluation. Train scores are diagnostic; they do not select the highlighted candidate. `dev-policies.json` stores every distinct per-question path/score/token vector; repeated grid policies share a signature. Exact action values and beliefs for the highlighted candidate are in `dev-candidate-paths.json`. The independent audit checks summary arithmetic and the candidate rule. The eight observed policies are not eight independent experiments; all use the same dev questions.

The initial best-accuracy tie-break picks low tokens, then low lambda. It is an exploratory highlighting rule, not a replacement of the primary lambda. Utilities at the varying lambdas have different objectives; `utility_at_reference_lambda_0.01` is supplied for consistent reweighting of these POMDP policies. Controller A was not rerun or tuned in this diagnostic.
