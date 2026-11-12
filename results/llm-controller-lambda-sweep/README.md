# Controller B (LLM) at additional token penalties

Naming follows the project proposal: Controller A is the two-state POMDP, Controller B is this LLM controller.

The unchanged Qwen2.5-14B-Instruct controller from the lambda 0.01 run ([full evaluation](../llm-controller-full-evaluation/README.md)) was rerun at lambda 0.1, 0.248 and 1. Each run collected all 60 dev questions, froze, then collected all 400 test questions, on private Modal RPC (one A100-80GB). Driver: [`experiments/llm-controller/modal_lambda_sweep.py`](../../experiments/llm-controller/modal_lambda_sweep.py). It reuses the original GPU worker in `modal_full_evaluation.py` without modification; only the penalty shown to the controller (and used in utility), the Modal profile and the spending reservation differ. 0.1 and 1 are on the predeclared grid in `reward_design.json`; 0.248 is not, and was chosen from Controller A's dev-only diagnostic so A and B can be compared at one penalty.

## Test results (400 questions)

| Lambda | Accuracy | Mean total tokens | Mean utility | Best fixed baseline (utility) |
|---:|---:|---:|---:|---|
| 0.01 (original run) | 44.25% | 1,616.0 | 0.426 | repair twice (0.534) |
| 0.1 | 43.75% | 1,612.7 | 0.276 | repair once (0.444) |
| 0.248 | 45.00% | 1,619.2 | 0.048 | repair once (0.321) |
| 1 | 43.75% | 1,613.9 | -1.176 | submit initial (0.022) |

Test action paths are nearly identical at every penalty:

| Lambda | stop | repair → stop | repair → retry | repair → repair | retry → repair |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 241 | 139 | 16 | 3 | 1 |
| 0.1 | 242 | 138 | 16 | 3 | 1 |
| 0.248 | 240 | 143 | 15 | 1 | 1 |
| 1 | 242 | 138 | 16 | 3 | 1 |

Paired test utility differences, Controller B minus baseline (95% paired bootstrap interval; not adjusted for multiple comparisons):

| Lambda | vs submit initial | vs repair once | vs repair twice | vs retry once | vs retry twice |
|---:|---|---|---|---|---|
| 0.1 | -0.052 [-0.082, -0.023] | -0.168 [-0.202, -0.136] | -0.135 [-0.170, -0.102] | -0.038 [-0.086, +0.010] | -0.024 [-0.072, +0.024] |
| 0.248 | -0.230 [-0.261, -0.198] | -0.272 [-0.305, -0.239] | -0.162 [-0.198, -0.126] | -0.166 [-0.217, -0.114] | -0.101 [-0.151, -0.050] |
| 1 | -1.199 [-1.283, -1.118] | -0.871 [-0.947, -0.793] | -0.366 [-0.443, -0.290] | -0.879 [-0.973, -0.791] | -0.558 [-0.641, -0.475] |

Dev accuracy was 50.00% (lambda 0.1), 48.33% (0.248) and 50.00% (1), at about 1,540 mean tokens.

## What this shows

Within this setup, raising the penalty in the prompt from 0.01 to 1 did not change the controller's behavior: action paths, accuracy and tokens stay essentially fixed. Because its own decisions cost about 1,000 tokens per question, Controller B falls below every fixed baseline on utility once lambda is 0.248 or higher, and below all but the two retry baselines (intervals overlap zero) at 0.1. For comparison, Controller A's separately frozen lambda 0.248 test run scored 53.00% at 913 mean tokens. This is a result for one prompt, one 14B controller and one solver cache; it does not establish that LLM controllers in general ignore stated cost objectives. Existing test outcomes had already been inspected.

## Files and limits

Each `lambda-*` folder holds `config.json`, `protocol.json` (hashes of driver, worker and prompt; reservation), `backend.json`, `frozen.json`, `dev/` and `test/` (journals, per-question results, manifests, reports, summaries), frozen copies of both driver files, `modal-app.json`, the watchdog record and `collection.log`. All three Modal apps were confirmed stopped after collection. Each run reserved a worst case of $3.15 against a $20 cap ($9.45 total); actual charges come from the Modal ledger and are not recorded here. Wall time per run was about 7–8 minutes for 644–645 controller decisions. Manifests contain absolute local paths for provenance.
