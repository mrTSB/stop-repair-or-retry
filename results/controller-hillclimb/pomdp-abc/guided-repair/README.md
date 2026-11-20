# Frozen POMDP, LLM, and hybrid results

All three observed accuracy targets passed. The table contains all five fixed Stop/Repair/Retry variants and the best development-selected controller from each family, evaluated on the same 600 held-out questions.

| Method | Correct | Accuracy (%) | Solver tokens | LLM-controller tokens | Compute proxy |
|---|---:|---:|---:|---:|---:|
| Stop | 338/600 | 56.33 | 253.7 | 0.0 | 328.4 |
| Repair once | 389/600 | 64.83 | 675.9 | 0.0 | 875.0 |
| Repair twice | 423/600 | 70.50 | 1177.5 | 0.0 | 1524.3 |
| Retry once | 398/600 | 66.33 | 603.6 | 0.0 | 781.4 |
| Retry twice | 401/600 | 66.83 | 951.7 | 0.0 | 1231.9 |
| Controller A — POMDP | 435/600 | 72.50 | 741.6 | 0.0 | 960.1 |
| Controller B — LLM | 494/600 | 82.33 | 604.7 | 701.9 | 1464.3 |
| Controller C — hybrid | 490/600 | 81.67 | 638.3 | 547.6 | 1357.9 |

Token and compute columns are means per question. Every visited solver call and acquired LLM review is charged, including reviews that recommend stopping.

The prespecified primary baseline was **Repair twice**, selected by development accuracy and then lower compute. C changes accuracy by **+11.17 percentage points** (paired bootstrap 95% interval +8.17 to +14.33) and saves **10.92%** of the active-parameter compute proxy against it. The paired comparison has 82 C wins and 15 losses.

B has the highest controller accuracy point estimate. Compared with B, C changes accuracy by -0.67 points (paired 95% interval -1.83 to +0.50) and uses 7.27% less of the compute proxy. The accuracy-only ordering is B ≈ C > A; the requested strict accuracy floors are met. C also matches the always-review hybrid's total correct count at 20.65% less compute, with 3 paired wins and 3 losses.

The fixed method with the highest observed test accuracy is **Repair twice**. This is a descriptive comparison; policy selection used development data.

| Controller | Required accuracy | Observed accuracy | Accuracy target |
|---|---:|---:|---|
| A | >65% | 72.50% | Passed |
| B | >65% | 82.33% | Passed |
| C | >75% | 81.67% | Passed |

The targets concern observed held-out accuracy. Individual 95% Wilson intervals are: A 68.79–75.92%, B 79.08–85.18%, C 78.37–84.56%.

A models hidden current-answer correctness, learns action transitions and observation likelihoods, updates its belief by Bayes’ rule, and plans over the two-call remaining horizon. Current action estimates can condition on observed response features; future Bellman values use pooled stage-specific dynamics. Selected settings: `{"contextual": true, "initial_mode": "binned", "penalty": 0.08, "science_weight": 1.0, "threshold": 0.0}`.

B uses the frozen LLM’s independent answer and action assessment, with a development-selected confidence/agreement rule. Its inference function does not call the POMDP. Selected settings: `{"judge_stop": "agreement", "reuse_review": true}`.

C combines the POMDP with selective paid LLM reviews, review-conditioned action dynamics, and an optional Bayesian agreement observation after a repair. B receives a matching paid-review reuse candidate. A review acquired at the initial checkpoint can be passed to the compact solver during repair; the latest compact-solver answer remains the submitted answer. Selected settings: `{"contextual": false, "initial_mode": "binned", "penalty": 0.15, "reuse_review": true, "science_weight": 0.7, "threshold": -0.02}`.

Training used 192 paired questions and 2048 additional archived initial responses. Development used 120 disjoint retired study-test questions. The fresh panel contains 300 unused GSM8K official training-source questions, 150 MMLU-Pro chemistry questions, and 150 physics questions. IDs and normalized question text were checked against prior project use and recorded exclusions. The unused biology pool is exhausted. The cohort and continuation budget differ from earlier project panels.

All methods use the same frozen Qwen3.5-4B solver checkpoint, unchanged initial prompt with a 128-token cap, shared 256-token repair/retry continuations, and at most two additional solver calls. B and C share the same Qwen3.5-35B-A3B reviewer and a 512-token review cap. Completed malformed/truncated reviews are retained with the declared fallback; no question is removed because of a poor model answer.

The compute proxy is `(4 × solver tokens + 3 × reviewer tokens) / 3.09`, using nominal active parameter counts in billions. The [official reviewer model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) specifies 35B total and 3B activated parameters. C uses 1185.8 raw solver-plus-reviewer tokens per question versus 1177.5 for Repair twice (+0.71%). Its compute saving therefore refers specifically to active-parameter weighting. This proxy approximates model token computation; it does not measure memory, latency, energy, or offline data-collection cost. The full summary also reports a conservative sensitivity using all 35B reviewer parameters.

| Controller | Active-parameter proxy | Total-parameter sensitivity | Mean solver calls | Mean review calls |
|---|---:|---:|---:|---:|
| A | 960.1 | 960.1 | 2.070 | 0.000 |
| B | 1464.3 | 8733.5 | 1.513 | 1.113 |
| C | 1357.9 | 7028.3 | 1.585 | 0.875 |

Matched controls are archived in the summary: C without review: 71.17% at 817.9 compute, C always reviews: 81.67% at 1711.3 compute, Random reviews: 78.33% at 1348.8 compute. They distinguish review acquisition from independently selected planner fits.

The decision models and settings were frozen before the test. The 0.19.0 GPU runtime then omitted completion/token metadata on a solver request in both API modes. Before any test score was examined, the initial seven unscored cases were archived and the runtime was updated to 0.40.16 with identical model-weight digests. All 600 cases were regenerated on that runtime. The runtime amendment changed only the deployment image; the selected-model/settings/baseline hash remains `77e57d3bc8f7cc1055c8e73581d61ae0f3af62c5565eb7f8d46b7e47c21a570c`.

Thirteen focused unit tests passed. The completed-test audit independently checks every fixed/adaptive policy outcome, solver/review token charges, the three-call solver horizon, role separation, hidden-reference invariance, and replay using only paid observations. CPU replay of the packed archive is verified separately.

The frozen model, development search, full response archive, per-question decisions, paired intervals, and checksum manifest are included here. Source and collection instructions are in `experiments/controller-hillclimb/pomdp-abc/guided-repair`.

```sh
data=results/controller-hillclimb/pomdp-abc/guided-repair/data
out=/private/tmp/srr-guided-replay
mkdir -p "$out"
cp results/controller-hillclimb/pomdp-abc/guided-repair/frozen.json "$out/frozen.json"
uv run --python 3.12 --with numpy==2.5.4 --no-project python \
  experiments/controller-hillclimb/pomdp-abc/guided-repair/extended_study.py score \
  --data "$data" --output "$out"
uv run --python 3.12 --with numpy==2.5.4 --no-project python \
  experiments/controller-hillclimb/pomdp-abc/guided-repair/audit.py \
  --data "$data" --output "$out"
```
