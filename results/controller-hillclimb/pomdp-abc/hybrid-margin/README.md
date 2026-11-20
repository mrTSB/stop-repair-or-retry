# Fresh accuracy-margin evaluation

All eight methods were frozen before evaluating the same 800 previously unused study questions. The compact solver checkpoint and the two additional solver-call budget are shared. A is a POMDP, B is an independent LLM controller, and C combines POMDP planning with paid LLM reviews.

| Method | Correct / 800 | Accuracy (%) | Mean raw tokens | Active-parameter compute proxy |
|---|---:|---:|---:|---:|
| Stop | 486 | 60.75 | 256.6 | 332.1 |
| Repair once | 550 | 68.75 | 683.4 | 884.6 |
| Repair twice | 564 | 70.50 | 1188.8 | 1538.8 |
| Retry once | 526 | 65.75 | 607.8 | 786.8 |
| Retry twice | 521 | 65.12 | 959.2 | 1241.7 |
| A | 578 | 72.25 | 872.5 | 1129.4 |
| B | 682 | 85.25 | 1298.9 | 1455.8 |
| C | 665 | 83.12 | 1136.9 | 1309.1 |

The highest observed fixed baseline is **Repair twice**. The confirmed targets use percentage points: A and B each need at least +10 over the highest baseline; C needs at least +10 over the higher of A/B, with +15 a stretch target. A clear C-versus-B win also requires paired 95% intervals favoring C for accuracy and compute.

| Criterion | Passed |
|---|---|
| A_plus_10_points | No |
| B_plus_10_points | Yes |
| C_plus_10_points | No |
| C_plus_15_points_stretch | No |
| C_accuracy_clearly_above_B | No |
| C_compute_clearly_below_B | Yes |

All required criteria passed: **False**.

Intervals are paired, subject-stratified bootstrap intervals with 5,000 resamples. The highest baseline and highest A/B are recomputed inside each resample. The full intervals are in `test/summary.json`.

The compute proxy is `(4 × solver input/output tokens + 3 × reviewer input/output tokens + verifier parameters in billions × verifier padded token slots) / 3.09`. The frozen solver is Qwen3.5-4B and the shared reviewer is Qwen3.5-35B-A3B (35B total, 3B activated parameters). All acquired reviews are charged, including reviews recommending Stop. Reused review text in a solver prompt is also charged. This proxy does not measure wall-clock time, memory, energy, or offline collection/training cost. Total-parameter sensitivity is retained in the summary.

If selected, the small non-generative verifier supplies noisy observations to the POMDP; it does not generate answers or choose actions. Its input text and padded computation slots are reported separately, and every pass is charged. The compact solver remains frozen.

Training uses 672 retired questions and development uses 240 disjoint retired questions. The fresh study panel contains 400 GSM8K official training-source questions, 200 MMLU-Pro chemistry questions, and 200 physics questions. This is a study holdout, not the official GSM8K test split. The historical exclusion snapshot and exact split hashes are included. Prior test panels are explicitly retired into training/development and are not claimed as new validation.

The audit checks 6,400 outcomes against raw solver/review receipts and repeats 4,800 adaptive decisions with changed references and only paid observations. Feature caches are cleared before these replays. Completed malformed/truncated responses are retained. Transport-only retries are logged separately; a completed poor answer is never retried by the transport wrapper.

The four hidden POMDP states combine persistent question difficulty with current-answer correctness. A grouped, cross-fitted observation classifier supplies evidence; Bayesian filtering updates the state belief. C may plan for acquiring a review at a future checkpoint. Both B and C may use paid review feedback in a final repair when selected by development; the submitted answer always comes from the compact solver.

Frozen controller configurations:

```json
{
  "A": {
    "family": "A",
    "model": "leaf32-smooth32-late1",
    "settings": {
      "penalty": 0.0,
      "science_weight": 0.7,
      "reuse_review": true,
      "late_feedback": true
    },
    "dev": {
      "n": 240,
      "correct": 183,
      "accuracy": 0.7625,
      "mean_compute": 1109.940668824164,
      "mean_solver_tokens": 857.4291666666667,
      "mean_judge_tokens": 0.0,
      "mean_judge_calls": 0.0,
      "mean_solver_calls": 2.404166666666667,
      "mean_total_parameter_compute": 1109.940668824164
    },
    "probe": "runs/controller-hillclimb/pomdp-abc/hybrid-margin/late-routing-probe",
    "protocol": "late"
  },
  "B": {
    "family": "B",
    "model": null,
    "settings": {
      "judge_stop": "agreement",
      "reuse_review": true
    },
    "dev": {
      "n": 240,
      "correct": 198,
      "accuracy": 0.825,
      "mean_compute": 1503.089266450917,
      "mean_solver_tokens": 636.1083333333333,
      "mean_judge_tokens": 700.0375,
      "mean_judge_calls": 1.1208333333333333,
      "mean_solver_calls": 1.5416666666666667,
      "mean_total_parameter_compute": 8752.668554476808
    },
    "probe": "runs/controller-hillclimb/pomdp-abc/hybrid-margin/late-routing-probe",
    "protocol": "late"
  },
  "C": {
    "family": "C",
    "model": "leaf32-smooth32-late0",
    "settings": {
      "penalty": 0.06,
      "science_weight": 0.7,
      "reuse_review": true,
      "late_feedback": false,
      "contextual_transitions": true,
      "context_weight": 0.7,
      "threshold": 0.03,
      "plan_reviews": false
    },
    "dev": {
      "n": 240,
      "correct": 200,
      "accuracy": 0.8333333333333334,
      "mean_compute": 1347.4689859762677,
      "mean_solver_tokens": 661.0416666666666,
      "mean_judge_tokens": 506.50416666666666,
      "mean_judge_calls": 0.7833333333333333,
      "mean_solver_calls": 1.8208333333333333,
      "mean_total_parameter_compute": 6592.819579288026
    },
    "probe": "runs/controller-hillclimb/pomdp-abc/hybrid-margin/contextual-probe",
    "protocol": "original"
  }
}
```
