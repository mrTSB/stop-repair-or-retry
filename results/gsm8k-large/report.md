# GSM8K baseline results

Dataset: openai/gsm8k main; split seed 20261101; 300 training, 100 development, 500 test questions. The test set excludes the inspected pilot questions.

Primary cost penalty: 0.01 per 1,000 total input and output tokens. All policies use paired cached solver trajectories. Intervals are question-level bootstrap intervals; their validity assumes sampled questions are representative and does not include model-seed or benchmark contamination uncertainty.

| Policy | Accuracy | 95% CI | Mean tokens | Utility | Accuracy difference vs stop (95% CI) |
|---|---:|---:|---:|---:|---:|
| stop | 37.8% | [33.4%, 42.0%] | 275 | 0.3753 | +0.0% [+0.0%, +0.0%] |
| repair | 59.4% | [55.0%, 63.8%] | 1112 | 0.5829 | +21.6% [+17.8%, +25.6%] |
| retry | 36.8% | [32.6%, 41.2%] | 817 | 0.3598 | -1.0% [-5.8%, +4.0%] |
| schedule | 50.8% | [46.6%, 55.2%] | 959 | 0.4984 | +13.0% [+8.0%, +18.2%] |
| threshold | 59.4% | [55.0%, 63.8%] | 1112 | 0.5829 | +21.6% [+17.8%, +25.6%] |
| one_step | 59.4% | [55.0%, 63.8%] | 1112 | 0.5829 | +21.6% [+17.8%, +25.6%] |
| supervised | 59.2% | [54.8%, 63.6%] | 1106 | 0.5809 | +21.4% [+17.6%, +25.2%] |
| pomdp | 59.4% | [55.0%, 63.8%] | 1112 | 0.5829 | +21.6% [+17.8%, +25.6%] |
| judge_full | 38.0% | [33.8%, 42.4%] | 599 | 0.3740 | +0.2% [+0.0%, +0.6%] |
| judge_compact | 40.0% | [35.4%, 44.4%] | 740 | 0.3926 | +2.2% [-1.6%, +6.2%] |
| hybrid | 59.4% | [55.0%, 63.8%] | 1112 | 0.5829 | +21.6% [+17.8%, +25.6%] |
| majority | 41.6% | [37.2%, 45.8%] | 817 | 0.4078 | +3.8% [+1.2%, +6.4%] |

The full-history and compact judges use the solver model. Their prompts are conditioned on the primary cost penalty; the additional penalty-grid outputs include only non-judge policies. The full cached tree has one initial plus six solver-branch calls per question, and dev/test have six judge calls per question. These offline data-generation costs are separate from the per-policy inference costs in the table.

## Response quality
- train: 2100 solver calls (6 invalid); 0 judge calls (0 invalid); 1763.4 seconds summed generation time.
- dev: 700 solver calls (5 invalid); 600 judge calls (1 invalid); 740.5 seconds summed generation time.
- test: 3500 solver calls (10 invalid); 3000 judge calls (9 invalid); 4388.4 seconds summed generation time.

An output with recoverable fields may still be wrong, and numeric grading is not a proof of mathematical equivalence. Review examples and scorer edge cases before research claims. This single-model, single-seed study does not establish transfer to other models or datasets. CoRefine and ARR are not reproduced.

Aggregate summaries, paired comparisons, cost sweeps, training-size checks, repeat-seed analysis, and response diagnostics are included beside this report. Per-question outputs and raw trajectories remain in the local run directory and are not part of this aggregate bundle. Source hashes and selection seeds are in manifest.json and methodology.json.
