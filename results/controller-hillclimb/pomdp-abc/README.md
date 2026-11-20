# POMDP A, LLM judge B, and hybrid C

This is the archived 420-question study. The [latest frozen 600-question results](guided-repair/README.md) meet the requested A/B/C accuracy targets and include all five fixed baseline variants. The two panels differ and should be reported separately.

The frozen hybrid scored 273 of 420 on the fresh study holdout. It gained 18 correct answers over Repair and used 34.5% less mean estimated compute. The six primary policies share their initial solver responses, action prompts, generation limits, and sampled solver branches.

| Method | Correct | Accuracy | Mean compute |
|---|---:|---:|---:|
| Stop | 238/420 | 56.67% | 327.05 |
| Repair | 255/420 | 60.71% | 1,236.48 |
| Retry | 227/420 | 54.05% | 654.25 |
| A, POMDP planner | 257/420 | 61.19% | 628.45 |
| B, LLM judge | 264/420 | 62.86% | 2,409.75 |
| C, selective hybrid | 273/420 | 65.00% | 809.86 |

Repair uses two fixed repairs and Retry uses one fixed retry, with these depths selected on development accuracy. Every adaptive policy has at most two additional solver calls. Mean compute charges all executed solver and judge input/output tokens, weighted by nominal model size and expressed in Qwen2.5-3B-equivalent tokens. The solver is the same frozen Qwen3.5-4B checkpoint across all six policies. B and C use the same frozen Qwen3.5-9B judge. Offline collection and fitting are separate from these inference totals.

C gained 4.29 percentage points over Repair, with a paired 95% bootstrap interval of +0.48 to +8.10 points. Against A, it gained 3.81 points [+0.95, +6.67] while using 28.9% more compute. Against B, it gained 2.14 points [-0.95, +5.48] and used 66.4% less compute. Its accuracy lead over B remains uncertain. Stop remains the cheapest operating point. C used 23.8% more compute than Retry while gaining 10.95 points in accuracy.

On math, C scored 150/210, compared with 146 for Repair, 136 for A, and 140 for B. On science, it scored 123/210, compared with 109 for Repair, 121 for A, and 124 for B. The aggregate ranking reflects both domains.

Matched controls help identify the contribution of acquiring judge evidence. C's fitted planner with acquisition disabled scored 269/420 at 661.57 equivalent tokens. The selected gate added four correct answers, with a paired accuracy interval of -0.48 to +2.38 points. Always acquiring the same evidence scored 274/420 at 2,637.29 tokens. The selected gate scored one fewer correct answer at 69.3% lower compute. A development-fixed random acquisition control scored 268/420 at 889.37 tokens. A and C use separately selected classifier settings and planning penalties, so their aggregate difference also reflects these choices. These controls remain in the archived analysis.

The initial observation model used 1,820 archived initial responses and the new branches from 144 training questions. All seven solver nodes were collected for each training question and each of 120 development questions. Forty-three missing initial confidence observations were recaptured with exact response-text and token-count matches. Judge observation tables used 432 training verdicts. The POMDP hidden state is current answer correctness, with Bayesian filtering and two-step Bellman planning. Initial belief calibration, linear or contextual features, smoothing, planning penalties, and acquisition thresholds were selected on development data. All three families used development accuracy minus 0.05 times mean equivalent tokens divided by 1000 as the selection objective.

The selected development accuracies were 72.50% for A, 70.83% for B, and 75.00% for C. The strongest development baseline reached 65.83%. These operating points were frozen before holdout collection and scoring. The holdout scores show a substantial generalization gap from development.

The 420-question holdout contains 210 unused questions from the official GSM8K training source and 210 unused MMLU-Pro questions. Only 28 unused biology questions remained, so science contains 28 biology, 91 chemistry, and 91 physics questions. IDs and normalized question stems exclude earlier recorded uses. This is a study holdout with a different subject mix from the earlier 400- and 600-question panels.

The collector used 1,909 solver responses and 709 judge responses across all nine frozen policies and controls, collecting their requested union. Each method was charged only for its own calls. A recovery journal preserves completed responses within interrupted questions. The packed archive reproduced the model freeze, development search, test summaries, and per-question scores byte for byte. Independent reconciliation checked all 3,780 policy outcomes against raw paths, grades, and token charges. Decision replay remained invariant when references were removed from all 420 questions. The nine new numerical tests, 18 core tests, and original cached-dataset validator passed. [Code and replay instructions](../../../experiments/controller-hillclimb/pomdp-abc/README.md)
