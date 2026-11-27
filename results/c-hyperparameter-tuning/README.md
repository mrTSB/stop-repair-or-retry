# C hyperparameter tuning: findings and evidence

The first search improves the **repeatedly reused development-panel** result from 177/256 to 178/256 while reducing the original compute proxy by 10.67%. It does **not** establish a generalization improvement: selection on three development folds and scoring the fourth yields 171/256, versus the already development-selected original checkpoint's 177/256. Keep the candidate separate from the frozen reference.

## Experiment index

| Run | Objective | Development result | Evidence |
| --- | --- | --- | --- |
| [2026-11-27 cost-constrained](2026-11-27-cost-constrained/REPORT.md) | Highest accuracy at no greater compute than original C | 178/256; compute 3330.65 | Full grid, checkpoint, per-question outputs, audit and integrity receipts |
| [2026-11-27 accuracy-first](2026-11-27-accuracy-first/REPORT.md) | Highest accuracy without a compute ceiling | 178/256; compute 3255.71; no further accuracy gain | 61,098 subject candidates, full outcomes, checkpoint, audit, plots and fold sensitivity |

The [accuracy-first refinement](2026-11-27-accuracy-first/REPORT.md) evaluated 61,098 subject settings with no compute ceiling and found **no additional accuracy gain**: 178/256 at 3255.71 compute, with fold sensitivity 169/256 versus original C's 177/256. It improves the compute proxy further, not the established accuracy result. Neither candidate is independently validated.

The repository contains complete reports, predeclared search settings, all candidate outcomes, saved checkpoints, standalone audits, reproducible plotting code and development-only charts. No new paid compute or reserved-test evaluation was used.

## Relationship to existing work

This extends the audited `abc-runtime-2026-11-25` checkpoint. [Cayleigh's independent review-gate sweep](https://github.com/mrTSB/stop-repair-or-retry/blob/cayleigh/c-review-gate-sweep/results/c-review-gate-sweep/README.md) explores a coarser penalty/threshold grid and reports no setting above 177/256. The present search additionally varies the existing physics ensemble's prior weight and uses finer small penalties. Neither search changes the neural architecture, trained observer weights, tokenizer, subject router, action horizon or cache/billing rules.

All reported accuracies here are development results. The reserved 1,024-question test set is not used for selection, inference or grading. Costs are the archived parameter-weighted token proxy, not measured FLOPs, latency or dollar charges.
