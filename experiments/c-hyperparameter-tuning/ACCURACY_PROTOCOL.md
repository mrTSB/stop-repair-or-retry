# Accuracy-first refinement protocol (2026-11-27)

This protocol is recorded before inspecting the new run's outcomes. The earlier 178/256 candidate and its adverse 171/256 fold-sensitivity result motivate a bounded refinement; they are not independent evidence for the next search.

## Question and permitted changes

Can finer tuning of the existing C review gate, compute penalty and physics prior improve development accuracy without changing architecture? This run retains the exact frozen subject experts, parent models, trained weights, tokenizer, feature extraction, dynamics, signal adapter, action budget and accounting. Only the existing penalty, threshold and positive normalized two-parent prior may change.

## Predefined search

- Penalties: `0, .0001, .00025, .0005, .00075, .001, .0015, .002, .0025, .003, .0035, .004, .0045, .005, .006, .007, .008, .009, .01, .0125, .015, .0175, .02, .025, .03, .04, .05, .075, .1, .2, .5, 1` (32).
- Physics first-parent weights: `.01, .025, .05, .075, .1, .125, .15, .175, .2, .225, .25, .3, .35, .4, .45, .5, .6, .65, .7, .75, .8, .9, .95, .975, .99` (25). Second weight is the complement. The two original parents and their ordering remain fixed.
- Math and chemistry use their existing single parent, giving 32 parameter pairs each. Physics has 800. Total: 864 penalty/prior pairs before threshold choices.
- Threshold candidates include the preceding 20 fixed thresholds and one representative from each interval where the existing gate changes its outcome. The interval boundaries are `information_value - 1e-9`, obtained from frozen policy predictions without reference labels. Threshold zero covers the first nonnegative interval; midpoints cover subsequent intervals; a point above the last boundary covers no review. No question-specific rule is introduced: one selected threshold is deployed per subject.
- Retain all three original subject configurations. Retain every evaluated candidate and per-question result, using lossless compression where needed.

## Objective

Primary: maximize total development correctness **without a compute ceiling**, breaking ties by lower total compute. Also report optima under original C's and B's compute ceilings. The primary accuracy score must not be described as an efficiency improvement unless the measured costs support that description.

Selection combines subject candidates through integer-correctness dynamic programming. An independently enumerated set of subject correctness/cost frontier combinations checks the selected objective. The final model is always directly replayed; the threshold shortcut is not accepted as sufficient evidence for the winner.

## Validation and reporting

1. Reproduce frozen C's 256 per-question paths, correctness and bills before searching.
2. Keep the original no-test-read and no-model-call guards active.
3. Save an unloaded candidate checkpoint and all settings, grids, source hashes and outcome artifacts.
4. Directly replay all 256 selected outcomes, then replay each with the reference label erased and only paid-prefix observations exposed.
5. Independently recompute correctness and the complete solver, reviewer and observer component bills.
6. Reload the saved checkpoint in a separate process and repeat the audit. Reverting only the permitted settings/prior weights must yield the exact original model serialization.
7. Repeat the same four subject-stratified development-fold selection-sensitivity diagnostic (seed `20261127`), now selecting on accuracy without a compute ceiling. Report every fold and aggregate, including deterioration. This is not fresh validation: the panel and the original checkpoint have already been selected repeatedly.
8. Record which questions improve/regress and report matched accuracy/compute tradeoffs and negative findings.
9. Verify archived input hashes after the run. Preserve the frozen checkpoint and previous results.

The reserved 1,024-question test set receives no inference or grading and is not consulted for model selection. No new paid compute or neural forwards are needed. A strong result on this search alone does not justify a generalization or statistical-significance claim.
