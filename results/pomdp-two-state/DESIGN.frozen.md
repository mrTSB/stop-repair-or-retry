# Controller B design decisions

Recorded November 10, 2026 (America/Los_Angeles), before fitting the new milestone model. This document is the durable record of the user's choices and the implementation defaults. It supersedes chat-only planning for this workflow. Older repository controllers are historical implementations, not the model or evidence used here.

## Scope selected by the user

| Decision | Milestone choice | Reason and boundary |
|---|---|---|
| Hidden states | Two: current answer incorrect / correct | A small, interpretable correctness belief; remaining solver calls are observed, not hidden. |
| Observation features | Self-reported confidence and answer-format validity | No extra LLM inference. Do not add answer-change features to this milestone. |
| Domain pooling | One model shared by math and science | Avoid splitting 100 training questions into overly small estimation groups. Report domain results separately. |
| Primary reward | Correctness minus 0.01 × total executed tokens / 1,000 | Same primary penalty as Controller A. |
| Sensitivity analysis | A small predeclared penalty sweep | Report tradeoffs without choosing the best-looking test penalty. |
| Planning comparison | Two-step POMDP and existing comparators only | The user declined an additional one-step/myopic controller comparison. |
| Future state expansion | Design interfaces for four states later | Four states are not implemented, fitted, or claimed as a milestone contribution. |

## Implementation defaults

These are implementation choices made by the assistant, rather than separate user selections. They must be declared before results, and changed only through a new versioned experiment.

- Train exclusively on the published 100-question training split. Use all 60 dev questions for checks, then freeze before replaying all 400 test questions. Project splits, not source benchmark labels, determine membership.
- Use the published solver cache and frozen scoring. Do not regenerate solver responses, call Modal, load older fitted controllers, or use their results.
- Use terminal correctness reward 1 for correct and 0 for incorrect. Repair and retry replace the current answer. Stop submits it; at zero remaining solver calls, submit automatically.
- Plan exactly through the two remaining solver calls. Integrate over possible future observations. Do not approximate with one-step decisions.
- Estimate the initial correctness prior from training roots. Estimate repair and retry transitions conditional on current correctness from training edges. Retry is not assumed independent of current correctness: question difficulty can correlate separate attempts.
- Use additive smoothing with pseudocount 1 for initial, transition, and observation probabilities. Keep all probabilities nonzero; disclose that smoothing does not solve model misspecification.
- Treat confidence and answer-format validity as one categorical observation, rather than assuming they are conditionally independent. Confidence categories are low, medium, high, and missing; validity is valid/invalid, giving eight possible symbols.
- Answer-format validity means the existing parser found a nonempty usable answer (`answer_parse_valid`), not whether it matches the gold answer. Correctness labels are available during training/scoring only. The controller receives no gold labels, unseen branch contents, question IDs, or domain features.
- Estimate initial observation probabilities from training roots and action-conditioned observation probabilities from successor nodes. All training edges are pooled across depths; disclose the resulting stationarity approximation and correlation of branches within a question.
- Estimate expected action token costs from training edges conditional on current correctness. Use pooled action means as the fallback for an unobserved state/action cell. The planner uses expected future costs; evaluation charges actual visited cache tokens.
- Use belief vectors, transition matrices, observation tables, and terminal-reward vectors of configurable state dimension. Keep two-state supervised fitting separate from generic belief updates and planning. A four-state model will require a new estimator and evidence for its state definitions, not just changing a constant.
- Break numerical ties deterministically in the order stop, repair, retry, using a small fixed tolerance. Never break ties using cache outcomes.
- Use the already declared penalty grid [0, 0.001, 0.01, 0.1, 1]. The primary penalty remains 0.01 regardless of results. Report accuracy, tokens, and within-penalty utility comparisons; do not select a penalty by comparing raw utilities across penalties.
- Include the five fixed baselines and frozen Controller A at its original penalty 0.01. Do not present Controller A as adapted to other penalties: changing its prompt would require new inference. No new one-step controller or oracle is added.
- The POMDP makes no LLM calls. Its controller-token cost is zero, but its CPU planning time is not zero; record it separately. Research tokens are not FLOPs or USD, particularly across differently sized solver/controller models.

## Freeze and interpretation

Predeclare these choices, fit once on train, inspect dev for implementation defects, and freeze the model/configuration before test. Do not change parameters in response to dev/test scores without creating a separate experiment and documenting that change. Existing Controller A and baseline test results have already been viewed, so the test set is not described as untouched. Do not target particular test failures when designing this controller.

The two-state model is an approximation: correctness alone need not make repairability, difficulty, or observations Markov. Training edges share questions and are not independent samples. Report this limitation alongside sample counts, smoothing, and uncertainty. A good milestone does not require outperforming repair once.

## Deferred extensions for the final project

| Extension | What it could test | Evidence or implementation needed |
|---|---|---|
| Four hidden states | Whether different kinds of incorrect answers improve action selection | A defensible latent-state estimator and adequate training evidence. Candidate states: correct; incorrect/repair promising; incorrect/retry promising; incorrect/neither likely to help within budget. These labels are provisional, not ground truth inferred from one cached outcome. |
| Answer-change observations | Whether agreement or change across attempts helps belief updates | A history-aware observation definition/model. Do not silently treat answer agreement as conditionally independent of previous answers. |
| Domain-specific dynamics | Whether math/science differences justify separate estimates | Training/dev evidence and sufficient counts, potentially partial pooling. |
| More training questions or repeated samples | More reliable estimation of transitions, uncertainty, and latent repairability | Separate collection plan and budget; preserve dev/test boundaries and seeds. |
| One-step versus two-step ablation | Whether lookahead itself adds value | Not part of this milestone; requires a new explicit experiment decision. |
| New confirmation set | Whether gains generalize beyond an already inspected benchmark | Fresh held-out questions if later choices are influenced by current test outcomes. |

Future work is conditional on training/dev evidence. Four states are not assumed to outperform two. Preserve the two-state model and its frozen results as the comparison when expanding.
