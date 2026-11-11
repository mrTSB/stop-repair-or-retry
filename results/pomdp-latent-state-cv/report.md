# Four-state latent-model diagnostic

The user requested extra hidden states as the next POMDP improvement. We implemented and compared four states with a matched two-state control. **This prototype did not improve accuracy and is not promoted to a test experiment.** It is separate from the frozen milestone and lambda-0.248 experiments.

## Chosen scope and estimator

Keep the existing confidence/answer-format validity observations, pooled math/science fitting, two-step planning, and known previous-action context. Add three latent incorrect types alongside one correct state. The terminal rewards are [1, 0, 0, 0]. State names deliberately do not claim proven “repairable” or “retryable” semantics.

Initialization uses cached training outcomes: incorrect nonleaf nodes start in a repair-success type, otherwise a retry-success type, otherwise a neither-success type. Incorrect leaves start uniformly across the three types. These are starting assignments, not permanent labels: one sampled failed repair does not establish that a question is inherently unrepairable. At training roots there are 40 correct, 15 repair-success, 3 retry-only-success, and 42 neither-success examples, so some initial types have little support.

Training performs 20 fixed iterations of correctness-constrained tree sum-product inference and expected-count updates. It fits root priors, observation tables, context-dependent transitions and expected action costs. Gold correctness masks are used only on fitting questions. During replay, decisions use confidence, validity, action history and remaining calls; they never use gold labels or unvisited branches.

Regularization preserves total correct/incorrect prior and transition pseudomass across state counts, uses Laplace observation smoothing, shrinks each context toward pooled tables with strength 10, and gives globally unsupported observations equal likelihood across states. These choices are fixed before this comparison. The two-state control uses the identical estimation pipeline. Empirical shrinkage and calibration are reapplied each iteration; this is not a claim of monotonic likelihood improvement or convergence to a global optimum.

## Protocol and results

Five deterministic, domain-balanced folds partition the 100 training questions: 80 fitting and 20 held out in each fold. All seven branches stay with their question. Compare two/four states at the existing penalties 0.01 and 0.248. Select by held-out training accuracy, then mean tokens, then fewer states, then smaller penalty. The protocol is saved before fitting; selection is saved before loading dev. Only the selected candidate is fitted on all 100 training questions and evaluated on 60 dev questions.

| Hidden states | Lambda | Out-of-fold accuracy | Mean total tokens | Action paths |
|---|---:|---:|---:|---|
| 2 | 0.01 | 57/100 (57%) | 1,304.77 | 97 repair twice; 3 retry then repair |
| 4 | 0.01 | 57/100 (57%) | 1,316.53 | 100 repair twice |
| 2 | 0.248 | 50/100 (50%) | 750.04 | 86 repair once; 14 stop |
| 4 | 0.248 | 46/100 (46%) | 477.22 | 82 stop; 12 repair once; 6 repair twice |

The selected two-state, lambda-0.01 model obtains **32/60 = 53.33% on dev, at 1,253.03 mean tokens**, choosing repair twice on all 60 questions. No four-state dev result or new test result is claimed. All tokens are cached solver input/output tokens along the selected path; offline controller arithmetic uses no LLM tokens.

## Interpretation and next choices

Extra states can represent distinct repair/retry dynamics, but the controller still needs observations that distinguish them. This experiment shows no accuracy gain from this particular four-state estimator. At 0.248 it buys a substantial token reduction by stopping much more often, with lower accuracy. That is a tradeoff, not the requested accuracy improvement.

The training set is small, rare types are weakly supported, and same-question branches may violate conditional independence given the latent parent state. One deterministic initialization and 20 iterations do not establish that all four-state models fail. In fact, four-state training log likelihood still increases by 0.23–1.28 in the last iteration across folds, so the fixed iteration budget does not demonstrate convergence. Any longer or multi-start fitting should be a separately recorded training-only follow-up, not a silent replacement of this run. More latent parameters also change observation regularization even though coarse transition prior mass is matched. Repeated development means these diagnostic results are not untouched confirmatory evidence.

A useful next separately declared experiment is an observable signal such as answer agreement/change after an action, with question-grouped validation and explicit history handling. It is **not implemented or selected here**. More training questions or train-only multi-start fitting are other options. Preserve the frozen test archives; do not repeatedly select models against their scores.

## Artifacts and spending

Protocol, fold assignments, ten fold models, all held-out paths, likelihood traces, selection, final selected model and dev paths are preserved here. Source snapshots and an artifact manifest bind this run to its implementation. This experiment makes no model/API/Modal calls, adds $0 inference spending, and never loads the test split.

Independent validation passed three targeted tests and reproduced all ten fold models, twenty held-out path sets, selection, full-training fit and dev results. The matched two-state control agrees to a maximum parameter difference of 2.22e-16. Ten existing planner tests and six context/calibration tests also passed. All seven prior experiment archive manifests were verified unchanged. See [validation.json](validation.json).
