# Improvement diagnostic: known action context

This is a separate train/dev-only investigation after the frozen lambda0.248 test experiment. It does not replace any frozen result and performs no test evaluation or paid inference. The user requested further improvement; the assistant chose one targeted model variant before evaluating it.

## Finding: pooled repair dynamics hide diminishing returns

The existing model pools repair transitions across three different situations. The training cache shows that these situations have different repair success rates:

| Current answer came from | Wrong training answers fixed by repair | Wrong dev answers fixed by repair |
|---|---:|---:|
| Initial solve | 15/60 (25.0%) | 9/35 (25.7%) |
| Previous repair | 3/46 (6.5%) | 0/27 (0.0%) |
| Previous retry | 22/69 (31.9%) | 8/35 (22.9%) |

The pooled estimate can overstate the usefulness of a second repair because it also reflects repair after a fresh retry. These are descriptive counts from the cached trees, not independent repeated trials or causal claims beyond the cache. The dev second-repair sample is small; zero observed fixes does not establish a zero population success probability.

## One candidate, declared before evaluation

The candidate retains two hidden states, confidence/answer-format validity observations, pooled math/science data, the original root prior, and exact two-call planning. It conditions transitions, successor observation likelihoods, and expected costs on the **known context**: initial answer, after repair, or after retry. Context is observed action history, not an extra hidden state or an answer-change feature.

Each context probability row is shrunk toward the pooled training estimate with ten pseudo-observations. Each context cost mean uses ten pooled-mean pseudo-samples. Strength 10 was fixed before evaluation; it was not searched. The pooled model includes the context samples, so this is regularization toward a pooled estimate, not independent Bayesian evidence. Only the two existing penalties 0.01 and0.248 were evaluated. No new parameter tuning followed the dev result.

For a chosen action, belief updating uses the parent-context action/emission tables. The next planning step uses the context corresponding to that action. Two analytic tests and an independent audit verify this distinction. All fitted quantities use the same 100 training questions.

## Development result

| Policy at λ = 0.248 | Correct | Accuracy | Mean tokens |
|---|---:|---:|---:|
| Previous pooled POMDP | 32/60 | 53.33% | 820.55 |
| Context-conditioned candidate | 33/60 | 55.00% | 760.85 |
| Fixed repair once | 33/60 | 55.00% | 779.22 |

The candidate repairs once then stops on 58 questions and stops immediately on 2. Relative to the previous POMDP it gains one correct answer and saves 59.70 tokens/question. It matches repair-once accuracy and saves 18.37 tokens/question (about 2.4%). This is a modest dev-only change, not evidence of test generalization or an established adaptive advantage. At lambda0.01 it still repairs twice on every dev question, matching the previous model.

## Important caveat about the two early stops

Both early-stop dev examples have `low:valid` confidence/format observations. That category has **zero training-root examples**. The unchanged original smoothing gives it a correctness belief of about0.488, higher than the observed high-confidence category’s0.440. That behavior comes from pseudocounts, not evidence that low confidence is reliable. Both submitted dev answers are wrong; their repair-once answers are also wrong. Thus the small token saving over repair once relies on these two outcomes and should not be interpreted as strong learned discrimination.

This points to a concrete next issue: confidence categories absent from training need defensible fallback behavior. We did not change that behavior after seeing this result. Any backoff/calibration variant should be specified separately, fit on training only, and evaluated with question-grouped validation rather than selected to fix these two dev examples.

## Answer-change diagnostic: useful in some contexts, not uniformly

Answer agreement was analyzed descriptively using parsed, canonically normalized answers; invalid answers form an unavailable category. It is **not an input to this candidate**. Reference answers are not used to define agreement.

| Previous action | Agreement group | Train current correctness | Dev current correctness |
|---|---|---:|---:|
| Repair | Same answer | 39/74 | 24/42 |
| Repair | Changed answer | 13/23 | 9/17 |
| Retry | Same answer | 25/50 | 19/27 |
| Retry | Changed answer | 6/47 | 6/32 |

In this cache, agreement after retry separates correctness more clearly than agreement after repair. A universal rule that changed answers are worse would discard this context dependence. This supports a history-aware future observation model, but does not yet establish an improved policy.

## Recommended next sequence

1. Validate a fallback/calibration rule for unseen confidence bins using only training questions, keeping all branches from a question together in cross-validation. Do not turn the two dev early stops into a hand-coded rule.
2. Check whether the context-conditioned model remains useful across training folds and reasonable predeclared shrinkage settings; avoid a broad search on the repeatedly used 60-question dev set.
3. If warranted, add answer-change observations through an explicitly history-aware model. This remains a separate extension from the milestone’s chosen confidence/validity-only observations.
4. Investigate four hidden states after the simpler model and observations are reliable; more states alone do not supply new information.
5. Freeze one revised candidate before any further test evaluation. Current test outcomes have already been inspected; new confirmatory evidence would require a fresh holdout.

## Reproduction and evidence

`protocol.json` records the variant and its fixed settings before fitting/evaluation. `model.json` contains pooled and context-specific models with counts. The train/dev dynamics and answer-change tables preserve the diagnostic evidence. Path files contain beliefs, known contexts, action values, scores, tokens, and CPU time. Summary files include all five fixed baselines and the unchanged pooled POMDP at matching penalties. `validation.json` records independent checks. No original model, design record, or frozen archive was overwritten.
