# Controller B: two-state milestone results

The implemented POMDP uses the user-selected confidence/validity observations, pooled math/science dynamics, and exact two-step planning. It was fitted only on the 100 training trees, checked on all 60 dev questions, frozen, and evaluated on all 400 test questions. No paid inference or older controller model was used. See [the design decision record](../../experiments/pomdp-controller/DESIGN.md) for user choices versus defaults and deferred extensions.

## Primary result: lambda = 0.01

**The learned policy chose repair twice on every dev and test question.** It therefore matches the repair-twice baseline exactly on accuracy and token cost. It outperforms Controller A on these metrics, but this is not evidence of an advantage from adaptive decisions or belief tracking. We preserved this finding rather than changing settings after seeing it.

## Test

| Method | Accuracy | Mean total tokens | Mean utility |
|---|---:|---:|---:|
| Controller B: two-state POMDP | 54.75% | 1,358.3 | 0.53392 |
| Submit initial | 36.25% | 340.1 | 0.35910 |
| Repair once | 52.75% | 833.0 | 0.51917 |
| Repair twice | 54.75% | 1,358.3 | 0.53392 |
| Retry once | 38.25% | 679.6 | 0.37570 |
| Retry twice | 40.25% | 1,020.7 | 0.39229 |
| Controller A: LLM | 44.25% | 1,616.0 | 0.42634 |

Total recorded belief/planning CPU time: 0.1708 seconds across 400 episodes. This is a local timing observation, not a serving-latency benchmark. The POMDP uses zero LLM decision tokens, but computation is not literally free.

| Domain | POMDP accuracy | POMDP mean tokens |
|---|---:|---:|
| math | 68.50% | 1,059.3 |
| science | 41.00% | 1,657.2 |

| Comparator | Mean POMDP minus comparator utility | 95% paired bootstrap interval |
|---|---:|---|
| Submit initial | +0.17482 | [+0.13262, +0.21524] |
| Repair once | +0.01475 | [-0.00025, +0.03210] |
| Repair twice | +0.00000 | [+0.00000, +0.00000] |
| Retry once | +0.15821 | [+0.10570, +0.21078] |
| Retry twice | +0.14162 | [+0.08925, +0.18937] |
| Controller A: LLM | +0.10758 | [+0.07534, +0.14196] |

Intervals use the fixed seed and 2,000 paired question resamples. They are conditional on the fitted model and single solver cache, do not include model-fitting uncertainty, and are not adjusted for six comparisons. The identical repair-twice comparison is identically zero.

## Dev

| Method | Accuracy | Mean total tokens | Mean utility |
|---|---:|---:|---:|
| Controller B: two-state POMDP | 53.33% | 1,253.0 | 0.52080 |
| Submit initial | 41.67% | 325.6 | 0.41341 |
| Repair once | 55.00% | 779.2 | 0.54221 |
| Repair twice | 53.33% | 1,253.0 | 0.52080 |
| Retry once | 41.67% | 650.8 | 0.41016 |
| Retry twice | 40.00% | 975.4 | 0.39025 |
| Controller A: LLM | 50.00% | 1,544.4 | 0.48456 |

Total recorded belief/planning CPU time: 0.0259 seconds across 60 episodes. This is a local timing observation, not a serving-latency benchmark. The POMDP uses zero LLM decision tokens, but computation is not literally free.

## Predeclared sensitivity sweep

These penalties were declared before fitting. They are cost preferences, not candidates selected by maximizing test utility. Compare policies within each penalty; utilities across different penalties have different objectives. Controller A was not re-prompted or evaluated as a tuned policy at other penalties.

| Penalty | Test accuracy | Test mean tokens | Test utility | Test action sequence |
|---|---:|---:|---:|---|
| 0 | 54.75% | 1,358.3 | 0.54750 | {'repair -> repair': 400} |
| 0.001 | 54.75% | 1,358.3 | 0.54614 | {'repair -> repair': 400} |
| 0.01 | 54.75% | 1,358.3 | 0.53392 | {'repair -> repair': 400} |
| 0.1 | 54.75% | 1,358.3 | 0.41167 | {'repair -> repair': 400} |
| 1 | 36.25% | 340.1 | 0.02239 | {'stop': 400} |

The observed policy is repair twice at penalties through 0.1, and immediate stop at 1, on both splits. This coarse, predeclared grid shows a change in cost preference but no observation-dependent behavior on the observed trajectories. It does not establish that the policy is constant for all possible beliefs or intermediate penalties. No extra grid points were selected after seeing test results.

## What was learned and what remains future work

The planner maintains and updates beliefs correctly, but at the declared penalties its action values lead to the same choices across observed episodes. Confidence/validity alone and pooled correctness dynamics did not produce a useful adaptive policy on this cache. This is a descriptive result; it does not prove these signals are universally useless or that a larger state space would fix the problem.

For the final project, the decision record identifies four-state estimation, answer-change observations, domain-specific dynamics, and more training/repeated samples as deferred extensions. Each needs new training/dev justification. Four states require an estimator and defensible interpretation, not simply relabeling one successful branch as inherently repairable. The generic planner accepts arbitrary state dimensions; synthetic four-state unit tests verify interface behavior, not a trained four-state experiment. The one-step ablation was explicitly excluded from this milestone.

## Evidence, limitations, and reproduction

`model.json` contains prior/transition/observation/cost tables plus training counts. There are 100 roots and 300 training edges per action; edges from the same question are correlated. The correctness-only state, pooling over depth/domain, and stationary observation tables are approximations. Pseudocount-one smoothing avoids unjustified certainty but does not remove these limitations. The state labels use the existing frozen automated scoring rule.

Every penalty has saved paths including observations, beliefs, all action values, and actual visited solver costs. Initial tokens count in episode cost but are sunk for planning. No reference answer, current gold correctness, or unseen branch informs an action. Gold correctness is used only for supervised training and evaluation. The five fixed baselines use identical cached questions; Controller A is the frozen archive at its original penalty.

The source/helper, design, model, configuration, dataset, and Controller A archive are hashed. The dev evidence is verified by complete deterministic replay before freezing. `frozen.json` precedes the recorded test start. Existing test outcomes had already been viewed, and the primary hypothesis is not treated as an untouched confirmation experiment. There was no parameter tuning after dev or test.

No Modal app was started and no new paid LLM calls were made. This experiment adds local CPU work only. `validation.json` records the independent audit. `artifact-sha256.json` covers the archive; source and design snapshots are preserved. To reproduce, use the repository revision matching the frozen code and run the offline verification command in the README.
