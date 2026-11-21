# Frozen controller confirmation

The requested joint accuracy and compute targets are **not met** on all 912 reserved questions.

This separately frozen follow-up uses the **same 912 questions** as the retained original confirmation. Its candidates were selected only on retired development data and frozen before their own replay and before any test grades were opened. Exact ungraded solver/judge responses and compatible raw encoder observations were reused, with every virtual policy call still charged. These are not an additional independent 912 test examples.

| Method | Correct | Accuracy (%) | Avg. LLM tokens | Compute proxy |
|---|---:|---:|---:|---:|
| Stop | 462/912 | 50.66 | 247.4 | 320.3 |
| Repair once | 518/912 | 56.80 | 670.6 | 868.0 |
| Repair twice | 559/912 | 61.29 | 1,178.1 | 1,525.1 |
| Retry once | 493/912 | 54.06 | 593.2 | 768.0 |
| Retry twice | 498/912 | 54.61 | 939.9 | 1,216.7 |
| Controller A (POMDP) | 572/912 | 62.72 | 634.7 | 885.4 |
| Controller B (LLM) | 691/912 | 75.77 | 1,612.3 | 1,782.1 |
| Controller C (hybrid) | 683/912 | 74.89 | 1,364.5 | 1,635.6 |

The baselines are the five fixed stop/repair/retry policies. Every method uses the same frozen compact solver and initial response, with at most two additional solver calls. A uses a POMDP with learned observations, B uses the independently selected ordinary LLM judge, and C uses the POMDP with paid LLM observations.

| Required comparison | Observed | Target | Result |
|---|---:|---:|---|
| A above highest baseline | +1.43 pp | +10 pp | Miss |
| B above highest baseline | +14.47 pp | +10 pp | Pass |
| C above higher A/B | -0.88 pp | +10 pp | Miss |
| C compute saved versus B | +8.22% | Above 0% | Pass |

Paired 95% intervals:

- A minus highest baseline: +1.43 pp [-1.10, +3.84].
- B minus highest baseline: +14.47 pp [+11.62, +17.21].
- C minus higher A/B: -0.88 pp [-2.30, +0.66].
- C minus B: -0.88 pp [-2.30, +0.66].
- C compute saved versus B: +8.22% [+6.13, +10.29].

Post-grading failure diagnosis: of C's 229 errors, 7 had no paid review, 36 had a correct paid review available, and 186 had paid reviews that were all wrong. These are hindsight categories, not estimates of achievable controller gains.

## same-source panel (192 questions)

| Method | Correct | Accuracy (%) | Avg. LLM tokens | Compute proxy |
|---|---:|---:|---:|---:|
| Stop | 119/192 | 61.98 | 241.0 | 311.9 |
| Repair once | 122/192 | 63.54 | 651.0 | 842.8 |
| Repair twice | 135/192 | 70.31 | 1,141.3 | 1,477.4 |
| Retry once | 121/192 | 63.02 | 573.9 | 742.9 |
| Retry twice | 120/192 | 62.50 | 908.9 | 1,176.6 |
| Controller A (POMDP) | 135/192 | 70.31 | 587.7 | 820.7 |
| Controller B (LLM) | 158/192 | 82.29 | 1,360.1 | 1,497.0 |
| Controller C (hybrid) | 158/192 | 82.29 | 1,153.8 | 1,387.2 |

## transfer panel (720 questions)

| Method | Correct | Accuracy (%) | Avg. LLM tokens | Compute proxy |
|---|---:|---:|---:|---:|
| Stop | 343/720 | 47.64 | 249.2 | 322.6 |
| Repair once | 396/720 | 55.00 | 675.8 | 874.8 |
| Repair twice | 424/720 | 58.89 | 1,188.0 | 1,537.8 |
| Retry once | 372/720 | 51.67 | 598.4 | 774.6 |
| Retry twice | 378/720 | 52.50 | 948.2 | 1,227.4 |
| Controller A (POMDP) | 437/720 | 60.69 | 647.3 | 902.7 |
| Controller B (LLM) | 533/720 | 74.03 | 1,679.5 | 1,858.1 |
| Controller C (hybrid) | 525/720 | 72.92 | 1,420.7 | 1,701.8 |

## Interpretation and reproducibility

Both source panels were reserved before their scores were examined. The original 192 panel and the additional 720 panel are reported separately as well as jointly. Bootstrap draws preserve the subject-by-source composition and recompute the highest baseline and higher A/B score inside every paired draw.

Fresh means unused in this study's fitting and model selection. Pretraining contamination has not been measured.

Average LLM tokens include every solver and physical review input/output phase. The compute proxy additionally charges each actual encoder token slot at its parameter count: `(4*solver_tokens +3*review_tokens +encoder_billions*actual_slots)/3.09`. Offline training and response collection are outside this per-question inference proxy. Raw tokens/calls and the 35B total parameter sensitivity are retained in the JSON metrics.

Selected C protocol: `late`. C uses the ordinary review and guided-repair protocols, with every acquired review charged. The original fixed references and strongest ordinary B are retained.

Where the frozen settings include `by_subject`, the POMDP cost penalty and review threshold were selected separately by observed subject on development data; individual questions never select their own setting.

The audit checks independent cost accounting, blinded replay using paid prefixes only, training-only model reconstruction, frozen source and weight hashes, and regenerated neural observations. The fixed baseline response prefix remains unchanged.

An additional independent encoder check samples every source-by-subject stratum using fixed question hashes, including the GPQA transfer panel. It replays only paid inputs with reference answers removed.

Raw GPQA questions, options, answer strings, generated rationales, and text-bearing neural caches remain private. This artifact contains aggregate metrics, sanitized correctness/cost decisions, model provenance and audit receipts. Trained observer weights remain in the private Modal volume named in the frozen manifest.
