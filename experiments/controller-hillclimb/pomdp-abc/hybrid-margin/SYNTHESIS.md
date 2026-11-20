The single frozen 800-question evaluation did **not** achieve the requested A/B/C hierarchy. B is the accuracy leader at 85.250%; C reaches 83.125% with 10.08% less active-parameter compute proxy. C loses 17 correct answers relative to B. Its paired accuracy difference is −2.125 percentage points (95% interval −3.375 to −0.875), so this is an accuracy–compute tradeoff.

All five prespecified baselines and the best development-selected controller from each family are shown below. Every method uses the same frozen compact solver and the same initial-response budget, with at most two additional solver calls.

| Method | Correct / 800 | Accuracy (%) | Mean input + output tokens | Compute proxy |
|---|---:|---:|---:|---:|
| Stop | 486 | 60.750 | 256.6 | 332.1 |
| Repair once | 550 | 68.750 | 683.4 | 884.6 |
| Repair twice | 564 | 70.500 | 1188.8 | 1538.8 |
| Retry once | 526 | 65.750 | 607.8 | 786.8 |
| Retry twice | 521 | 65.125 | 959.2 | 1241.7 |
| A — POMDP | 578 | 72.250 | 872.5 | 1129.4 |
| B — LLM | 682 | 85.250 | 1298.9 | 1455.8 |
| C — POMDP + LLM | 665 | 83.125 | 1136.9 | 1309.1 |

Stop submits the initial response. Repair checks and revises the current solution; Retry solves the original question in a fresh context. Once/twice fixes the number of additional solver calls, and the latest response is submitted. The strongest observed fixed baseline is Repair twice at 70.500%. The compact solver and reviewer are shared model components, not additional baseline rows.

The confirmed margins are percentage points. A and B therefore need at least 80.500% here: A misses that target at 72.250%; B passes at 85.250%, exceeding the strongest baseline by 14.750 points (paired 95% interval 12.125–17.250). C needs at least 95.250% to exceed the stronger A/B by ten points. It reaches 83.125%, missing the target by 97 answers. A fifteen-point stretch above this B would require 100.250%, beyond the accuracy ceiling. The targets were retained unchanged in the frozen acceptance tests.

C saves 10.08% compute proxy and 12.47% raw tokens versus B. The paired compute saving is 146.77 units (95% interval 116.16–178.95). C is 12.625 points more accurate than Repair twice with 14.93% less compute proxy, but it does not dominate B. C fixes five cases B misses; B fixes 22 cases C misses. B is more accurate within math, chemistry and physics separately. These are descriptive fresh-set results; no policy was reselected after scoring.

A remains a genuine four-state POMDP over persistent question difficulty and current-answer correctness. A question-grouped observation classifier, learned transition probabilities, Bayesian belief updates and a two-call Bellman backup determine its actions. B is an independent frozen LLM reviewer with a development-selected agreement/reuse rule, without a POMDP. C learns contextual repair/retry success probabilities and buys LLM observations when their estimated value exceeds their cost and selection threshold. B receives the stronger development-selected late reviewed-repair protocol; C independently selects the earlier contextual protocol because it performed better on development. Every final answer comes from the compact solver.

The hillclimb expands training to 672 questions and development to 240, with 18,240 paired observations across the two splits. It tests richer POMDP dynamics and observations, late paid reviews guiding final repairs, review-prompt variants, a cross-fitted neural verifier, and learned review calibration. The strongest independently selected B is retained. The verifier and calibrated-review trials do not improve the selected cheaper C. Their development reports and training receipts are archived, with no ablation rows substituted for the final controllers.

The development-only coverage analysis helps explain the remaining gap. A needs 203/240 correct to clear baseline-plus-ten, while a reference-aware oracle over all seven plain compact-solver paths reaches only 204. Across all thirteen tested paths, including reviewed repairs, the oracle reaches 220/240 (91.667%). That is below the 222 correct needed for C to exceed the selected development B by ten points, and the 227 needed if A/B both qualified. This bounds the cached development response bank, not all future prompts or generations. Reaching the full requested margins requires better correct-answer coverage from allowed continuations and more reliable correctness observations.

The fresh panel contains 400 GSM8K official training-source questions, 200 MMLU-Pro chemistry questions and 200 physics questions. It is a historically disjoint study holdout, not an official GSM8K test result. Prior test panels are explicitly retired into training/development. IDs and normalized stems are excluded before reservation; policies, code, data and model hashes are frozen before inference. No fresh accuracy is inspected until all 800 questions finish. Malformed and truncated completed responses remain included.

The compact solver is Qwen3.5-4B; B/C share Qwen3.5-35B-A3B as reviewer. The reported compute proxy is (4 × solver tokens + 3 × reviewer tokens) / 3.09, using every input/output token and charging reused feedback inside later solver prompts. This is a nominal active-parameter proxy, not measured latency, memory, energy, or total training/collection cost. Total-parameter sensitivity and call counts remain in the machine-readable summary. No neural verifier is used by the selected controllers.

Validation passes 29 focused tests, refits both selected trained models independently, reproduces 480 development outcomes and 720 blinded development controller replays, reconciles all 6,400 fresh outcomes and charges, and repeats 4,800 fresh controller runs with references changed and unpaid branches removed. Feature caches are cleared for blinded replays. Replaying the compressed archive reproduces the final summary and all per-question outcomes byte for byte. The private GPU app is stopped with zero running containers.

Full results, acceptance flags, paired intervals, raw receipts, frozen models, split provenance and checksums are in [the archived evaluation](../../../../results/controller-hillclimb/pomdp-abc/hybrid-margin/README.md). The honest conclusion is that the stronger LLM controller clears the baseline accuracy margin, while the POMDP and hybrid still miss the requested accuracy margins.
