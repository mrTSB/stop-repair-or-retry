# Modal LLM-controller development pilot

Completed November 10, 2026 (Pacific time; November 11 UTC). This is an exploratory ten-question pilot on the project development split, not a final evaluation or a POMDP result. The first ten dev questions were selected before inference: five math and five science. Source IDs containing `gsm8k-train` identify the source benchmark split, not the project split. All ten belong to project dev.

## Result

| Method | Correct | Mean total tokens | Mean utility |
|---|---:|---:|---:|
| LLM controller | 8/10 | 1,657.3 | 0.7834 |
| Submit initial | 5/10 | 326.5 | 0.4967 |
| Repair once | 8/10 | 793.6 | 0.7921 |
| Repair twice | 8/10 | 1,279.8 | 0.7872 |
| Retry once | 3/10 | 678.6 | 0.2932 |
| Retry twice | 6/10 | 1,011.6 | 0.5899 |

The controller matched repair-once accuracy but used 863.7 more tokens per question, so repair once had higher utility under the declared penalty. It stopped immediately on five questions and chose repair then stop on five. It never chose retry or a second solver call. All 15 action responses were valid JSON; there were no unresolved requests, truncations, or invalid-action fallbacks.

The controller preserved all five initially correct answers and fixed three of five initially incorrect answers. One incorrect answer was submitted immediately; one remained incorrect after repair. Both domains finished at 4/5 correct. These counts are too small to establish superiority, and this run does not validate retry or depth-two decision behavior.

Selective repair reduced mean solver tokens from 793.6 for repair once to 562.3, but added 1,095.0 controller tokens, for 1,657.3 total. Input/output tokens include prompt chat templates and EOS. Token counts are a research cost measure across different model sizes, not a conversion to dollars. Cached solver calls incurred no new solver inference bill during this pilot.

## Protocol and evidence

Controller: Qwen/Qwen2.5-14B-Instruct, revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`; bfloat16; greedy decoding; 128-token maximum; no constrained decoding or JSON repair. The existing solver cache uses Qwen2.5-3B-Instruct. The reward is final correctness minus 0.01 times total executed solver/controller tokens divided by 1,000. This penalty was set before this pilot, not selected from its outcomes.

The controller receives only the problem, remaining call budget, penalty, and visited responses. It does not receive gold answers, correctness labels, or unvisited cache branches. It outputs one action, not an explanation or replacement answer. Gold labels below are added only for human inspection after collection. No training, prompt tuning, POMDP, older controller outcomes, or test inference was used.

The run used private authenticated Modal RPC on the gancone account: one A100-80GB, four CPU cores, 48 GiB RAM. Collection took 91.30 seconds including startup; summed decision RPC latency was 8.81 seconds across 15 calls. Solver batch times are not serial latency measurements. Both pilot apps are confirmed stopped with zero tasks.

A first launch failed before model loading or controller decisions because the uploaded module had a different filesystem path. It was stopped, preserved separately, and the path handling was fixed and regression-tested. No question response was repeated or selected among attempts.

The original run directory contains the immutable request/response journal, frozen protocol/configuration, exact formatted controller prompts, pinned backend metadata, completion record, and offline evaluation. `walkthroughs.md` shows every executed path with full solver responses and controller actions. `billing.json` distinguishes posted provider costs from reserved/estimated amounts. The five baselines replay exactly these same ten cached questions; older test-set baseline percentages are not mixed into this table.

## Per-question overview

| ID | Domain | Actions | Initial correct | Final correct | Total tokens |
|---|---|---|---:|---:|---:|
| mmlu-pro-biology-03272 | science | stop | 1 | 1 | 1172 |
| gsm8k-train-05034 | math | repair → stop | 0 | 1 | 1606 |
| gsm8k-train-00787 | math | stop | 1 | 1 | 972 |
| gsm8k-train-01476 | math | repair → stop | 0 | 1 | 2264 |
| mmlu-pro-physics-09594 | science | repair → stop | 0 | 0 | 2447 |
| gsm8k-train-04277 | math | stop | 0 | 0 | 961 |
| gsm8k-train-05236 | math | stop | 1 | 1 | 826 |
| mmlu-pro-chemistry-04300 | science | stop | 1 | 1 | 1105 |
| mmlu-pro-physics-10213 | science | repair → stop | 1 | 1 | 2504 |
| mmlu-pro-physics-10218 | science | repair → stop | 0 | 1 | 2716 |

## Spending and shutdown

Prior project charges were **$3.00059499**. The failed startup currently has zero posted GPU/CPU/RAM charges. The successful pilot charge has not posted yet; absence from the ledger is not zero cost. Its 91.30-second app lifetime implies approximately **$0.078** at the requested resource rates, before any additional charges. We retain **$1.50** for all new charges and uncertainty, putting prior charges plus that reservation at **$4.50059499**, below the $20 cumulative cap. Both apps are stopped with zero tasks. Recheck the provider ledger before another paid run; `billing.json` retains the exact distinction between posted charges, estimates, and reservation.

## Verification

An independent agent reproduced request/response pairing, selected IDs, observed-history payloads, frozen hashes, real token usage, all paths, domain scores, baseline comparisons, and bootstrap statistics. See `validation.json`. The 27 experiment tests and the new remote-import regression test pass. The latter covers the packaging failure from the first attempt. No full-dev freeze or test run was performed.
