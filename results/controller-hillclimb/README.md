# Controller hillclimb results

This experiment uses the published 100/60/400 math and science split. Policy settings and regressions were selected using out-of-fold training and development results. The original-tree controllers use only cached Qwen2.5 solver outputs for submission. The compact-first experiment adds independent Qwen3.5 responses in `data/` and can submit those responses.

## Test results

| Method | Correct out of 400 | Accuracy | Mean estimated compute |
|---|---:|---:|---:|
| Submit immediately | 145 | 36.25% | 340.1 |
| Repair once | 211 | 52.75% | 833.0 |
| Repair twice | 219 | 54.75% | 1,358.3 |
| Retry once | 153 | 38.25% | 679.6 |
| Retry twice | 161 | 40.25% | 1,020.7 |
| Fixed retry then repair | 221 | 55.25% | 1,173.2 |
| A, trace planner on original trees | 214 | 53.50% | 675.9 |
| B, always acquire 4B verifier | 216 | 54.00% | 1,216.6 |
| C, gated 4B verifier | 221 | 55.25% | 1,059.0 |
| Short-first controller | 249 | 62.25% | 324.5 |
| Compact-only | 261 | 65.25% | 320.1 |
| Compact with retry on invalid answer | 270 | 67.50% | 346.6 |
| Compact-first controller | 274 | 68.50% | 439.0 |

Mean estimated compute uses parameter-weighted Qwen2.5-3B-equivalent tokens per question. The compact-first controller uses 362.9 raw tokens per question and submits the compact answer immediately on 332 of 400 questions. It scores 71.50% on math and 65.50% on science. Repair twice scores 68.50% on math and 41.00% on science.

| Comparator for compact-first controller | Accuracy gain, percentage points | Paired 95% interval | Compute change |
|---|---:|---:|---:|
| Repair once | +15.75 | +10.50 to +21.00 | 47.3% lower |
| Repair twice | +13.75 | +8.50 to +19.00 | 67.7% lower |
| Fixed retry then repair | +13.25 | +8.00 to +18.25 | 62.6% lower |
| Compact-only | +3.25 | +1.50 to +5.25 | 37.2% higher |
| Compact with retry on invalid answer | +1.00 | −0.75 to +2.75 | 26.7% higher |

Most of the gain comes from the new single-call Qwen3.5-4B solver and compact prompt. Compact-only gains 10.50 points over repair twice (paired 95% interval +5.00 to +16.00) and uses 5.9% less estimated compute than submit immediately. A short retry on the 28 invalid compact answers recovers nine further correct answers. The learned controller recovers four net correct answers beyond that simple rule. Its paired interval against the rule includes zero. The short-first controller beats the original fixed points on both reported metrics and trails compact-only on both metrics.

On the original trees, A gains 0.75 points over repair once (paired 95% interval −1.75 to +3.25) at 18.9% lower estimated compute. C acquires verifier evidence on 190 questions and matches the accuracy of fixed retry then repair at 9.7% lower compute. Against repair twice, C gains 0.50 points (paired 95% interval −1.75 to +2.75) at 22.0% lower compute. The old full-history B scored 44.25% at approximately 5,541 equivalent tokens on the same questions. The new B improves by 9.75 points and uses approximately 78% less estimated compute than the old B. The old point uses archived token counts from `origin/codex/llm-controller-experiment` at `3de0fef` and the published 14.7B parameter count. [Qwen2.5-14B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)

## Data audit

The published cache passed its hash, split separation, parsing, grading, prompt lineage, and fixed-baseline checks. On training, 27 of 50 science questions have zero correct nodes across all seven cached possibilities. On development, 12 of 30 science questions have zero correct nodes. The compact first answer solves 18 of those 27 training questions and three of those 12 development questions. On test, the compact first answer solves 46 of the 116 questions with zero correct nodes in the original trees, including 37 science questions. Across the original 1,120 training and development response nodes, 930 report high confidence.

## Selection evidence

| Method | Training out-of-fold accuracy | Training mean compute | Dev accuracy | Dev mean compute |
|---|---:|---:|---:|---:|
| A, trace planner | 53% | 669 | 53.33% | 653 |
| B, always acquire 4B verifier | 58% | 1,235 | 56.67% | 1,144 |
| C, gated 4B verifier | 58% | 1,060 | 60.00% | 1,036 |
| Compact-only | 67% | 314 | 60.00% | 319 |
| Compact-first controller | 71% | 385 | 65.00% | 459 |
| Short-first controller | 62% | 321 | 60.00% | 303 |

The compute column shows parameter-weighted Qwen2.5-3B-equivalent tokens per question. All visited model calls count towards it. The gates and regressions are locked in `frozen.json`, `compact/frozen.json`, and `short/frozen.json`. The compact-first controller learns when to submit, take a short independent retry, or use the original solver and optionally repair. The short-first controller reverses the two new response formats and targets the original submit-immediately compute budget. All variants use the same scoring functions and at most two additional solver calls after their respective first answer.

The test questions come from a previously reported split. The paired intervals reflect question sampling within this specific cache and set of model responses. The compute proxy charges all executed calls and approximates model arithmetic from parameter counts and input/output tokens. All new inference ran on the local Ollama 0.19.0 server. The archived formats took 1,280 offline collection calls across train, development, and test. All three frozen test replays reproduced byte-for-byte from the compressed sidecars. [Replay instructions](../../experiments/controller-hillclimb/README.md)
