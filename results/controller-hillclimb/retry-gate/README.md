# Retry gate follow-up

| Test set | Policy | Correct | Accuracy | Mean compute |
|---|---|---:|---:|---:|
| Published, 400 questions | Retry on invalid | 270 | 67.50% | 346.60 |
| Published, 400 questions | Learned compact-first gate | 269 | 67.25% | 336.12 |
| Published, 400 questions | Learned first-call routing | 269 | 67.25% | 340.19 |
| Fresh, 180 questions | Retry on invalid | 122 | 67.78% | 347.91 |
| Fresh, 180 questions | Learned compact-first gate | 123 | 68.33% | 338.05 |
| Fresh, 180 questions | Learned first-call routing | 123 | 68.33% | 337.93 |

Compute is estimated in Qwen2.5-3B-equivalent tokens, with all input and output tokens on the executed path charged. The learned compact-first gate saved 3.0% on the published test set and 2.8% on the fresh one. The paired 95% intervals for its accuracy change were −2.00 to +1.50 points on the published set and −1.67 to +3.33 on the fresh set. The data support near-baseline accuracy at lower mean estimated compute on these two replays. The gate gained five math answers and lost six science answers on the published set. It shifted retries toward cheaper math questions, where the baseline never retried, while giving up some successful science retries.

The baseline's mandatory compact pass accounts for 320.06 of 346.60 equivalent tokens on the published set. An oracle that can identify correct answers from the two existing responses and extracted alternatives can get at most 294 of 400 correct at lower compute with compact first, or 138 of 180 on the fresh test. That bound charges every retry and spends additional tokens only on the cheapest certain improvements. It shows the room available for better evidence or candidate selection within the two frozen prompts.

## Development findings

| Science retry prompt | Correct out of 30 | Correct among five invalid initial answers | Mean retry input tokens |
|---|---:|---:|---:|
| Existing short retry | 14 | 1 | 233.4 |
| Abbreviated choices and full question stem | 7 | 0 | 209.3 |
| Shortened question and choices with initial scratch work | 13 | 1 | 271.8 |

The tested shortened choices sacrificed too many correct answers for their token savings. Including initial scratch work raised the retry cost with the same one recovered invalid answer. Both pilots used the same Qwen3.5-4B weights and compact first answers.

The token-probability pilot exactly reproduced 220 previously cached compact responses and their token counts. Mean answer-token probability ranked compact-answer correctness with an AUROC of 0.74 on sampled training questions and 0.78 on development where that probability was available. The compact-first gate with confidence features tied the invalid baseline at 37 of 60 correct on development. A gate restricted to low-confidence or invalid answers reached 38 of 60 with lower mean compute, while its out-of-fold accuracy was 105 of 160 versus 106 for the invalid baseline. Its accuracy gain was therefore limited to development. The archived pilot includes the inputs and all three matched ablations.

## Evidence

`frozen.json` records 280 fit questions, 60 development questions, and the selected model weights. `published/` and `fresh/` retain per-question actions, token accounting, and paired intervals. The sample contained 90 math and 90 science questions in each new split. The fitted test gates were frozen before scoring either test set. The confidence pilot uses training and development responses. All new generation ran on the local Ollama model. The two scored runs reproduced byte-for-byte from the archived inputs. [Code and replay instructions](../../../experiments/controller-hillclimb/retry-gate/README.md)
