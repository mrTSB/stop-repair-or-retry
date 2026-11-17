# Paired retry uplift results

The frozen learned gate scored 198 of 300 on the new test questions, compared with 196 for retry-on-invalid. It used 0.65% less mean estimated compute overall and 9.01% less compute on retries. The paired accuracy difference was +0.67 percentage points, with a 95% bootstrap interval of −1.00 to +2.33 points.

| Policy | Correct | Accuracy | Mean compute | Retries | Initial errors fixed |
|---|---:|---:|---:|---:|---:|
| Compact only | 192/300 | 64.00% | 318.90 | 0 | 0 |
| Retry on invalid | 196/300 | 65.33% | 343.91 | 22 | 4 |
| Learned gate, preselected | 198/300 | 66.00% | 341.65 | 25 | 6 |
| Learned gate without token confidence | 196/300 | 65.33% | 341.80 | 26 | 4 |

Mean compute counts every executed input and output token, scaled to Qwen2.5-3B-equivalent tokens. The learned gate used 263.93 raw tokens per question versus 265.67 for retry-on-invalid. It made 19 science retries and six math retries, while the baseline made 22 science retries. The six math retries fixed four errors. The gate skipped two science fixes the baseline would have made. It preserved all initially correct answers.

The gate used Qwen3.5-4B with the same weights, prompts, generation limits, and answer parsers. The batch allocator credits a fraction of the baseline’s retry input tokens and debits the maximum possible cost of each chosen retry. The runner checked the prompt-token offsets and output caps on every collected retry. The paired bootstrap reflects uncertainty on this test split. The observed accuracy difference is small.

## Training and selection

The 1,400 fit pairs include 480 newly generated pairs and 920 archived pairs, including earlier reported test questions. Across these pairs, a short retry could fix 125 wrong first answers and could overwrite 284 right first answers with wrong ones. Compact token probabilities were available for 700 fit pairs. Neither the 120 fresh development pairs nor the 300 fresh test questions entered the fits.

| Fix prediction measure | Out-of-fold training | Fresh development |
|---|---:|---:|
| Possible fixes | 125/1,400 | 11/120 |
| Learned average precision | 0.212 | 0.294 |
| Invalid-answer flag average precision | 0.117 | 0.180 |

The learned gate gained five correct answers out of fold and one on development relative to retry-on-invalid. The selected policy credits 94% of the baseline’s retry input tokens, targets retaining 85% of predicted invalid-answer fixes, and penalizes predicted regressions 1.5 times as much as it rewards fixes. It was frozen before test retry collection. A fixed plan from the 300 first responses identified the union of 33 retry requests across the baseline, primary gate, and confidence ablation. Each method was charged for the requests it selected.

The archived data reproduce the model freeze, test plan, and scores byte-for-byte in the reference environment. All new inference ran on the local Ollama solver. The compact solver and answer scoring archives passed their existing validation and unit tests. [Training and replay instructions](../../../../experiments/controller-hillclimb/retry-gate/paired-uplift/README.md)
