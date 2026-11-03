# MMLU-Pro science baseline

Model: `qwen3.5:9b` via `ollama`.
Held-out test size: 300 questions (biology, chemistry, physics).

This is a local science-subset experiment, not an official MMLU-Pro leaderboard score. The source release has a large public test split but only 70 validation questions; we formed disjoint study-specific controller-train/dev/test partitions from the official test file. The model saw the question and options only; MMLU-Pro reference chain-of-thought was excluded.

## Stop, repair, and retry results

| Policy | Accuracy (95% CI) | Paired change vs stop (95% CI) | Mean tokens |
|---|---:|---:|---:|
| stop | 56.3% [50.7, 62.0] | +0.0 pp [+0.0, +0.0] | 485 |
| repair | 65.0% [59.7, 70.0] | +8.7 pp [+5.0, +12.3] | 1143 |
| retry | 53.3% [47.3, 59.0] | -3.0 pp [-6.3, +0.0] | 971 |
| schedule | 65.0% [59.7, 70.0] | +8.7 pp [+5.0, +12.3] | 1143 |
| threshold | 66.3% [61.0, 71.3] | +10.0 pp [+6.7, +13.3] | 809 |
| one_step | 65.0% [59.7, 70.0] | +8.7 pp [+5.0, +12.3] | 1143 |
| supervised | 66.0% [60.7, 71.0] | +9.7 pp [+6.3, +13.3] | 816 |
| pomdp | 65.0% [59.7, 70.0] | +8.7 pp [+5.0, +12.3] | 1143 |

## Accuracy by subject

| Policy | Biology | Chemistry | Physics |
|---|---:|---:|---:|
| one_step | 86.0% | 49.0% | 60.0% |
| pomdp | 86.0% | 49.0% | 60.0% |
| repair | 86.0% | 49.0% | 60.0% |
| retry | 79.0% | 35.0% | 46.0% |
| schedule | 86.0% | 49.0% | 60.0% |
| stop | 83.0% | 36.0% | 50.0% |
| supervised | 87.0% | 49.0% | 62.0% |
| threshold | 88.0% | 49.0% | 62.0% |

## Cost-aware controllers vs fixed repair

| Policy | Paired accuracy change vs repair (95% CI) | Mean token savings vs repair |
|---|---:|---:|
| threshold | +1.3 pp [+0.3, +2.7] | 29.2% |
| one_step | +0.0 pp [+0.0, +0.0] | 0.0% |
| supervised | +1.0 pp [+0.0, +2.3] | 28.6% |
| pomdp | +0.0 pp [+0.0, +0.0] | 0.0% |

## Method

The one-checkpoint tree generated 1 decision level(s) at temperature 0.2 with a 256-token output limit. Train/development/test questions are disjoint by stable question ID and balanced across the three subjects. Confidence intervals use the repository's per-question bootstrap; differences are paired against stop.

The result is conditional on the model, prompt, and public benchmark. It does not measure multimodal science reasoning, experimental design in an interactive simulator, or general scientific discovery.
