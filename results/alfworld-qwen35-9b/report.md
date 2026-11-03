# ALFWorld Qwen3.5 9B fixed-policy follow-up

This follow-up evaluates 40 tasks with local Qwen3.5 9B through Ollama. The separate 10-task pilot used a different selection seed; the two sets are disjoint. The pilot was exploratory. The 40-task sample was added after inspecting that pilot and is also exploratory, not preregistered.

All 40/40 retries reset to the same task instance and matched the initial observation. Exact success is binary ALFWorld completion. The fixed checkpoint is 6 actions into a 50-action attempt.

| Policy | Exact success (95% bootstrap CI) | Paired change vs stop (95% bootstrap CI) | Mean actions | Mean tokens |
|---|---:|---:|---:|---:|
| stop | 2.5% [0.0, 7.5] (1/40) | — | 6.0 | 3027 |
| repair | 40.0% [25.0, 55.0] (16/40) | +37.5 pp [+22.5, +52.5] (wins/losses 15/0) | 34.4 | 22369 |
| retry | 40.0% [25.0, 55.0] (16/40) | +37.5 pp [+22.5, +52.5] (wins/losses 16/1) | 42.0 | 26716 |

## Exploratory 10-task pilot

This pilot is separate from the follow-up and is too small for stable estimates.

| Policy | Exact success (95% bootstrap CI) | Paired change vs stop | Mean actions |
|---|---:|---:|---:|
| stop | 20.0% [0.0, 50.0] (2/10) | — | 6.0 |
| repair | 30.0% [10.0, 60.0] (3/10) | +10.0 pp | 37.2 |
| retry | 20.0% [0.0, 50.0] (2/10) | +0.0 pp | 47.3 |

## Success by ALFWorld task family

| Family | N | Stop | Repair | Retry |
|---|---:|---:|---:|---:|
| look_at_obj_in_light | 4 | 1/4 | 3/4 | 1/4 |
| pick_and_place_simple | 7 | 0/7 | 1/7 | 1/7 |
| pick_clean_then_place_in_recep | 7 | 0/7 | 3/7 | 4/7 |
| pick_cool_then_place_in_recep | 8 | 0/8 | 3/8 | 4/8 |
| pick_heat_then_place_in_recep | 7 | 0/7 | 3/7 | 4/7 |
| pick_two_obj_and_place | 7 | 0/7 | 3/7 | 2/7 |

## Limits

This follow-up contains 40 test tasks and evaluates fixed stop, repair, and retry branches only; it does not include controller train/dev splits or support claims about adaptive policies. The sample was chosen after a separate exploratory pilot, so treat these results as promising but exploratory. The underlying ALFWorld test set and task distribution are public.

The earlier full 3B study used a different 134-task selection and had low absolute completion. These 9B results should not be pooled with it: model, sample, and backend differ. See `summary.json` and `per_question.jsonl` for hashes, costs, per-task outcomes, and paired intervals. Raw trajectories remain under ignored `runs/`.
