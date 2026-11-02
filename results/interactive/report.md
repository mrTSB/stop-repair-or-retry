# Interactive benchmark results

Local MLX/Metal results for ALFWorld.

## alfworld

Train/dev/test counts: 100/50/134.
Retry reset matches: 134/134; paired comparisons exclude mismatches.
Attempt budget: 50 actions; controller checkpoint: 6 actions.

| Policy | Success (95% CI) | Mean tokens | Mean actions | Mean invalid actions |
|---|---:|---:|---:|---:|
| stop | 0.030 [0.007, 0.060] | 3000.4 | 6.0 | 1.09 |
| repair | 0.060 [0.022, 0.104] | 28635.3 | 47.7 | 16.27 |
| retry | 0.030 [0.007, 0.060] | 32203.8 | 55.0 | 14.97 |
| tuned_fixed | 0.030 [0.007, 0.060] | 3000.4 | 6.0 | 1.09 |
| threshold | 0.030 [0.007, 0.060] | 3000.4 | 6.0 | 1.09 |
| one_step | 0.030 [0.007, 0.060] | 3000.4 | 6.0 | 1.09 |
| supervised | 0.030 [0.007, 0.060] | 3000.4 | 6.0 | 1.09 |
| judge_full | 0.030 [0.007, 0.060] | 5599.9 | 9.0 | 2.25 |
| judge_compact | 0.052 [0.015, 0.090] | 27541.7 | 46.4 | 13.75 |
| hybrid | 0.030 [0.007, 0.060] | 3000.4 | 6.0 | 1.09 |

Success is exact benchmark completion. For ALFWorld, reward is binary completion (0 or 1).

### Paired difference from stop

| Policy | Success difference (95% CI) |
|---|---:|
| repair | +0.030 [+0.007, +0.060] |
| retry | +0.000 [-0.045, +0.045] |
| tuned_fixed | +0.000 [+0.000, +0.000] |
| threshold | +0.000 [+0.000, +0.000] |
| one_step | +0.000 [+0.000, +0.000] |
| supervised | +0.000 [+0.000, +0.000] |
| judge_full | +0.000 [+0.000, +0.000] |
| judge_compact | +0.022 [+0.000, +0.052] |
| hybrid | +0.000 [+0.000, +0.000] |

### Success by task family

| Task family | N | Stop | Repair | Retry |
|---|---:|---:|---:|---:|
| look_at_obj_in_light | 18 | 0/18 | 3/18 | 2/18 |
| pick_and_place_simple | 24 | 4/24 | 4/24 | 0/24 |
| pick_clean_then_place_in_recep | 31 | 0/31 | 0/31 | 0/31 |
| pick_cool_then_place_in_recep | 21 | 0/21 | 0/21 | 0/21 |
| pick_heat_then_place_in_recep | 23 | 0/23 | 1/23 | 2/23 |
| pick_two_obj_and_place | 17 | 0/17 | 0/17 | 0/17 |

## Method and limitations

At a fixed checkpoint, stop submits current progress, repair continues the live attempt, and retry restarts the same task instance. All controller policies use cached paired branch outcomes. Fitted models use train trajectories; tuned settings use dev; test is held out. Confidence intervals and per-task results are in the JSON artifacts.

Per-question artifacts retain task IDs, controller actions, scores, and costs; they exclude free-form observations, prompts, and generations. Raw trajectories remain in the local `runs/` directory.

These runs support pipeline validation and feasibility analysis. The local model has limited task success; larger test samples alone will not make this a useful workshop result. Interpret results only for the benchmark(s) listed above.

## ScienceWorld feasibility gate

Current-code gate: 5 held-out tasks, 30-action attempt budget, 6-action checkpoint. Exact completions were stop 0/5, repair 0/5, and retry 0/5. Mean normalized progress was 0.082, 0.082, and 0.082, respectively. Retry resets matched the initial observations on 3/5 tasks.

This gate did not pass: no branch completed a task, and reset mismatches make two task pairs unsuitable for paired claims. The full ScienceWorld run was therefore not launched with this 3B agent. See `scienceworld-feasibility.json` for task IDs, family scores, configuration, and the trajectory hash.
