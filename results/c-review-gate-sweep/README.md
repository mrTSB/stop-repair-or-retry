# C review-gate cost sweep

Question: can the latest C (subject-routed POMDP + paid 35B reviewer, `bayesian_model_hybrid.routed`) keep most of its accuracy while buying the reviewer less often?

C already has a value-of-information gate (`stronger_base_hybrid.py:156-157`, `bayesian_model_hybrid.py` `evaluate`): it queries the reviewer only if `information_value > threshold`, and `penalty` charges `penalty * tokens / 1000` (reviewer tokens included) inside its own planning. The selected C uses `penalty = 0` everywhere and `threshold = 0` (chemistry, math) / `0.08` (physics), so the reviewer is effectively free to the planner and is queried on 88% of dev questions.

## Method

- Input: the `abc-runtime-2026-11-25` release, unpacked on a Modal CPU container (`experiments/c-review-gate-sweep/modal_sweep.py`).
- `sweep_c.py` loads C exactly as `scripts/reproduction/replay_latest.py` does (same registration, caches, `cached_model` contexts, `no_test_reads`, `no_model_calls`).
- Sanity check first: the selected settings reproduce the archived per-question items (`path`, `correct`, `compute`) for all 256 dev questions: **177/256, mean compute 3,728.58**, identical to the checkpoint.
- Then, for each subject, the expert's `settings` are replaced by every `(penalty, threshold)` in 8 penalties `{0, 0.01, 0.03, 0.1, 0.248, 0.5, 1, 2}` x 11 thresholds `{0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5}` (264 subject-combos). Models, observers, and cached responses are unchanged: no refitting, no new inference, no test reads.
- Compute is the same proxy as the checkpoint (`item['compute']`). `analyze.py` builds the frontiers; its full output is `frontier.txt`; per-question outcomes for every combo are in `sweep.json.gz`.

## Results (256 dev questions)

| Method / setting (penalty, threshold) | Correct / 256 | Accuracy | Compute | vs C | Review rate |
|---|---:|---:|---:|---:|---:|
| Best fixed baseline (repair twice) | 132 | 51.6% | 1,711 | -54% | 0 |
| A (POMDP, no reviewer) | 153 | 59.8% | 1,087 | -71% | 0 |
| Selected C (per-subject) | 177 | 69.1% | 3,729 | - | 0.88 |
| C, uniform (0.01, 0) | 174 | 68.0% | 3,170 | -15% | 0.67 |
| C, uniform (0.01, 0.06) | 170 | 66.4% | 2,730 | -27% | 0.49 |
| C, uniform (0.03, 0) | 166 | 64.8% | 2,503 | -33% | 0.50 |
| B (35B reviewer LLM controller) | 179 | 69.9% | 3,858 | +3% | 1.00 |

"Uniform" means one `(penalty, threshold)` for all three subjects. Allowing a different setting per subject (exact Pareto over all per-subject combinations) gives, for example: 176 at 3,185 (-15%), 175 at 3,004 (-19% vs C, -22% vs B), 170 at 2,574 (-31%), 162 at 2,073 (-44%). See `frontier.txt`.

Per subject (Pareto points, from `frontier.txt`):
- Math (128): the reviewer adds little. 117 correct with no reviews at 621 compute vs 123 at 2,618 with 84% reviews.
- Chemistry (64): the reviewer is necessary. 14 correct with no reviews vs 28 at 94% reviews.
- Physics (64): in between. 16 at 42% reviews (2,747) vs 26 at 89% (5,063).

## Takeaways

- The gate works: reviews can be cut roughly in half for a few questions of accuracy, and at matched accuracy C becomes cheaper than B.
- No setting in the grid exceeds 177, so C still does not beat B's 179 on accuracy. Differences of 2-4 questions on 256 are within noise in either direction.
- Below roughly 1,300 compute, A dominates C (A: 153 at 1,087; the cheapest C reaching 153 costs 1,271). C's unqueried branch is A-like but uses the swept `penalty`, which differs from A's own tuned settings.
- Most of the savings come from skipping reviews on math.

## Caveats

- Everything is on the 256-question development panel, which has already been used repeatedly for selection. Per-subject settings are additionally selected here, so they are more optimistic than the uniform curve.
- The grid is coarse; only these 88 settings per subject were tried.
- No result here is a test result. The reserved 1,024-question test set still has no collected responses.

## Reproduction notes

- `modal run --detach modal_sweep.py --run-name run1` from `experiments/c-review-gate-sweep/` (needs a Modal account; 8 CPU / 16 GiB / 512 GiB ephemeral disk; about 15 minutes end to end; three runs including two failed ones cost $0.55 in total).
- The archive's `.venv` is macOS-built and makes `uv run` fail on Linux (`Exec format error`), even with `--isolated`. The launcher instead builds a clean venv from `scripts/reproduction/requirements.txt` in the image and calls its Python directly.
- `scripts/reproduction/replay_latest.py` failed on Linux in its A section with `ValueError: Cached input differs from the exact paid question/response` (`development_nonlinear.py:91`, via `development_completion_pomdp.cached_observer`). The C-only path in `sweep_c.py` reproduced exactly. The A-section failure was not investigated further.
