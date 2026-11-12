# Controller B v2: cost-explicit prompt

Naming follows the project proposal: Controller A is the two-state POMDP, Controller B is the LLM controller.

v1 (the [original run](../llm-controller-full-evaluation/README.md) and the [lambda sweep](../llm-controller-lambda-sweep/README.md)) chose nearly the same actions at every penalty. v2 tests whether stating the cost explicitly changes that. Only the system prompt changed ([`experiments/llm-controller/prompt_v2_template.txt`](../../experiments/llm-controller/prompt_v2_template.txt)). It now:

- states lambda and the estimated additional cost of repair and retry, in tokens and objective units, separately for 2 and 1 remaining solver calls;
- states the controller's own per-decision cost (about 713 tokens) and that continuing with a call remaining triggers one more decision;
- replaces the one-sided "do not stop solely because the solver sounds confident" nudge with two-sided guidance about stopping vs continuing.

The GPU worker (`modal_full_evaluation.py`, unmodified), Qwen2.5-14B-Instruct revision, greedy decoding, 128-token cap, request schema, action parser, replay and scoring are the same as v1. Cost estimates use mean solver tokens per action and depth on the 100 training trees (repair 478 / 492, retry 335 / 340 tokens with 2 / 1 calls remaining) and mean controller tokens per decision from v1's archived dev run (712.6). No test data informed the prompt. The v2 prompt is about 200 tokens longer than v1, so actual v2 decisions cost more than the stated 713; utility always charges actual tokens.

Driver: [`experiments/llm-controller/modal_controller_b_v2.py`](../../experiments/llm-controller/modal_controller_b_v2.py). Each penalty ran `--phase dev` (all 60 dev questions, evaluate, freeze), then dev results were reviewed, then `--phase test` (all 400 test questions under that freeze; refuses if the rendered prompt changed). Private Modal RPC, one A100-80GB.

## Test results (400 questions)

| Lambda | v1 accuracy | v1 tokens | v1 utility | v2 accuracy | v2 tokens | v2 utility | Best fixed baseline (utility) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.01 | 44.25% | 1,616 | 0.426 | 45.00% | 2,004 | 0.430 | repair twice (0.534) |
| 0.1 | 43.75% | 1,613 | 0.276 | 46.00% | 2,029 | 0.257 | repair once (0.444) |
| 0.248 | 45.00% | 1,619 | 0.048 | 45.50% | 2,039 | -0.051 | repair once (0.321) |
| 1 | 43.75% | 1,614 | -1.176 | 47.00% | 2,085 | -1.615 | submit initial (0.022) |

v2 chose the same action path as its own lambda 0.01 run on 386 (0.1), 382 (0.248) and 373 (1) of 400 test questions. Its first decision continued (repair or retry) on 190 questions at lambda 0.01 and 212 at lambda 1, i.e. slightly more often at the higher penalty. One invalid action occurred (lambda 0.01); it is scored as stop and its tokens are charged.

Paired test utility differences, v2 minus baseline (95% paired bootstrap interval; not adjusted for multiple comparisons):

| Lambda | vs submit initial | vs repair once | vs repair twice | vs retry once | vs retry twice |
|---:|---|---|---|---|---|
| 0.01 | +0.071 [+0.041, +0.101] | -0.089 [-0.120, -0.061] | -0.104 [-0.136, -0.073] | +0.054 [+0.002, +0.104] | +0.038 [-0.012, +0.085] |
| 0.1 | -0.071 [-0.103, -0.039] | -0.187 [-0.218, -0.160] | -0.155 [-0.186, -0.124] | -0.057 [-0.110, -0.007] | -0.043 [-0.093, +0.005] |
| 0.248 | -0.329 [-0.365, -0.292] | -0.372 [-0.404, -0.341] | -0.261 [-0.296, -0.229] | -0.265 [-0.321, -0.209] | -0.200 [-0.253, -0.147] |
| 1 | -1.637 [-1.730, -1.547] | -1.310 [-1.392, -1.229] | -0.804 [-0.882, -0.726] | -1.318 [-1.420, -1.222] | -0.997 [-1.093, -0.904] |

## Development results (60 questions)

| Lambda | v1 accuracy / tokens / utility | v2 accuracy / tokens / utility |
|---:|---|---|
| 0.01 | 50.00% / 1,544 / 0.485 | 48.33% / 1,918 / 0.464 |
| 0.1 | 50.00% / 1,543 / 0.346 | 50.00% / 1,933 / 0.307 |
| 0.248 | 48.33% / 1,537 / 0.102 | 48.33% / 1,921 / 0.007 |
| 1 | 50.00% / 1,543 / -1.043 | 50.00% / 1,930 / -1.430 |

## What this shows

Stating costs explicitly, in objective units, did not make this controller cost-sensitive: its decisions remain essentially fixed across a 100x range of lambda, and the longer prompt lowers utility. The controller does discriminate correctness: on test at lambda 0.01, 163 of v2's 190 first-step repair/retry decisions (86%) targeted initially incorrect answers, against a 64% base rate (v1: 138 of 159, 87%). Why cost is ignored is not established here; candidate explanations (single-token decision with no reasoning, correctness-driven preferences that dominate the cost text, no calibrated estimate of repair benefit) are untested. Results are for one prompt revision, one 14B model and one solver cache. Existing test outcomes had already been inspected.

## Files and limits

Each `lambda-*` folder holds `config.json`, `prompt.txt` (the rendered v2 prompt), `protocol.json` (changes, cost provenance, driver/template/prompt/worker hashes), `frozen.json`, `dev/` and `test/` (journals, per-question results, manifests, reports, summaries), per-phase `backend-*.json`, `modal-app-*.json`, `reservation-*.json`, `completion-*.json`, watchdog logs and `collection-*.log`. All eight Modal apps were confirmed stopped. The Modal workspace's metered cost across the v1 lambda sweep and v2 (11 Modal apps) was $3.22, covered by credits. Manifests contain absolute local paths for provenance.
