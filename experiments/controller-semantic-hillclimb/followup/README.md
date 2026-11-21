# Follow-up: larger observers and selective deeper reviews

This area continues the semantic-controller study without changing the frozen
compact solver, initial response, plain repair/retry templates, or two further
solver-call budget. The five fixed references are Stop, Repair once, Repair
twice, Retry once, and Retry twice. The requested accuracy gaps are percentage
points: A and B each exceed the strongest fixed reference by 10 points, and C
exceeds the higher of A/B by 10 points while using less compute than B.

The requested accuracy margins are **not achieved**. The original prospective
confirmation, frozen before its replay and grading, gives these results on all
912 reserved questions:

| Method | Accuracy (%) | Avg. LLM tokens | Compute proxy |
|---|---:|---:|---:|
| Stop | 50.66 | 247.4 | 320.3 |
| Repair once | 56.80 | 670.6 | 868.0 |
| **Repair twice** | **61.29** | 1,178.1 | 1,525.1 |
| Retry once | 54.06 | 593.2 | 768.0 |
| Retry twice | 54.61 | 939.9 | 1,216.7 |
| Controller A (POMDP) | 62.94 | 639.1 | 890.3 |
| Controller B (LLM) | 75.77 | 1,612.3 | 1,782.1 |
| Controller C (hybrid) | 75.11 | 1,424.6 | 1,640.8 |

B exceeds the strongest fixed baseline by 14.47 percentage points. A's gap is
1.64 points. C is 0.66 points below B, with a paired 95% interval of
[-2.08, +0.77] points. C saves 7.93% compute versus B [5.97%, 9.94%]. The
evidence supports a compute reduction, not an accuracy improvement over B.

At these observed reference scores, the requested minimum is 71.29% for both
A and B, and 85.77% for C. The original run's failure audit found that 188 of
C's 227 errors had paid reviews that were all wrong; 37 had a correct paid
review available, and two acquired no review. These hindsight counts suggest
that better expert observations matter alongside gate selection. They do not
estimate gains an implementable controller could recover.

The same-source panel (192 questions) gives B and C exactly 82.29% accuracy;
C uses 6.50% less compute. On the additional transfer panel (720 questions),
B scores 74.03% and C 73.19%, with C using 8.24% less compute. Neither panel
supports the requested ten-point advantage for C.

A separately registered follow-up was selected only on the retired development
data and frozen before its own replay and **before any test grades were opened**.
Its development gain did not transfer: A scores 62.72%, B 75.77%, and C 74.89%
on the same 912 questions; C saves 8.22% compute versus B. The original frozen
run is retained. These are two controller versions on **one shared 912-question
panel**, not 1,824 independent test examples.

The [original audited report](artifacts/confirmation-912-v1/REPORT.md) includes
all fixed references, both source panels, paired intervals, target flags, and
aggregate failure counts. The [follow-up report](artifacts/confirmation-912-context-v2/REPORT.md)
preserves the separately frozen comparison and exact cache-reuse provenance.

Both A and C use a four-state difficulty/correctness POMDP with learned
discriminative observations and a two-call planning horizon. A acquires no LLM
reviews. B uses the independent frozen LLM judge. C buys LLM observations by
their estimated value and updates its belief before acting. The observation
models and subject-specific settings for A and C were selected independently
on development data. All final answers come from the same frozen compact solver.

Development scores were substantially more optimistic: the original selection
gave A 78.33%, B 86.25%, and C 88.75%; the follow-up gave A 78.75%, B 86.25%,
and C 88.75%. Larger observers, richer forecasts, and more paired labels did
not establish the requested held-out accuracy hierarchy. A deeper-review
prototype reached 91.25% on development data at a compute proxy of 3,012.90,
exceeding B's compute budget and therefore failing selection.

## Data and protocol

Only the existing 1,472 training questions fit models, and only the existing
240 development questions select settings. The earlier 800-question panel is
retired. The prospective 912-question confirmation combined the previously ungraded
192-question same-source panel with 720 additional questions: 360 unused GSM8K
questions, 180 GPQA-main chemistry questions, and 180 GPQA-main physics
questions. Both constituent panels and the joint result must be reported. The
joint panel preserves the 50% math / 25% chemistry / 25% physics mixture.

`reserve_panel.py` pins the official GPQA repository release, checks structural
choice quality before inference, excludes recorded IDs and normalized stems,
and selects by fixed hashes. `combine_panels.py` registers the joint panel
before scoring. The raw questions, options, reference answers, generated
rationales, and text-bearing neural caches remain in ignored local storage.
GPQA's authors request that examples stay out of online artifacts. Public
outputs can include code, hashes, aggregate metrics, and sanitized correctness
and cost records.

The larger observation model starts from the pinned 24-layer NLI encoder in
`large-model-pin.json`. Training uses the same grounded inputs and three
question-grouped folds. `refine_large.py` adds a correctness-focused pass with
the same excluded-question folds. `local_supplement.py` downloads fold weights
and scores additional observations locally; labels stay outside its neural
inputs. A fold that trained on a question cannot supply that question's
supplemental prediction.

`binary_core.py` implements an actual two-state correctness POMDP with exact
two-call Bellman backups and optional future paid observations. The original
four-state difficulty/correctness POMDP remains a candidate. The action-specific
calibration trial in `calibrated_pomdp.py` fits numeric and neural observations
with grouped calibration and incoming-action priors. The strongest ordinary B
is retained independently of these changes.

`medium_late_pairs.py` collects 192 prespecified training pairs, including both
wrong and already-correct current answers. `transfer_variants.py` evaluates
every selected pair under matched compact seeds. A deeper review followed by
an answer-first compact repair reached 162/192 correct training answers;
compressed feedback reached 146/192. `medium_pomdp.py` fits the deeper late-review
hybrid with the new held-out-fold observations. Its initial review remains the
ordinary brief review.

`root_pairs.py` expands every question in the earlier 48-question root pilot to
a 192-question training bank, adding 144 hash-selected, state-balanced training
questions. Each receives the paid deeper review, an answer-first guided answer,
and both plain final continuations. `root_medium_pomdp.py` tests a hybrid with
one optional deeper root review; the second decision uses the POMDP. The new
guided template, 128-token cap, and deeper review effort are explicit parts of
this candidate. The compact model weights and ordinary references stay fixed.
`root_controls.py` fills historical guided controls absent from sparse retired
trajectories, keeping them separate from the new paired-training bank.

The matched format control in `root_effort_controls.py` holds the 128-token
answer-first repair prompt fixed: ordinary reviews yield 136/192 correct
training repairs and deeper reviews yield 165/192. The original full guided
repair yields 150/192 on the same questions. These matched training counts
describe transfer into the compact solver, not held-out performance.

`reuse_pairs.py` adds all 384 final repair counterfactuals for retaining the paid
root review after either a guided response or a fresh retry. The eight-state
POMDP in `mentor_pomdp.py` tracks difficulty, current correctness, and hidden
mentor correctness. It updates on the joint neural signal and answer/proposal
agreement, preserving the same two additional solver-call budget. The review
is acquired once; a second guided repair reuses it without a new judge call.

`fusion_pomdp.py` and `subject_fusion_pomdp.py` combine two observation models,
charging both physical encoder passes. They are development alternatives, not
an unreported source of free compute. `refine_all_solver.py` tests additional
already-labeled solver outputs, with low-weight terminal responses and equal
question weight; its three question-grouped folds remain unchanged. None of
these variants may be selected using confirmation outcomes.

The all-solver refinement completed using 17,483 labeled nodes from the same
1,472 training questions. It improved the best cost-eligible C to 213/240 on
development data after subject-specific budget selection; it did not improve A.

`subject_policy_pomdp.py` uses the same observer, learned dynamics, finite
Bellman backups, and accounting, with cost penalties and review thresholds
selected separately for math, chemistry, and physics. Subject is observed
before inference. `development_subject_policy.py` chooses a policy for each
subject and checks the resulting joint C cost against B; it does not fit a
different setting for individual questions. All groups are included in final
validation. Later planning checks used a separate copy of the retired fitting
banks, preserving the running original freeze. `snapshot_ungraded_cache.py`
registered the follow-up before any grading and copied every exact ungraded
response without outcome filtering; compatible encoder predictions were reused
only when their complete model recipe hashes matched.

`full_packet_pilot.py` tests a cheaper repair format without dropping the brief
review's worked solution. On the same 192 matched training questions it yields
159 correct repairs versus 150 for the original guided repair (12 paired wins,
3 losses), with 469.90 versus 574.07 compact-solver tokens. `packet_pairs.py`
collects every follow-up action for the eight-state retained-mentor POMDP in
`packet_pomdp.py`. Its review is the exact ordinary brief review; there is no
deeper LLM call. These training-format gains require development selection and
fresh confirmation before any performance claim.

## Validation

`fixed_cache.py` precomputes the same five fixed-policy responses for every
reserved question without using reference answers. Existing same-source raw
responses are reused exactly and charged per policy use. The entire fixed cache
is hashed before final controller replay, and its prefix must remain unchanged
as adaptive responses are appended.

The `confirmation*` scripts implement development-only selection, source and
weight freezing, independent training refits, blinded paid-prefix replays,
blind collection, final grading, and auditing. Fresh scoring requires all 912
questions to complete under the same freeze. Paired bootstrap resampling
preserves each subject-by-source stratum and recomputes the highest baseline
and higher A/B accuracy inside every draw. Every target gets its own pass/fail
flag.

Each version's development audit independently reconstructed both selected
planners and passed 1,920 charge checks, 1,440 blinded paid-prefix replays, and
480 training-only refit comparisons before replay. Each held-out audit checks
7,296 charges and 5,472 blinded paid-prefix replays. `audit_encoder_panels.py`
also regenerates observations for fixed-hash samples in every source-by-subject
stratum, including GPQA. `error_causes.py` produces post-grading aggregate
diagnostics, never a policy or an achievable-gain estimate. `pack_confirmation.py`
publishes only audited, explicitly allowed numeric outcomes and provenance;
it never copies the raw question banks or neural caches.

The separate engineering audit of the eight-state retained-review protocol
also passed 1,920 charge checks, 1,440 blinded replays, 480 independent refit
comparisons, and 48 regenerated observations. It is a development protocol
check, not an additional fresh result or a reason to select that variant.

The compute proxy includes all solver input/output tokens, all physical review
phases, and actual unpadded encoder token slots:

```
(4 * solver_tokens + 3 * reviewer_tokens
 + encoder_parameter_billions * actual_token_slots) / 3.09
```

Reports also retain raw tokens, logical and physical calls, and the 35B
total-parameter sensitivity. This proxy is separate from measured FLOPs or wall
time. Offline training and data collection are outside the per-question
inference proxy. At most two cloud GPUs are used; completed observer training releases
capacity before inference scales to two containers.

Use `uv` for Python. Dependency pins match the parent README. Runtime files are
under `runs/controller-semantic-followup/`; pretrained and trained checkpoint
digests, grouped-fold contracts, and private-volume provenance are retained.
