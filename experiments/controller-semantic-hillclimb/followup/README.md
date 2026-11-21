# Follow-up: larger observers and selective deeper reviews

This area continues the semantic-controller study without changing the frozen
compact solver, initial response, plain repair/retry templates, or two further
solver-call budget. The five fixed references are Stop, Repair once, Repair
twice, Retry once, and Retry twice. The requested accuracy gaps are percentage
points: A and B each exceed the strongest fixed reference by 10 points, and C
exceeds the higher of A/B by 10 points while using less compute than B.

The targets remain unmet. These are current **development** results on the same
240 retired questions, before the root-review expansion finishes:

| Method | Accuracy (%) | Weighted token compute proxy |
|---|---:|---:|
| Stop | 62.08 | 330.81 |
| Repair once | 70.00 | 887.21 |
| Repair twice | 69.17 | 1,548.59 |
| Retry once | 65.42 | 786.64 |
| Retry twice | 65.00 | 1,244.51 |
| Best A so far | 75.83 | 739.59 |
| Strong ordinary B | 86.25 | 1,624.98 |
| Best C so far below B's compute | 87.92 | 1,461.19 |

Deeper late reviews have reached 90.00% on development data at a compute proxy
of 3,175.76. That candidate fails the compute target. Development and training
pilot gains cannot establish a fresh-test win.

## Data and protocol

Only the existing 1,472 training questions fit models, and only the existing
240 development questions select settings. The earlier 800-question panel is
retired. A prospective 912-question confirmation combines the still-ungraded
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
