# Further paired-data and review-budget work

This continuation preserves the completed 912-question study. Its A and C
accuracy margins failed; its scored panel is retired and cannot select a new
controller while remaining a fresh test. Original runtime sources and banks
are read-only.

The targets remain percentage points: A and B must each exceed the highest
accuracy of Stop, Repair once, Repair twice, Retry once, and Retry twice by ten
points. C must exceed the higher of A/B by ten points. C must also be cheaper
than B under the declared inference compute proxy. A remains a POMDP with no
LLM review, B remains the independently selected ordinary LLM controller, and
C remains a POMDP with paid observations from the same LLM reviewer. The compact
solver weights, initial prompt, plain repair/retry templates, decoding, and
two-continuation horizon remain fixed.

`reserve.py` registers 512 new training questions, 256 development questions,
and 1,024 untouched test questions, reserving test first. Each has the same
50% math / 25% chemistry / 25% physics composition. Math uses previously unused
GSM8K questions. Science uses the official
[SuperGPQA release](https://huggingface.co/datasets/m-a-p/SuperGPQA), pinned to
revision `4430d4458112c7d4497fdcf94d7cc223313d6acf`. This is a separate source
extension, not another sample from the earlier GPQA/MMLU-Pro population.

Selection uses fixed hashes before any inference. Exclusions include recorded
historical IDs and stems, plus high word-fivegram overlap. Official science
option order and all distractors are retained. No model success, difficulty
label, or desired controller ordering selects questions. SuperGPQA's ODC-BY
attribution and the respective licenses of its referenced datasets apply.
The raw source and all question/reference/response banks remain private and
ignored; only source pins, aggregate results, and sanitized receipts may be
published. Freshness here concerns this study's fitting and selection, not
absence from model pretraining.

`collect.py` collects every prespecified paired branch for train/development,
using the exact ordinary solver and review protocols. It cannot collect test.
The `plain` mode collects all seven plain solver states; `ordinary` adds the
root review, guided states, and all three matched late-review pairs. Completed
wrong responses are retained. Reference answers are blanked before inference.
Every policy must pay for the observations it uses even when replayed from a
training or inference cache. At most two cloud GPUs run concurrently.

`budget_pairs.py` tests 256- and 512-token bounded reasoning budgets on the
entire existing matched 192-question training cohort. Both use a 160-token
answer phase when required and a 128-token answer-first compact repair. All
physical phases are charged. This tests whether the earlier deeper-review
accuracy can be retained at a lower cost; training pairs do not establish a
held-out controller gain.

`refine_stream.py` pipelines local observer fine-tuning with collection. It uses
registered blocks of 16 new questions and fixed old-question replay. Every
question's group is excluded from its prediction model throughout all updates.
All 512 new paired questions must complete before model selection; final
out-of-question outputs are regenerated from the archived, rounded weights.
The process sends no training text or labels to the cloud. `refine.py` provides
the batch alternative, with a synthetic end-to-end exclusion test.

`mentor_pairs.py` registers a mixed 192-question training cohort: 64 fixed-hash
old pilot questions and 128 new questions selected by registered order and
subject quotas. It collects the strong reasoning dose and every retained-review
repair counterfactual, reusing compatible paid controls exactly. These pairs
support a POMDP that explicitly tracks hidden reviewer correctness alongside
question difficulty and current-answer correctness. `blend.py` creates a
separate fitting view without editing either source bank. Candidate dynamics
can also be fitted to just the new source's training questions; this choice
must be made on development data.

Runtime files live in `runs/controller-semantic-followup/round2/`. No new test
scores are available. Use `uv` for Python; dependency pins match the parent.

## Current continuation

The shorter review pilot is complete: on the same 192 training questions,
256-token reasoning produced 123 correct short-guided repairs, and 512-token
reasoning produced 146. The original full-strength review produced 165 with
the same short-guided format. These smaller budgets are not selected.
The actual mixed cohort in `mentor_pairs.py` therefore uses one strong dose:
1,024 reasoning tokens and a bounded 512-token answer phase. The earlier
two-dose registration is preserved privately as abandoned before inference.

`repairability_pomdp.py` fits eight hidden states: current correctness,
correctness after repair, and correctness after retry. All seven paired plain
branches supervise the training labels; online inputs contain only paid
responses. Question-grouped forecast predictions fit the observation model.
The POMDP uses an exact two-step finite-observation Bellman backup.
`continuous_repairability.py` also tests an approximate belief estimator that
blends the finite filter with a discriminative posterior over paid history;
it does not claim an exact continuous-likelihood Bayesian update.

On all 256 new development questions, the first repairability model achieved
141/256 (55.08%), and the continuous variant achieved 142/256 (55.47%). The
strongest fixed baseline, Repair twice, achieved 132/256 (51.56%). Thus the
best A development margin is 3.91 percentage points, below the ten-point goal.
These development results are used for selection and are not fresh test
confirmation. The reserved 1,024-question test bank remains empty.

The training-only hindsight oracle over all seven plain branches is 343/512
(66.99%), versus 271/512 (52.93%) for the strongest fixed baseline. In each
science subject, 80/128 training questions have no correct plain branch.
This is an upper bound on selection, not an achievable controller result.

`prepare_mixed_mentor.py` constructs a separate 576-question fitting view
(all 512 new questions and 64 old controls), with exactly 192 strong review
pairs. Old questions use excluded-fold neural predictions; new questions use
the frozen old-only observer. `hybrid_repairability.py` connects the eight
repairability states to eight retained-review states through a learned
truth-preserving transfer. C can acquire one strong review and reuse it on a
second repair. `development_mixed_mentor.py` keeps these responses in its own
bank. `development_b.py` independently selects the ordinary LLM controller.

Two further changes address the observation and action limits. The local
science observer uses 1,440 additional disjoint public training questions and
160 auxiliary calibration questions; all 1,600 are sealed as historical
exclusions. It learns option correctness without solver rationales. Its fixed
final weights are archived before auxiliary temperature calibration, and
each online observation incurs the actual encoder cost. It does not train
on project development or test outcomes.

The registered 128-question textbook pilot feeds reference-free OpenStax
exposition through the existing repair template's optional feedback field.
Its 64 math cases reuse exact ordinary repairs; the 64 science cases acquire
new compact responses. This is an added controller feedback channel, beyond
gate tuning. Solver weights, decoding, seeds and output cap stay fixed; all
feedback input tokens are charged and CPU retrieval time is reported.

The textbook pilot completed with 65/128 correct versus 67/128 for matched
ordinary Repair once, at higher compute. It is not selected. The correlated
binary-forecast/IPF POMDP also completed: 140/256 development answers correct,
below the continuous joint-state candidate's 142/256.

Additional registered work includes an option-proposal pilot on 64 new science
training questions. After auxiliary training, it scores every official option
and supplies the highest-scored option through the existing repair feedback
field. All candidate encoder passes are charged; this is another new feedback
channel, not a completed controller result.

`crossfit_mentor_emissions.py` refines the hybrid's emission estimates with
question-excluded calibration predictions. It preserves the original runtime
calibration, transition dynamics, costs and belief transfer. Its separate
development replay reuses complete original responses without outcome
filtering. `matched_review_b.py` supplies an independent LLM-only control with
the same strong retained-review protocol as C. Final comparisons retain the
higher accuracy of ordinary and matched-review B.

`prospective_freeze.py` and `prospective_audit.py` provide a staged freeze and
paid-prefix audit, not a complete test collector/scorer. The freeze requires
all development gaps, C's compute advantage, and completed matched-review B;
it rejects choosing a weaker completed A to ease C's target. The audit reports
logical reviews and physical API phases separately. Neither tool has opened
test outcomes or established a fresh-data success.
