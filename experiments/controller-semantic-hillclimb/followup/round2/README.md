# Further paired-data and review-budget work

This continuation preserves the completed 912-question study. Its A and C
accuracy margins failed; its scored panel is retired and cannot select a new
controller while remaining a fresh test. Original runtime sources and banks
are read-only.

The targets remain percentage points: A and B must each exceed the highest
accuracy of Stop, Repair once, Repair twice, Retry once, and Retry twice by ten
points. C must exceed the higher of A/B by ten points. C must also be cheaper
than B under the declared inference compute proxy. A remains a POMDP with no
LLM review, B remains the strongest independently selected LLM controller, and
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

## Latest completed comparison

All rows below use the same 256 development questions. These questions have
been used repeatedly for selection; this table is not fresh confirmation.

| Method | Correct / 256 | Accuracy | Mean compute proxy |
| --- | ---: | ---: | ---: |
| Stop | 110 | 42.97% | 381.39 |
| Repair once | 129 | 50.39% | 990.21 |
| Repair twice | 132 | 51.56% | 1,711.21 |
| Retry once | 127 | 49.61% | 886.56 |
| Retry twice | 123 | 48.05% | 1,394.69 |
| Best A: nonlinear-observation POMDP with terminal Complete action | 150 | 58.59% | 986.26 |
| Best B: strong retained-review LLM | 177 | 69.14% | 3,772.86 |
| Best C: expanded-data POMDP + LLM | 177 | 69.14% | 3,747.96 |

The proxy counts all paid solver, reviewer, and observer tokens weighted by
active parameters, normalized to 3.09 billion parameters. Both physical phases
of a bounded review are billed. This is not measured FLOPs or wall-clock time;
the full-35B reviewer sensitivity is also retained in the receipts.

A is 7.03 percentage points above the strongest original baseline, B is 17.58
points above it, and C is 0.00 points above the stronger of A/B. A needs at least
158 correct answers and C needs at least 203 on this panel. C saves only 0.66%
compute relative to B. The requested accuracy gaps are not met.

Complete is a separate answer-changing action with matched controls. The best
such simple control, Repair then Complete if invalid, obtains 150/256 (58.59%)
at 1,232.71 units, using exactly the same retained math A as the new POMDP.
It matches best A's accuracy; A uses 19.99% less compute. The original five baselines
remain unchanged, and all six new fixed/validity controls remain reported.
The separately reserved 1,024-question test set has not been inferred or graded,
and its labels have not been used for controller fitting or selection. Earlier
reservation/exclusion utilities decoded full protected records while using IDs
and question text for deduplication (`reserve_science_aux.py` and
`science_auxiliary_data.py`). Therefore "never opened/decoded" is too strong.
The later capacity audit projected IDs and public-source stems only, and the
current training/development commands explicitly block test-question reads.

`regularized_completion_pomdp.py` and `fit_regularized_completion.py` retain
the exact sixteen-state dynamics and finite Bellman backup. Nested whole-question
training folds select regularization for raw130 features and raw130 plus the
existing encoder's project256 representation. The richer observation uses the
same encoder forward, with actual trimmed input slots charged at runtime.
`development_regularized_completion.py` evaluates sixteen fixed candidates on
all256 development questions after sealing all384 science-prefix observations.
Every original decision/cost and all five original/six matched controls reproduce
exactly. The selected richer model retains146 correct while reducing the compute
proxy by12.97% from the previous best A; it does not improve the accuracy gap.
This proxy does not measure added CPU classifier latency or real FLOPs.

`auxiliary_completion_features.py`, `fit_auxiliary_completion.py`, and
`development_auxiliary_completion.py` prepare the next expansion to768 paired
science training questions. All5120 new compact responses must finish and seal
before feature extraction or label joining. The old V1 encoder excludes all
auxiliary questions; the auxiliary-trained V3 encoder is not used for in-sample
features. The1536 new prefix observations share raw20/project256 forwards, and
three outer and three inner whole-question folds prevent sibling-branch leakage.
Both feature families and the previous stronger A remain in the comparison.
The512-question collection is now complete:5120 physical calls and5120 logical
responses, zero recorded transport errors,2,496,543 input tokens and719,648 output
tokens. These are offline data-acquisition costs, not per-question policy costs.
All1536 new observation passes, nested fits, and sixteen development candidates
are complete. The dense386 model at the single global science penalty0.02
improves A to150/256 (58.59%) at1,057.84 compute, a1.56-point accuracy gain over
the previous146/256. The cheaper146/256 model at943.95 remains a retained
tradeoff. The accuracy target still requires eight additional correct answers.

`diagnose_factorized_completion.py` tests four independent binary correctness
heads as another sixteen-state belief model. Nested TRAIN marginal Brier scores
improve slightly, but joint log loss worsens from1.166 to1.614 for raw130 and
from1.159 to1.622 for dense386. Both fail the registered joint-loss guard, so
neither is promoted to development. `quantile_completion_pomdp.py` instead
retains the joint forecaster and tests TRAIN-derived observation thresholds;
its original-threshold parity tests reproduce all old dynamics and charges.
All sixteen quantile candidates have now completed development evaluation.
They do not improve the retained 150/256 model's accuracy or compute, so the
previous model remains selected. The full round2 regression passes 502 tests.

The subsequent nonlinear categorical observation study passes its nested TRAIN
screen. Both shallow forest procedures improve joint loss from 1.15856 to about
1.1479; the dense procedure reduces mean marginal Brier from 0.13098 to 0.12922.
The paired uncertainty intervals still include no improvement. All sixteen
development candidates are retained. The selected forest-observation POMDP
keeps 150/256 correct and lowers the compute proxy to 986.26, a 6.77% saving
against the preceding 150/256 model. It still has sixteen hidden states and an
exact finite Bellman backup. The forest's CPU work is outside the token and
parameter proxy; this does not establish a wall-clock speedup.
On all 384 cached science decision nodes, five single-thread CPU timing passes
measure about 1.890 ms per forest forecast versus 0.074 ms for the preceding
categorical head. These timings include probability normalization but exclude
feature construction, the encoder, and Bellman planning.

`science_review_critic.py` evaluates every original TRAIN science root review.
The final V3 option observer excludes these 256 questions from neural training.
Whole-question-held calibration improves review-correctness AUC from 0.5478 to
0.6282 and Brier from 0.2320 to 0.2233. Both the option-only and combined critics
pass the declared prediction screen; their Brier-difference intervals include
zero. Their separately fitted hybrid POMDPs use TRAIN-derived review bins and
charge each observer forward. All 52 development candidates finish, but peak
at 176/256, below retained C's 177/256. They are not promoted.

The first critic replay stopped on a missing archived old-observer value.
`development_critic_review_extended.py` preserves that run and supplies all
seven missing old-observer passes. It reuses the 128 completed new-critic
passes and imports 56 missing branches from already sealed canonical solver
receipts or exact retry aliases. No new solver/reviewer call was needed. The
previous C and every math decision and charge reproduce exactly.

All checked cached first-phase reviews exhausted the 1,024-token reasoning
cap. A new fixed 64-question TRAIN pilot tests 4,096 thinking tokens with the
same root prompt, reviewer weights, seeds, and compact repair protocol. It
allows at most 256 physical calls and two existing-app cloud GPUs. Novelty is
measured against the prior plain, Complete, retained and late paths plus 400
answers from six additional completed pilot banks. Its acquisition screen
requires at least six newly solvable questions and a trajectory cost ratio at
most 2.5. This relaxed pilot screen is not the final efficiency criterion:
any promoted C must still cost less than the strongest matched B, which must
be offered the new protocol too. No fresh test inference or grading is started.
The full round2 regression passes 515 tests in 34.07 seconds.

The longer-review pilot completed all 64 questions with 240 physical calls.
It obtains 36/64 correct at both compact endpoints, versus 32/64 for retained
short review, but costs 9,886.07 units for the second endpoint (81.62% more).
It adds no questions beyond the full registered 50/64 prior union, below the
six-question screen. No expansion or C promotion follows. The cloud app was
returned to zero minimum containers, retaining the two-GPU maximum.

A separate fixed half-categorical/half-forest belief ensemble slightly reduces
joint TRAIN loss but worsens mean marginal Brier from 0.12922 to 0.12955. It
fails the declared two-metric screen and is not promoted to development.

A nested training-only correctness-sensor diagnostic slightly improves joint
loss (1.09839 to 1.09620) and current-answer Brier (0.09543 to 0.09525).
Its paired intervals include zero. The resulting A remains an exact sixteen-state
POMDP with the same 768-question transition fit; the additional V3 sensor and
calibrator use only the original 256 science training questions excluded from
V3 neural training. All 199 valid science-prefix sensor passes and eight development candidates
are now complete. They peak at 148/256, so the retained 150/256 A remains
selected. Every visited sensor pass is charged. The original math decisions,
charges, and original/matched controls reproduce exactly.

A bounded two-setting boosted categorical model also fails its nested TRAIN
screen: joint loss rises from 1.14784 to 1.18864 and mean marginal Brier from
0.12922 to 0.13208. It receives no development promotion.

A separate Complete16/mentor8 hybrid now joins the current forest A to the
retained LLM-review dynamics using a truth-preserving transfer learned from
all 256 original science training questions. This explicitly changes the
unqueried mode from eight to sixteen states. The first registered fit stopped
on a missing legacy `state_bits` receipt field; its files remain preserved.
Version two verifies the actual regularized-A schema and shared state core.
All 26 development candidates complete and peak at 176/256, below retained
C at 177/256. The strongest matched B, all baselines, prior C, and exact math
decisions and costs are preserved. No new inference call was needed. Its current
answer-availability ceiling remains below the final C target; this is an
incremental integration experiment. The conditional binary-chain joint model also fails its nested training screen:
joint loss rises to 1.26176 and mean marginal Brier to 0.13571, compared with
1.14784 and 0.12922 for the retained forest. No development promotion follows. All 530 regression tests pass in 34.63 seconds.

The next data expansion is collecting a second disjoint 512-question block
from the already reserved 1,440-question auxiliary TRAIN pool. It takes ranks
256:512 in each subject under the same fixed hash order, excluding the first
512 paired questions and all 160 calibration questions. The 5,120-call grid
uses the unchanged compact solver, seven plain paths, and three Complete paths.
At most two existing-app GPUs run; a `finally` block restores zero minimum
containers. No cloud training or new test inference is scheduled.

The prepared follow-up extracts 1,536 old-V1 observations locally, joins all
1,280 paired science training questions, and compares expanded forests against
the retained forest on the exact same original 768 held questions. Only a
lower joint loss and lower mean marginal Brier permits development evaluation.
Every old and new question remains in its whole-question fold; all new512
held predictions are reported separately. No gain is claimed before completion.
All 540 current regression tests pass in 34.27 seconds.

`science_proposal_headroom.py` separately tests the finalV3 option scorer on the
same fixed64 original science training questions used for recent action pilots.
All622 options were scored locally before grading. The final model proposed17/64
correct options, including four beyond the declared cached controls, below the
registered six-question expansion gate. These controls include plain, Complete,
strong-root, ordinary late-review, strong-late, and blind-first paths; some older
pilot variants are not in this union, and adding them could only reduce novelty.
No feedback cloud calls are promoted. A proposal is not a compact-solver output
or controller result. Its mean paid observer compute was404.59 units.

## Earlier continuation records

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
initial A development margin is 3.91 percentage points, below the ten-point goal.
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

The mixed strong-review bank is complete. On its 128 new training questions,
ordinary Repair twice obtains 72 correct answers, one strong guided repair
89, and a second repair retaining that review 93. Of the 39 errors after the
first guided repair, 34 also have an incorrect reviewer answer. These are
training action diagnostics, not fitted-controller or fresh-test results.

`stronger_base_hybrid.py` lets C inherit the strongest completed A forecaster
and its selected posterior blend while preserving the fitted mentor dynamics
and truth-preserving transfer. Separate observers incur separate charges;
identical assets can share predictions. The development driver compares C
against the strongest ordinary or matched-review B on all 256 questions.

`prospective_freeze.py`, `prospective_audit.py`, `prospective_collect.py`, and
`prospective_grade.py` implement the prospective confirmation pipeline. The
freeze requires all development gaps, C's compute advantage, and completed
matched-review B; it rejects choosing a weaker completed A to ease C's target.
All 2,048 development decisions must then pass sequential replay and cost
audits before test collection. All 8,192 test decisions must complete with
blank references before grading can open the reserved answers. The report
includes paired confidence intervals, recomputing the strongest baseline and
A/B within each resample, plus source and subject breakdowns. Logical reviews
and physical API phases are reported separately.

Synthetic tests cover this pipeline, but a real model archive and complete
development replay remain required before use. Some early A studies lack
historical pickle digests; current archival and replay cannot establish that
missing historical byte identity. No confirmation run has started and no new
fresh-data success has been established.

The auxiliary science observer finished its fixed two epochs. On its separate
160-question calibration set it ranks the correct option first on 27 questions
(16.875%); balanced binary cross-entropy is 0.69247. This is a weak calibration
result, not project accuracy. Its subject-routed POMDP nevertheless reaches
144/256 (56.25%) on development, at 856.54 compute units. The two-observer fusion
does not improve this result and retains it. The gap above the best fixed
baseline is 4.69 percentage points, still below target.

`nonlinear_repairability.py` fits four small gradient-boosted state forecasters
from the same training observations, retaining the eight-state POMDP dynamics
and actions. Nested question-grouped temperature calibration supplies its
emission forecasts. A fixed development preview also reaches 144/256, but at
892.93 compute units; the final comparison waits for the remaining registered
A references. The preview is not a fresh-test result or final selection.

`audit_a_refit.py` reconstructs selected A forecasters, transitions, emissions
and costs from training-only paired responses and saved neural observations.
The continuous, original repairability, factorized, science and retained-fusion
artifacts have passed exact numerical reconstruction. This binds their current
inputs and artifacts; it neither reconstructs neural training nor supplies
missing historical pickle identities. The scheduled audit uses no neural calls
or development/test question records.

The 64-question science critical-review pilot reached 34 correct final answers,
versus 32 for retained second repair and 31 for the first guided answer. It adds
only two correct cases to the union of all previously available paths, while
substantially increasing compute. Its fitted 16-state extension is retained as
an experiment; development collection has not been launched for it.

`format_fixed_controls.py` checks a potential comparison confound: many science
continuations hit their output cap before a usable final answer, whereas guided
repairs ask for the answer first. The new training-only control changes exactly
that formatting sentence for all four continuation branches on 128 prespecified
questions. It preserves the compact solver, original roots, seeds, temperature
and 256-token caps, uses no reviewer, and grades only after all 512 new responses
are complete. It does not overwrite the original baseline banks.

The option-proposal pilot completed with 7/64 correct answers, versus 13/64
for plain Repair once, at higher compute. It is not promoted. A separate
32-question auxiliary training diagnostic also shows weak option learning
(5/32 top-ranked answers correct), motivating a grouped training objective
before spending further compute on that feedback channel.

`train_science_listwise.py` tests a question-grouped objective: cross-entropy
over every official option plus balanced binary correctness loss. A fixed
four-question training sanity check improved top-ranked correctness from one
to four questions in 32 updates; this establishes learnability on those four
examples, not generalization. A separately registered continuation uses the
same 1,440 auxiliary training questions for three epochs, keeps only the final
checkpoint, and calibrates it after archive reload. The previous weights and
their development result remain available for comparison.

That continuation was stopped at 244/2,160 updates because concurrent local
training caused severe memory pressure and swapping. Its partial artifacts are
preserved, but it produced no deployable checkpoint or development result. A
serialized restart with the same objective and schedule is queued. New
`science_training_checkpoint.py` and `train_science_listwise_resumable.py`
seal model, optimizer, scheduler, random-number state and batch position every
100 completed updates. Interrupted training resumes the same fixed schedule;
intermediate checkpoints are not candidates for model selection. The queue
waits for the main observer, its A evaluation, and sufficient free memory.
The main streaming observer completed all 32 training blocks and is archiving
weights and regenerating excluded-question predictions.

`fit_new_source_mentor.py` adds a source-matched hybrid fit using all 512 new
training questions and exactly the 128 new strong-review pairs. It uniformly
excludes the 64 old pairs, reuses 4,224 verified neural observations, and makes
no new model calls. Both original-emission and question-excluded-emission
variants are archived separately. Fitting is complete; development comparison
and prospective provenance checks are still required before selection.

`audit_train_source_alignment.py` found matching source references for all
512 training questions, matching official option text/order for all 256 science
questions, and matching prompt hashes for all 3,584 plain solver nodes. It found
no context-window pressure under the returned token counts. A separate manual
review identified two training questions that require absent diagrams; these
are recorded without changing the frozen data. The audit checks source
alignment and parsing, not independent scientific verification of every label.

`prospective_parallel.py` provides a separately registered four-question runner
for future confirmation, retaining all eight policies' sequential decisions
within each question and the two-GPU cap. Durable physical-response journals,
singleflight requests, serialized observer access, and complete-grid checks
support resumable collection. It requires a full development replay through
that runner before any test collection. Nine synthetic tests and an independent
review passed; no actual prospective run has started.

The formatting control is complete on all 128 training questions. Asking for
the answer first reduced Repair once from 70 to 51 correct, Repair twice from
72 to 57, Retry once from 53 to 30, and Retry twice from 60 to 22. The reductions
were concentrated in math. This formatting change is not selected, and the
original baselines remain intact.

The independent second-review pilot finished at 34/64 science answers correct,
versus 32/64 for retained second repair, at about 90% more compute. It contributes
three correct answers beyond the union of all earlier plain, retained-review,
and critical-review paths. That union reaches 44/64, but it uses hindsight and
does not measure attainable controller accuracy. This expensive action is not
promoted to a development controller on its average gain alone.

A separate training-only diagnostic predicts paid-review correctness using
32 fixed metadata features. All review variants from a question remain in the
same outer fold; inner question-grouped folds calibrate probabilities without
access to outer held labels. On the 64 new science root reviews, logistic
metadata features improve Brier score from 0.2903 for confidence/agreement
features to 0.2363, and AUROC from 0.5132 to 0.6745. This motivates a new paid
observation model for C; it is not a controller or fresh-test accuracy result.

The metadata-review POMDP variants are now fitted for both the mixed and
new-source cohorts. They rebuild review likelihoods and review-conditioned
transitions, preserve child observations and terminal dynamics, and replace
the review encoder with a logistic metadata observer. All teacher calls and
child encoder passes remain charged. Their explicit runtime must be used;
the original semantic-review dispatch cannot evaluate these models correctly.
Development evaluation is pending.

Sparse word features improved question-excluded training state forecasts but
did not improve the resulting POMDP on development: 142/256 correct at 926.40
compute units. The retained A stays at 144/256 and 856.54. This experiment used
all 256 development questions and cached neural observations, with no new
neural or cloud calls. No further word-feature parameter search is selected.

The existing development panel has a perfect-hindsight ceiling of 165/256
(64.45%) over its seven plain paths: 123 math, 21 chemistry and 21 physics
questions have any correct path. A needs at least 158/256 to exceed the best
fixed baseline by ten points. This ceiling is an availability diagnostic,
not an attainable policy score or an independent test result.

All 512 ordinary training grids and 256 ordinary development grids are now
complete. Ordinary B reaches 165/256 (64.45%), a 12.89-point margin over Repair
twice, at 2,143.33 compute units. The original hybrid search reaches 173/256
(67.58%) at 3,375.63 units; under B's compute ceiling, its selected C reaches
162/256 (63.28%) at 2,058.77 units. The cross-fitted review variant also reaches
162/256 below that ceiling, at 2,064.10 units. Neither meets C's accuracy target.
The stronger matched-review B and additional hybrid comparisons remain active;
these are intermediate development selections, not fresh-test results.

`development_metadata_review.py` evaluates the explicit metadata-review POMDP
on the entire development panel, after importing complete compatible response
and neural caches. It seals all physical responses, errors, cache inputs,
candidate results and selection together. Both mixed-source and new-source
cohorts are queued, with the same strongest independently selected B reference.

`science_mentor_expansion.py` registers strong-review pairs for all 192 remaining
new training science questions, preserving the full set regardless of solver
success. It stages collection after the current B and metadata-C comparisons.
`finish_once_pilot.py` separately tests continuing the paid initial reasoning
under the original 256-token compact-solver cap, on all 128 registered new
mentor-training questions. This is a new repair-prompt study, with all five
original controls retained; it is not an unchanged-gate improvement. It can
advance only with at least seven net correct answers over the stronger
one-step control, no greater average compute than Repair once, and four newly
correct cases beyond the seven-path hindsight union. Neither staged collection
uses reserved test questions or authorizes a fresh-test success claim.

The matched strong-review B comparison is complete and improves B to 177/256
(69.14%), at 3,772.86 active-parameter compute units. The source-matched C
reaches 175/256 (68.36%); its cross-fitted version uses 3,241.68 units, saving
14.08% against that B. All 36 matched B rules were replayed independently and
the stronger B was retained. The two metadata-review C cohorts obtain 171 and
174 correct answers, so neither displaces the 175-answer C. A still needs at
least 158 correct answers and C now needs at least 203; both accuracy targets
remain unmet on development. The reserved fresh panel remains unopened.

`second_review_value_diagnostic.py` checks whether paid-prefix metadata can
selectively justify an independent second review. Its fixed three-fold joint
correctness and cost forecasters use all 64 earlier science training questions,
without held-question labels, held future costs or unpaid review text in the
decision. All five prespecified penalties produce 29–31 correct answers,
versus 32 for retained repair. This training-only diagnostic does not justify
promoting the expensive second-review action or searching more thresholds.

The Finish pilot completed all 128 physical calls and failed all promotion
criteria: 59 correct versus 70 for Repair once, 2.03% greater compute, and zero
newly correct questions beyond the original seven-path union. The three
validity heuristics also remain reported; no two-step expansion is selected.

The additional metadata comparison inherits the strongest completed A across seven
registered families, retaining a completed science-v3 result if available.
`stronger_metadata_review.py` preserves the learned mentor, transfer, and
metadata adapter while changing the unqueried A component. Its staged
development run compared all four fitted metadata parents on the same 256
questions; each distinct observer's actual token slots remained charged.
Its best result is 174/256 at 3,644.52 compute units. The corresponding
stronger-A semantic/reported-review comparison reaches 173/256 at 3,577.53.
Neither replaces the retained 175/256 C.

`prospective_metadata_freeze.py`, `prospective_metadata_parallel.py`, and
`metadata_paid_audit.py` add an explicit prospective route for metadata C.
They verify the saved training/calibration chain, preserve strongest-A/B and
accuracy/compute gates, and independently recompute metadata probabilities
from paid inputs. Metadata CPU timings are recorded separately from the
inference compute proxy. The full 2,048-decision development replay and
8,192-decision blind confirmation remain required. Independent synthetic
and saved-model provenance checks passed; no real freeze or test run occurred.

The completed V4 observer produced 30,804 question-excluded observations over
1,984 training questions. Its new A reaches 143/256 at 885.17 compute units;
the nonlinear alternative also fails to displace the existing A. A therefore
remains at 144/256 (56.25%) and 856.54 units. The neural and forecaster
cross-fitting stages are not nested; their training diagnostics do not replace
independent development or test evaluation.

`development_retained_path_audit.py` completed the 105 missing compact-solver
continuations needed to cover all ten distinct single-strong-review paths on
all 128 development science questions. They add no newly solvable science
question. The complete science hindsight union is 64/128, giving an optimistic
overall ceiling of 192/256 (75.00%) even if every math question were solved.
The C target against B's 177/256 requires at least 203/256 (79.30%). This rules
out reaching the current development target by changing only the gate on this
fixed response tree. It is an answer-availability bound, not policy accuracy
or a claim about a new action protocol or the unopened fresh test.

The already collected ordinary and late-review paths provide diversity. Their
union with the strong-review tree contains a correct answer on 204/256
development questions, including 78/128 science questions. Achieving the target
would require selecting 203 of those 204 available answers before considering
compute. A larger router alone consequently has very little error margin.

All 192 additional training science review pairs are collected. The expanded
fit uses all 512 plain training trees and 320 new-source reviewed roots
(64 math, 128 chemistry, 128 physics), with 448 question-grouped metadata
records. `extract_expanded_mentor.py` reuses 4,224 exact cached neural entries
and extracts only 960 missing entries. `fit_expanded_mentor.py` registers a
CPU-only weighted/calibrated POMDP fit. Within-subject question mass and
50/25/25 subject weighting are explicit, so this is a new fitting recipe,
not a pure more-data ablation. Its separately registered plain four-state A
evaluation reaches 139/256 and does not replace the stronger completed A.

The registered scoring pilot acquires one compact-solver continuation token
on each of three paid prefixes for all 64 fixed science training questions.
Its probabilities are new paid observations; historical full-token
probabilities were not retained. Original answers remain unchanged in the
primary diagnostic, and missing top-20 option probabilities are censored,
not assigned zero. A separately preregistered secondary analysis checks
hypothetical one-token completion availability after all 192 calls are sealed.
It cannot report a controller improvement: promoting that completion would
require a new answer-changing action, matched controls, and a new POMDP.
Neither analysis uses development labels or the reserved fresh test.

The scoring pilot is complete and fails its primary observation screen:
same-question held-fold joint log loss increases from 1.6642 to 1.7165.
Current-answer and repair Brier scores also worsen. The score-observation
variant is not expanded. The separate completion analysis finds six newly
solvable science training questions: the seven-path union increases from 29/64
to 35/64. Repair followed by hypothetical completion gets 23/64 correct,
versus 17/64 before completion; retry followed by completion gets 17/64,
versus 5/64. These consume two continuation calls and average 2,391.44 and
2,156.86 compute units, respectively. They are training action measurements,
not A/C accuracy or fresh validation.

`science_complete_actions.py` defines a separate, science-only action study
over all 256 training science questions and all three eligible prefixes.
It reuses all 192 identical pilot requests and acquires 576 remaining requests.
Only an exact offered option letter is submittable. Complete costs one call
and terminates; it cannot be followed by another action. The five original
baselines and six additional fixed/validity controls remain reported. The
fixed training feasibility screen requires at least 16 newly solvable questions
beyond the original seven paths before fitting the new controller.

`completion_pomdp.py` adds a genuine 16-state POMDP over correctness now,
after repair, after retry, and after Complete. Its finite-observation Bellman
backup preserves the two-continuation budget, retains fixability observations
for invalid answers, and counts the cost of all paid observations. It does not
use the failed scoring-probability features. Math retains its existing A;
the science model uses only paired training labels and original paid features.
The new fitting driver binds the complete physical-response ledger and the
original observer provenance. Its separate development evaluation is reported
below.

The full training action grid is now sealed and passes that screen: the
science union increases from 96/256 to 118/256, with 22 newly solvable cases
(10 chemistry, 12 physics). Six came from the original pilot and 16 from the
remaining 192 questions. A fixed Retry-then-Complete-if-invalid control obtains
62/256 correct versus 51/256 for the strongest original science baseline,
Repair twice, with 23.27% lower compute (1,868.50 versus 2,435.18 units).
This simple control is retained explicitly; its gain cannot be attributed to
learned planning. The single prespecified 16-state POMDP has been fitted with
no new neural calls. Its separate development study freezes seven penalties,
collects all 384 science completions before scoring, and reports both science
controls and full-256 comparisons sharing exactly the same retained math A.
The completed development selection obtains 146/256 (57.03%) at 1,084.64
compute units, versus the earlier A's 144/256. However, the matched
Repair-then-Complete-if-invalid control obtains 150/256 (58.59%) at 1,232.71,
using exactly the same retained math A in both full-panel comparisons. The
learned planner therefore does not outperform this new simple control on
accuracy. Its margin over the original best baseline is 5.47 points, below
the required ten. All six matched controls and their science-only panels
remain reported. None of these are fresh-test results.

The standard second repair with mentor feedback removed also failed its
64-question science training screen: 18 correct versus 32 for retained
repair, with zero paired wins and 14 losses. It added no newly solvable
question and cost 0.88% more. This repair path is not expanded.

The grouped science observer's resumable run paused safely at step 20 because
unused MPS allocator cache exhausted the memory guard. A separately registered
operational wrapper released unused cache before existing telemetry checks,
preserving optimizer state, RNG, batches, and the training recipe. The bounded
step-20-to-100 resume passed; only the final scheduled step 2,160 may become a
deployment checkpoint. No intermediate recovery checkpoint selects a model.
The subsequent full resume releases unused cache every five completed
updates. Its first 50 updates maintained 54–63% free system memory; the
fixed final checkpoint and auxiliary calibration remain pending.

The expanded 320-root hybrid fit finishes at 177/256 (69.14%), matching B's
accuracy, with 3,747.93 compute units under its collected receipts. It issued
50 compact retry/reuse requests that duplicated the prior availability audit:
36 text/token receipts differ, but all parsed answers agree. Because the
apparent compute saving against B is only 0.66%, all 156 candidate policies
were replayed using the earlier canonical receipts. Every decision, answer,
and encoder charge remained identical. The selected policy was unchanged,
and its canonical compute is 3,747.96, still only 0.66% cheaper than B.
The original run remains preserved; the 50 duplicate acquisition calls
(33,839 input and 3,147 output tokens) are disclosed separately. A pure
four-state POMDP fitted from the expanded data reaches only 139/256, so it
does not replace A. A's and C's required accuracy gaps remain unmet, and the fresh
1,024-question set is still untouched.

The separately registered Complete-enabled B comparison evaluates 24 fixed
LLM-only science rules and retains both earlier B families. It reproduces the
strong original B's 177/256 and compute exactly. The best new rule gets 175/256
at 3,586.37 units, so it does not replace B or lower C's required threshold.

The full development availability audit finds correct science answers on
42/128 questions in the original tree, 55/128 after adding Complete, 64/128
in the strong-review tree, and 70/128 in their combined tree. Even perfect math
would yield only 198/256, below C's required 203. A new transfer fit on that
unchanged tree is held; a training-only pilot will test strong review after a
paid first repair or retry instead of at the root. This is a new review-timing
protocol and requires matched controls before a controller claim.

The fixed training-only V4 observation diagnostic also fails its screen:
joint log loss increases from 1.391802 to 1.399549, and mean four-marginal Brier
increases from 0.140674 to 0.142630. It uses all 256 training science questions
and 768 decision nodes, with no new inference or development/test reads.
The neural and forecaster folds are not nested, so these remain diagnostic
training results. No V4 promotion is made. A separate richer-observation study
is being prepared to expose the old frozen observer's 256 hidden features
from its existing forward pass.

The grouped science observer completed all 2,160 scheduled updates. Its
auxiliary calibration ranking improved to 47/160 (29.38%), but its separate
development POMDP reaches only 145/256 (56.64%). The existing 146/256 A remains
the accuracy leader. The fixed half-posterior blend also finishes at 146/256,
with slightly higher compute (1,088.08); it does not replace the finite filter.

The strong late-review pilot completes all 384 physical calls and fails its
headroom screen. Review after repair obtains 28/64, review after retry 29/64,
and retained root review 32/64. The new paths add three questions beyond the
registered plain/Complete/root-review union, below the required six; they add
only two beyond the broader union including ordinary late reviews. No expansion
of this protocol is selected.

All 768 richer-observation training rows pass the fixed numerical parity
check. With the original C=0.1 regularization, adding 256 features worsens
held-question prediction loss. A separate nested training study gives both
feature families the same four regularization choices, with an optional
fold-fitted PCA32 for the larger family. It selects C=0.001 without PCA for
the larger family in all three outer folds. Joint loss improves from 1.178638
for the tuned control to 1.141229, and mean marginal Brier improves from
0.127284 to 0.123248. These are nested training diagnostics, not controller
accuracy or fresh validation; the earlier failed fixed-C result is preserved.
A deployable POMDP and separate development comparison remain required.

A new 64-question pilot is registered to generate independent reviewer
reasoning before exposing the original compact solution. It always pays for
two teacher phases and two compact continuations, for 256 physical calls in
total. It must add at least six questions beyond the entire known 48/64 union
and cost at most 1.25 times retained root review. The completed pilot fails
the novelty criterion. Its first guided repair gets 23/64 correct and its
second gets 22/64, below the original retained-review control's 32/64. The
second trajectory costs 5,305.26 units, a 2.54% saving that does not offset the
accuracy loss. This protocol is not promoted.

The separately registered 512-question auxiliary-TRAIN paired extension is
now collecting all seven original and three Complete paths, with a maximum
of 5,120 compact calls, four request workers, and at most two cloud GPUs.
Its 256 chemistry and 256 physics questions come from the sealed auxiliary
training pool; all 160 calibration questions are excluded. This will expand
the paired science training set from 256 to 768 questions. No fit or accuracy
claim is available from the additional data yet.

The latest full regression passes 481 tests. No fresh test inference or grading
has occurred. A capacity audit used reserved IDs and hashed public-source
stems solely to exclude overlapping training questions; it did not decode
test references or use test outcomes.
