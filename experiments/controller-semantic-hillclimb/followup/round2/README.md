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
| Best A: math-specialized POMDP with clustered science observations and terminal Complete | 153 | 59.77% | 1,087.27 |
| Best B: strong retained-review LLM | 177 | 69.14% | 3,772.86 |
| Best C: expanded-data POMDP + LLM | 177 | 69.14% | 3,747.96 |

The proxy counts all paid solver, reviewer, and observer tokens weighted by
active parameters, normalized to 3.09 billion parameters. Both physical phases
of a bounded review are billed. This is not measured FLOPs or wall-clock time;
the full-35B reviewer sensitivity is also retained in the receipts.

A is 8.20 percentage points above the strongest original baseline, B is 17.58
points above it, and C is 0.00 points above the stronger of A/B. A needs at least
158 correct answers and C needs at least 203 on this panel. C saves only 0.66%
compute relative to B. The requested accuracy gaps are not met.

Complete is a separate answer-changing action with matched controls. The best
such simple control, Repair then Complete if invalid, obtains 151/256 (58.98%)
at 1,271.37 units, using exactly the same retained math A as the new POMDP.
Best A gets two more questions correct and uses 14.48% less compute. Cheaper
previous A policies remain at 152/256 and 1,048.61 units, and 150/256 and 986.26 units. The original five baselines
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

A separate context-dependent LLM prior also fails its nested TRAIN screen.
Both raw and dense variants worsen mentor correctness Brier and joint-state
loss against a transfer model that also excludes each held question. They
preserve A's current-correctness belief exactly and require no extra neural
pass, but neither is promoted.

A fixed clustered-observation study now preserves the same 30 POMDP symbols:
24 valid and six invalid. Clusters use only square-root joint forecasts from
excluded-question TRAIN children. The existing forest, transitions, costs,
states, and exact finite Bellman backup stay unchanged; only the observation
partition and its fitted emission table change. All sixteen development
candidates complete. The selected dense model gains one answer to 151/256
(58.98%) at 1,074.87 compute, versus retained 150/256 at 986.26. This is a
0.39-point gain with 8.98% more compute than the cheaper prior A. All controls
and original math decisions/costs reproduce exactly; no new inference was needed.
A still needs seven more correct answers on this development panel to meet
the requested margin. The full regression passes 545 tests.

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


## Continuous belief and split-optimized forest follow-up

The separately registered posterior-blend comparison uses the same clustered
TRAIN observation model and exact finite-observation future-value backup. At
an observed child, it mixes the finite-filter belief with the full sixteen-state
forecast of the paid root/child history. This is an approximate belief estimator,
not an exact continuous-emission Bayesian filter. The zero-weight control
reproduces every previous development decision and charge exactly. All math
outputs and all original/matched controls also reproduce exactly.

Of 32 registered development candidates, the dense model with posterior weight
0.5 and science penalty 0.02 reaches 152/256 at 1,048.61 compute units. It improves
the previous 151-answer A while using 2.44% less compute. The cheaper 150-answer
A remains retained. No new neural or cloud inference was needed. This reused
panel remains development data; A still needs six additional correct answers
to pass its required gap here, and fresh validation has not started.

A separate split-optimized RandomForest comparison fails its TRAIN screen.
Nested joint log loss worsens from 1.14784 to 1.15678, although mean-four marginal
Brier improves slightly from 0.129221 to 0.128970. Its registered requirement
was improvement in both, so no development promotion follows. The complete
negative result and whole-question fold receipts are retained.

The matching continuous-A hybrid comparison is also complete. It retains the
same sixteen-to-eight state transfer, LLM reviewer, reviewed-mode dynamics,
math decisions, and all charged observation paths. Its 26 development candidates
peak at 176/256, so the previous 177/256 C remains selected. The improved A does
not establish an improved C. All 551 experiment regression tests pass.


## Registered next observations and grounded-review pilot

A smooth RBF-feature categorical forecaster also fails the TRAIN screen:
nested joint log loss is 1.16909 versus 1.14784 for the retained forest, and
mean-four marginal Brier is 0.131836 versus 0.129221. No development comparison
or policy promotion follows.

The new full-representation probe captures the frozen encoder's normalized
1,024-dimensional text representation and existing 256-dimensional task
projection in one forward. It extracts all 768 original science TRAIN prefixes
locally on two CPU threads, with the original logits checked against the
archived cache and every attempted chunk recorded durably. A separate nested
three-C comparison will test compressed386 versus full1410 features. This is
an observation experiment, not a new solver, larger runtime encoder, or claimed
controller gain. Development/test inference is not part of this extraction.

The registered grounded-review pilot adds question-only excerpts from the
unchanged textbook index to the same 1,024-token reviewer. All 64 fixed science
TRAIN questions have retrieved context. It preserves the 512-token answer cap,
the compact solver and its two 128-token guided continuations, all old controls,
and every added teacher input token in the cost calculation. Its maximum is
256 physical calls on two existing-app GPUs. Durable request intents block
unresolved automatic retries, and a finally block restores zero minimum
containers. Acquisition is blocked until the second512 paired batch completes.
The fixed screen requires six newly solvable questions beyond all registered
prior paths and mean full-trajectory compute at most 1.25 times the retained
strong-review control. Matched B and separate learned/fresh validation would
still be required after a successful pilot. No cloud training is introduced.

All 559 experiment tests pass; the actual grounded request journal also passes
its complete-cohort replay and unresolved-request tests.

The full1024 extraction and nested TRAIN diagnostic are complete. All 15,360
original logits pass the fixed parity bound; maximum absolute difference is
0.0000372. Full features worsen joint log loss from 1.14123 to 1.14924 and
mean-four marginal Brier from 0.123248 to 0.125724 on the same original256
held-question training comparison. The screen fails, so no auxiliary expansion,
development evaluation, or online full-representation adapter is promoted.

## Math-specific forecaster

A nested three-outer/three-inner whole-question TRAIN screen compares the
pooled forecaster with math-only logistic/forest alternatives. The selected
math forecaster lowers joint loss from 1.00324 to 0.92019 and average three-bit
Brier from 0.10868 to 0.10413; current-answer Brier alone worsens slightly.
The resulting POMDP preserves prior, transition and cost arrays, refits the
matched emission model, and keeps the strongest science policy unchanged.
Twenty-one registered development settings select 153/256 overall, including
118/128 math. This is one more correct answer than A152 at 3.69% more compute.
All five original baselines reproduce exactly; matched Complete controls use
the identical new math policy. This remains development selection, not a fresh
result, and does not meet the requested accuracy gaps.

Before the second512 labels are joined or the1280 fit is registered, the next
comparison has been extended to retain this selected math policy. Passing
TRAIN forecasters will each retain the original observation partition and a
24-valid/6-invalid clustered partition fitted from nested TRAIN predictions.
Development replay will compare posterior weights0 and0.5, seven global
science penalties and one subject combination per configuration (at most64
candidates). This compares additional data together with the already tested
observation recipes; gains cannot be attributed to data alone. Original and
stronger retained A decisions/costs must reproduce exactly, and the six matched
controls use the identical retained math policy. No new feature encoder or
cloud inference is introduced by this comparison.

## Separate expanded retrieval corpus

`advanced_textbook_corpus.py` prepares a separate, question-independent corpus
from all lecture PDF links in four fixed MIT OCW courses:
[Quantum Physics II](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/pages/lecture-notes/),
[Statistical Mechanics I](https://ocw.mit.edu/courses/8-333-statistical-mechanics-i-statistical-mechanics-of-particles-fall-2013/pages/lecture-notes/),
[Thermodynamics and Kinetics](https://ocw.mit.edu/courses/5-60-thermodynamics-kinetics-spring-2008/pages/lecture-notes/), and
[Physical Chemistry](https://ocw.mit.edu/courses/5-61-physical-chemistry-fall-2017/pages/lecture-notes/).
The build adds2,751 passages from112 PDFs to a copy of the original corpus.
It extracts283,601 words and includes1,025 of1,146 pages under the registered
minimum-text rule. PDF reading order and equations may be degraded; no OCR or
question-level retrieval evaluation has been performed. Raw files, text,
source hashes, author attribution, and CC BY-NC-SA4.0 notices remain in the
private archive. No benchmark records or model calls were used to build it.
The existing grounded-review pilot keeps its original corpus unchanged. This
new corpus has not been promoted into any controller or registered new pilot.

## Second paired block and grounded runner

The second512-question paired block is complete:5,120 logical responses,
5,120 physical calls/attempts, zero transport errors,2,391,479 input tokens and
703,412 output tokens. These acquisition costs are separate from inference
compute. The collector restored min0/max2 capacity. Registered local feature
extraction is in progress before the expanded1280 fit.

The first grounded runner failed before any inference because its prerequisite
check treated structured `input_evidence` metadata as a flat file manifest.
`grounded_review_runner_v2.py` verifies `input_evidence.files` instead, retaining
every completion/capacity check and exactly the original prompts, cohort,
budgets and scoring. The original source and empty registration remain intact.
The new registration binds the adapter, original sources, and the failed
predecessor registration; it refuses a predecessor with any inference artifacts.
This corrected pilot is collecting on at most two existing-app GPUs while
feature extraction runs locally. No pilot outcome or fresh result is claimed.

## Completed expanded-data and retrieval comparisons

The1280-question likelihood screen passed only raw130: common768 held-question
joint loss improved1.14784→1.12983 and mean-four Brier0.12922→0.12899 (the latter
interval includes zero). Dense386 failed the Brier criterion. All32 registered
raw130 development candidates completed; their best was147/256, so the stronger
A153 is retained. The original and retained parent decisions, five baselines,
and matched controls reproduced exactly.

The original-corpus grounded pilot scored31/64 at5771.52 compute units versus
the strong retained control's32/64 at5443.32. It added zero solves beyond the
50-question union of all prior paths. The separate expanded-corpus pilot scored
32/64 at5789.64 and also added zero novel solves. Both failed the registered
six-new-solve screen. `advanced_grounded_runner.py` reused 180 identical physical
responses covering 45 unchanged trajectories; only 76 calls were new, while all
256 recorded responses are still charged for inference. Both
collectors restored min0/max2 capacity. Neither experiment changes selected C.

`policy_crossfit_diagnostic.py` separately tests complete POMDP decisions with
outer question folds excluded from every forecaster, transition, emission and
cost fit. Inner question folds generate emission forecasts. On the identical
768 held TRAIN questions, the fixed old768 dense recipe gets193 correct at
1586.26 compute; expanded1280 raw gets194 at1234.61, and expanded1280 dense gets
196 at1260.43. Both save compute but miss the prespecified two-percentage-point
accuracy screen. This is exploratory TRAIN evidence, not fresh confirmation or
a replacement for the failed likelihood screen.

`mlp_completion_forecast.py` and `mlp_policy_crossfit.py` next test a small64-unit
regularized CPU head on the same cached observations and whole-policy folds.
The434M encoder is frozen and never called; no cloud training or new solver
inference occurs. Head latency is explicitly outside the token-based compute
proxy. The completed raw head gets198/768 at1886.19 compute and the dense head
gets195/768 at1797.92, versus193/768 at1586.26 for the retained forest control.
Both fail the registered accuracy-and-compute screen; neither is promoted.

## Local joint-outcome observer adaptation

`joint_outcome_observer.py` and `train_joint_outcome_observer.py` register one
fixed local MPS recipe on the1280 science TRAIN questions. The last four layers
of the existing434M discriminative observer and a new16-state head learn the
joint correctness of the current answer, Repair, Retry and Complete. The
observer never generates answers; the compact solver and response banks stay
fixed. Three whole-question outer folds each train for exactly two epochs.
The initializer excludes all1280 questions, and each held prediction reloads
the saved, half-rounded excluded-fold checkpoint. A direct joint loss and
marginal correctness loss replace the small frozen-feature classifiers.

This first stage screens only conditional last-step POMDP decisions on TRAIN,
requiring at least two percentage points higher accuracy and at most1.10 times
the excluded-question forest control's compute. It is not a complete controller
comparison. A passing screen would still require full fitting, full-policy
development evaluation and fresh validation. Training is local; there are no
cloud calls, development/test reads or full deployable model in this stage.
The registered training has completed; its negative result is recorded below. All576 tests passed,
including invalid-state support, checkpoint round-trip, excluded-question
cost fitting and whole-question fold checks. The selected A/B/C stay unchanged.

`joint_observer_pomdp.py` prepares paid-prefix inference and exact observer
charges for a passing model. `fit_joint_observer_pomdp.py` refuses a failed or
incomplete screen and would fit the same two-epoch recipe on all1280 TRAIN
questions. Both original and clustered finite observation kernels retain the
existing two-step Bellman backup. These preparation modules have not fitted or
promoted a full model. Tests check that references and unpaid future states are
rejected, changed paid roots invalidate the cache, and charging reuses the exact
paid observations without adding or omitting inference calls.

## Existing-review routing diagnostic

`review_portfolio_diagnostic.py` tests whether the paid root alone can choose
among existing review timings, budgets and retrieval protocols. It keeps all64
registered science TRAIN questions and all fixed controls. Four fixed recipes
(raw130/dense386 with regularized logistic/forest heads) predict outcomes and
costs with whole questions excluded. No continuation or review text is visible
before choosing its protocol. Full chosen-trajectory compute plus one root
observer is charged; the small CPU head latency remains separate.

All four fail. The best gets34/64 at6953.34 compute, below the strongest fixed
protocol's36/64 and above the retained rootstrong control's5443.32 compute.
The other recipes get31,28 and30 correct. This is a negative routing feasibility
result on TRAIN, not a new POMDP or fresh controller result. It causes no new
cloud calls, no new encoder calls and no change to selected C.

## Bounded arithmetic review pilot

The separately registered `arithmetic_review_runner.py` pilot adds an explicit
numerical tool to the frozen reviewer. Its1024-token reasoning phase may request
up to8 bounded scalar expressions; the mandatory512-token decision phase sees
their values or errors. The tool interprets a whitelist of arithmetic AST nodes
and unary math functions without evaluating arbitrary code or accessing files,
network or references. This is floating-point arithmetic, not a symbolic solver.
The compact model, its128-token guided prompts, original paired seeds and
two-continuation limit stay unchanged. It receives the existing answer/check
packet from the reviewer. All teacher/tool-output input tokens and compact calls
are charged; calculator CPU time is reported separately from the token proxy.

The pilot keeps all64 previously registered TRAIN questions and allows exactly
128 physical teacher phases plus128 compact calls, on at most two existing-app
GPUs. It preserves durable attempts, errors and wrong completions, and restores
min0/max2 capacity in a finally block. It must add at least six solves beyond
the50-question union of every prior registered path and use at most1.25 times
the retained strong-review trajectory compute. Those are acquisition screens,
not the final C accuracy/compute requirements. Matched B must receive identical
calculator availability if this capability is promoted. No controller or fresh
result has been claimed. All588 tests pass, including complete256-request
synthetic replay, tool tamper detection, numerical bounds and paid-prefix checks.

`development_joint_observer.py` prepares the subsequent full256 DEV comparison
only if the joint observer passes its TRAIN screen and completes the full fit.
It verifies those prerequisites before decoding development records. The
retained A153 decisions, all five baselines and matched controls must reproduce
exactly. It would cache384 new science observations and compare32 fixed
original/clustered-kernel settings, charging only observations purchased by each
policy. The best prior A remains eligible. This adapter is prepared, not run;
all590 regression tests pass, and no fresh inference is authorized by this code.

The arithmetic pilot completed all256 physical calls and restored min0/max2
capacity. It scores23/64 after one guided repair and22/64 after two, at5513.87
compute for the latter versus5443.32 for the retained strong control. It adds
zero correct questions beyond the50-question prior union. Only three questions
requested calculator expressions (four expressions, three valid). Mean measured
calculator replay time is0.00163ms per question, outside the token proxy. The
pilot fails its novelty screen and is not promoted.

## Disjoint supervised training examples

`training_example_pool.py` builds a private1216-question demonstration pool from
the sealed1280 science TRAIN cohort, excluding every question in the64-question
pilot. Each example contains its question stem and the correct offered answer
content from TRAIN labels, without a generated rationale. TFIDF retrieval uses
only the current question stem and observed subject, excludes equal stems and
fivegram Jaccard similarity at least0.3, and supplies one example with at most
120 stem words and60 answer words. No current/pilot reference can be retrieved.

The separately registered `example_review_runner.py` pilot explicitly supplies
these other TRAIN reference contents as in-context supervision. It does not
claim reference-free textbook retrieval. The original1024/512 bounded-review
continuation, conditional second teacher phase, compact prompts/seeds and
two-continuation limit remain intact. All evidence tokens are charged; retrieval
CPU timing is reported separately. Matched B receives the same examples if
promoted. The entire64-question cohort and known50-question union remain the
screen. Collection completed on at most two existing GPUs, with no cloud
training. All595 tests passed, including cohort exclusion, public-only retrieval,
reference isolation, full physical replay and scoped protocol restoration.

The local joint observer completed all three excluded-question folds and failed
its conditional decision screen:657/2560 correct (25.664%) versus661/2560
(25.820%) for the retained forest control, at2086.43 versus2006.28 compute
(3.995% higher). Joint log loss worsens1.12957→1.15234 and mean-four Brier
0.13144→0.13505. The paired-question accuracy-difference interval is
[-0.391,+0.039] percentage points. These are conditional TRAIN decisions, not
full-controller or fresh results. All checkpoints and forecasts are preserved;
the prepared full fit and development comparison are not run after this failure.

The disjoint-example pilot completed all256 physical calls with zero transport
errors and restored min0/max2 cloud capacity. Its first guided answer gets29/64
and its retained second repair28/64 at5715.69 compute, versus5443.32 for the
retained strong control. It adds zero correct questions beyond the50-question
prior union, failing the registered novelty screen. Mean retrieval CPU time is
2.678ms per question, reported separately. No controller is promoted.

## Sparse lexical outcome diagnostic

`lexical_outcome_diagnostic.py` tests paid question/root/current wording plus
the existing130 numeric features. All1280 science TRAIN questions use three
whole-question outer folds and three inner folds for vocabulary, scaling and
regularization selection. No additional neural or cloud calls occur.

The conditional last-step screen matches the control at661/2560 correct but
uses2114.80 compute versus2006.28 (5.409% more). Its paired-question accuracy
difference interval is[-0.273,+0.273] percentage points. It fails the required
two-point gain and no-higher-compute screen; no full fit or DEV comparison is
run. All597 regression tests pass. Selected A/B/C and fresh validation remain
unchanged.

## Frozen reward observation screen

`reward_outcome_diagnostic.py` registers a local-only inference screen using
[Skywork-Reward-V2-Qwen3-1.7B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-1.7B),
pinned to `e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc` (Apache-2.0). This
discriminative observer scores the paid current response to the public question;
it generates no answers. Its1,720,577,024 parameters replace the old neural
observer, with all current input tokens charged. The110 visible numeric features
remain, while all20 old neural outputs are excluded. The compact solver is
unchanged. No cloud calls or cloud training occur.

All1280 science TRAIN questions and3840 root/repair/retry observations are
registered before scoring. Three whole-question outer folds select a score-only
forest or a32-component representation with regularized logistic prediction,
using three inner folds. Scaling, PCA, priors and classifiers exclude held
questions. A conditional last-decision screen requires at least two points
more accuracy and at most1.10 times the retained forest control's compute.
This is a feasibility screen, not a full-controller or fresh comparison.

The official model card reports26million preference pairs; exact overlap with
this project's source questions has not been independently established. No
pretraining-disjointness claim is made. Inputs above1536 tokens preserve their
first768 and last768 tokens, with every truncation reported. Only paid question
and current response enter this scorer, without references or future branches.
The experiment has completed; its negative result is reported below. A/B/C remain unchanged.

`reward_math_diagnostic.py` separately registers the same observer for all256
math TRAIN questions (768 paid prefixes). It compares against the retained
nested math-specialized forecasts with the same two-point/1.10-cost conditional
screen. Eight current/repair/retry states are preserved; Complete is not added
to math. Original training caches lack token counts, so their exact verified
inputs are re-tokenized locally to bill actual lengths, without neural calls.
The current scorer replaces the old neural features and is fully charged.
No development/test questions enter either observer screen.

The math reward screen completed:447/512 correct conditional decisions versus
446/512 for the retained math forecaster (0.195 percentage points higher), at
914.83 versus752.72 compute (21.54% more). Its paired-question accuracy
difference interval is[-1.172,+1.563] points. All768 observations used127,773
input tokens without truncation. The registered two-point/1.10-cost screen
fails, so no full math fit or development comparison follows. All603 regression
tests pass. This does not change the selected controllers.

## Frozen reward review-critic screen

`reward_review_diagnostic.py` separately tests the same frozen scorer on all256
already-paid science TRAIN reviews. It receives the original question and the
complete paid review text; references and future solver responses are excluded.
Three outer whole-question folds contain independent three-fold selection of a
score-only or32-component critic. The comparator is the strongest Brier result
among all three prior excluded-question critics (old, option, combined), not
only the weakest control. The screen requires Brier improvement of at least
0.005 and strictly higher AUC.

This trial has completed with no new cloud calls; its negative result is reported below. It
creates no new correct-answer paths. Its1.72B scorer cost is additive to all
paid LLM phases and other solver observations; a successful prediction screen
would still require full POMDP integration and development accuracy/compute
validation against the strongest B. No controller or fresh result is claimed.

Both remaining reward-model screens failed. The science outcome trial gets
657/2560 conditional decisions correct versus661/2560, at2283.90 versus2006.28
compute (13.84% more). It scores3,840 inputs, of which34 exceed the input cap.
The review critic has Brier0.23049 and AUC0.60510, versus0.22296 and0.62589 for
the strongest prior combined critic. Its Brier difference is+0.00753, with
paired95% interval[-0.01166,+0.02541]. Mean additional scorer cost is779.34 per
paid review;58/256 review inputs exceed the cap. These critics are not promoted
and no full controller fit or DEV comparison follows. All605 tests pass.
The selected A/B/C and all five baselines remain unchanged.

## Active textbook retrieval pilot

`active_textbook_review_runner.py` tests a new review capability on the entire
same64-question TRAIN science cohort. The frozen reviewer starts its1024-token
reasoning phase with up to two concept queries, then its512-token decision phase
receives local textbook evidence. The sealed OpenStax+MIT corpus is unchanged;
no benchmark references, web searches or neural retrieval are used. Queries
are capped at200 characters; subject-filtered TFIDF returns at most three unique
85-word excerpts across both queries. Missing results and errors remain visible.

The compact128-token guided and retained repairs keep their original prompts,
weights, temperatures and paired seeds, with at most two answer-changing calls.
Every teacher phase and evidence token is charged. Retrieval CPU time is reported
separately. The matched B must receive the same active retrieval capability if
promoted. The unchanged screen requires at least six newly correct questions
beyond all prior50 available solves and at most1.25 times the retained strong
trajectory cost. This is a headroom pilot, not a controller or fresh result.

The pilot completed256 physical calls on at most two
existing-app cloud GPUs, with no cloud training. The frozen corpus cannot be
rebuilt implicitly. Tests cover bounded queries, immutable indexes, paid-only
inputs, original compact payloads, all256 physical receipts and tamper rejection.

## Persistent difficulty in A's hidden state

`difficulty_completion_pomdp.py` adds a persistent hidden difficulty bit to
the four existing correctness bits, yielding32 states. Difficulty means fewer
than three correct endpoints among the ten paired TRAIN responses. This label
is training supervision only; inference predicts it from the same paid frozen
observer features. Transition support preserves difficulty and enforces that
next current correctness equals the selected action's outcome bit. The same
finite two-step Bellman backup, action protocol and30-symbol correctness
observation kernel remain; realized beliefs use the declared0.5 posterior blend.

`difficulty_completion_diagnostic.py` registers a full-policy comparison with
three outer question folds and three inner emission folds. All forecasts,
priors, transitions, emissions and cost estimates exclude the held question.
Both raw130 and dense386 inputs use the fixed96-tree,depth8,leaf24 recipe.
Fitting uses1280 science TRAIN questions; all methods are compared on the same
768-question common held cohort. The comparator is the strongest accuracy
result among all three sealed prior whole-policy controls. Promotion requires
at least two points more accuracy and no higher compute. The experiment is
completed on local CPU, with no new neural/cloud calls or DEV/TEST reads. All613
regression tests pass; no full model or new controller result is claimed.

The persistent-difficulty screen fails for both feature families. Raw features
obtain198/768 at1323.14 compute and dense features201/768 at1377.72, versus the
strongest previous control's196/768 at1260.43. The best gain is0.651 percentage
points with9.306% more compute; its paired accuracy-difference interval is
[-0.651,+1.953] points. Neither meets the two-point/no-higher-compute screen.
No full fit or development selection follows this negative result.

The first active-retrieval pilot fails:22/64 after one guided answer and23/64
after two, at5586.36 compute for the latter. It adds zero solves beyond the
50-question prior union and restores min0/max2 capacity.45 questions request
47 searches, but only one question receives passages. Inspection identifies
45 literal placeholder queries:21 instances of `query` and24 of `...`. No
query exceeded the character cap. This is an observed protocol failure, not
evidence that relevant active retrieval cannot help. Its entire result remains
preserved.

The separately versioned `active_textbook_review_runner_v2.py` pre-fills the
opening search tag, explicitly asks for scientific keywords, and rejects
placeholder/punctuation-only tags before they consume the two-query budget.
If no usable completed query exists, it retrieves from the public original
question stem (at most800 characters), with the fallback recorded explicitly.
The corpus, model weights,1024/512/128/128 budgets, compact prompts/seeds,
64-question cohort and six-novel-solve screen remain unchanged. This version
retains the failed first pilot in its known availability union.


The corrected active-retrieval V2 pilot also fails its acquisition screen. All64
questions finish with256 physical calls and zero transport errors; capacity
returns to min0/max2. Real queries now deliver passages to63/64 questions
(126 passages, only one question-stem fallback). The first and second compact
answers score28/64 and29/64, respectively; their union is30/64. They add zero
questions beyond the registered50-question known union. The second trajectory
costs5,933.51 units,9.01% above the original strong retained-review control.
Retrieval replay averages1.213 ms per question on the local CPU. Better evidence
delivery alone has not improved answer availability, and no controller is promoted.

`evidence_first_review_runner.py` registers a distinct timing test on the same
complete64-question TRAIN cohort: a64-token question-only concept query, local
retrieval,960-token reasoning with evidence already visible, then512-token
decision. Total reviewer output allocation remains1,536 tokens, but all three
physical prompts and outputs are charged. Both128-token compact continuations
retain their exact prompts, seeds and settings. The new durable journal permits
at most320 physical attempts, at most two existing-app GPUs, and no automatic
request retry. The registered known union includes both prior active-retrieval
pilots; promotion still requires six novel questions and a cost ratio at most1.25.
Every promoted comparison must offer the same capability to B and subsequently
validate a learned POMDP hybrid on fresh data. This registration is not a result.

`latest_results.json` records the eight-row development table and target checks
from verified sealed artifacts. It explicitly distinguishes reused development
selection from the reserved1,024-question fresh test.


## Additional paired math training data

`reserve_math_auxiliary.py` has sealed512 additional GSM8K TRAIN questions,
selected by fixed hash before inference. Historical and reserved IDs/stems,
near duplicates, and the frozen observer's training membership are excluded.
Protected JSON records expose only IDs/question text through a field projector;
reference values are skipped without decoding. The pinned Arrow source is
projected to its question column before conversion to Python. This adds no
fresh test inference or grading and leaves3,710 eligible math questions unused.

`collect_auxiliary_math_pairs.py` prepares all seven ordinary solver branches
for every selected question: at most3,584 physical calls, no reviewer calls,
the original128-token root and256-token continuation caps, unchanged prompts,
seeds, temperatures and frozen compact weights. It cannot start before the
evidence-first pilot seals and releases its workers. Each physical attempt is
durable before dispatch, failed answers are retained, and unresolved requests
block automatic replay. No labels or training results have been collected yet.
The goal is to expand math-specific POMDP training from256 to768 questions;
selection will use whole-question folds and keep the stronger existing A.

`math_auxiliary_features.py` prepares local CPU extraction from the same frozen
observer, with one raw20/project256 forward per paid prefix and actual token
slots recorded. `expanded_math_diagnostic.py` prepares three outer and three
inner whole-question folds on all768 math questions. It compares with retained
A's forecast procedure on the same original256 held math questions. All512
new reference rows are joined only after their complete paid collection;
Arrow row selection precedes conversion of reference values to Python. The
registered screen requires lower joint loss and marginal Brier, at least a
two-point conditional-decision accuracy gain, and at most10% extra compute.
This screen is preliminary; it does not claim full-controller or fresh accuracy.

The evidence-first timing pilot completed all320 physical calls with zero
transport errors and restored min0/max2 capacity. All64 questions received
passages (127 total, no fallback). Accuracy was24/64 after one compact repair
and25/64 after the second; the two-endpoint union was26/64. It added zero
questions beyond the50-question known union. Its second trajectory cost was
6,679.21 units,22.70% above the original strong retained-review control. The
registered screen fails; no controller is promoted.

The subsequent512-question math collector is registered and running. It uses
the released existing-app capacity, so the two collection jobs do not overlap.
The full regression suite passes625 tests in38.934 seconds.

`fit_expanded_math.py` and `development_expanded_math.py` prepare the gated
continuation after a successful training screen. The fit changes the math
forecaster and its math-only observation likelihoods; the retained prior,
transition and cost arrays remain identical. Development keeps the best A153,
all previous stronger candidates, exact science decisions and all five original
baselines. New candidates use distinct identities so a retained model cannot be
confused with a new candidate. This complete pipeline passes626 tests; no new
math fit or development result is claimed while collection is incomplete.


## Broader reference corpus, separate pilot

`broad_textbook_corpus.py` seals a question-independent five-course expansion:
[Organic Chemistry I](https://ocw.mit.edu/courses/5-12-organic-chemistry-i-spring-2003/pages/lecture-handouts/),
[Organic Chemistry II](https://ocw.mit.edu/courses/5-13-organic-chemistry-ii-fall-2006/pages/lecture-notes/),
[Quantum Physics III](https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/pages/lecture-notes/),
[Electromagnetism II](https://ocw.mit.edu/courses/8-07-electromagnetism-ii-fall-2012/pages/lecture-notes/),
and [Applied Nuclear Physics](https://ocw.mit.edu/courses/22-02-introduction-to-applied-nuclear-physics-spring-2012/pages/lecture-notes/).
Only lecture-page materials are followed. Explicit answer keys, exams, assignments
and two third-party journal papers are excluded. The private corpus preserves
source attribution and extraction coverage. It adds1,591 passages from72 PDFs:
178 chemistry and1,413 physics passages. Seven documents yield no qualifying text
chunks; their coverage receipts remain. Chemical structures and equations are
not reconstructed, so this is a material limitation of the text-only extraction.
No benchmark records or model calls selected or built the corpus.

`broad_evidence_review_runner.py` registers a separate corpus-only comparison
with the exact64/960/512 reviewer budget and two unchanged compact continuations.
All64 identical paid query calls are reused;256 new physical calls are allowed
after the512-question math collector finishes. All320 phases/continuations are
charged at inference even when reused. Exact full compatible trajectories would
also be reused. Here, all remaining serialized prompts differ: an answer-free
audit finds17 questions with changed passage content,11 receiving newly added
lecture sources. Other prompt differences include changed TF-IDF relevance
scores, so not every changed prompt represents new factual content. The same
six-novel-question/1.25-cost screen and strongest matched-B requirement remain.
The full regression suite passes629 tests in40.519 seconds.
