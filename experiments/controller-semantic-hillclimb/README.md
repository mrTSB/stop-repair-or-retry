# Semantic observations and paired transition forecasts

This continuation is separate from the archived `controller-hillclimb` study. Its
goal is to improve the controllers while preserving the frozen compact solver,
the stop/repair/retry action space, and the two-continuation solver budget.

The requested margins are percentage points: A and B must each exceed the best
fixed baseline by 10 points, and C must exceed the higher of A/B by 10 points
while using less compute than B. These are evaluation targets, not assumed
outcomes. Strong reference policies remain in the comparison.

The baselines are **Stop, Repair once, Repair twice, Retry once, Retry twice**.
Each runs the same frozen solver. No solver model name is a separate baseline.
Repairs check the current solution; retries start from the original question.
All fixed policies submit their latest answer. Their original prompts and token
limits are preserved.

A is a four-state POMDP over persistent question difficulty and current-answer
correctness. Its observations come from a trained discriminative encoder and
visible response features. It never acquires an LLM review. B uses a frozen LLM
reviewer without a POMDP. C uses the POMDP to decide whether a paid LLM review is
worth acquiring, updates its belief, and selects stop, repair, or retry.

The runtime directory is `runs/controller-semantic-hillclimb/` and is ignored by
Git. Earlier panels, including the previously reported fresh800 panel, are
retired. There are 1,472 training questions and 240 development questions. A
separate 192-question same-source panel (96 math / 48 chemistry / 48 physics)
is reserved for one frozen confirmation. IDs and normalized question stems are
checked against recorded historical exclusions. Development scores and training
pilot scores must never be described as fresh-test results.

The initial semantic model uses DeBERTa-v3-base. The grounded model starts from
the pinned MNLI/FEVER/ANLI encoder and adds all answer options plus retrieved
OpenStax exposition. Retrieval uses the question stem, excludes answer keys and
worked examples, and supplies the same background for every candidate answer.
The grounded information condition must be disclosed when interpreting a gain.
Its source corpus and academic CC BY-NC-SA attribution are recorded separately.
The teacher-review grounding pilot did not improve accuracy and is retained as
a failed trial.

Training uses three question-grouped folds. Solver/reviewer correctness and
paired successor outcomes are training labels only. Out-of-fold neural outputs
fit the POMDP observation and transition models. `tree_context_pomdp.py` learns
conditional transition forecasts from those paired outcomes; hidden states are
enumerated at inference, never read from reference labels.

`stable_pomdp.py` adds a 1e-9 tolerance to review information values. This avoids
buying a review when a mathematically zero value rounds slightly positive. The
original preliminary run remains intact; the stable run independently passed
2,160 charge checks and 1,920 blind, paid-path-only development replays.

Compute is reported as a proxy, not measured FLOPs:

```
(4 * solver input/output tokens
 + 3 * LLM-review input/output tokens
 + encoder parameter billions * encoder padded token slots) / 3.09
```

The 35B reviewer is a mixture-of-experts model; the primary proxy uses its active
parameter count. Raw tokens, calls, and a total-parameter sensitivity are also
reported. Every observation-model invocation is charged even when its raw
prediction is cached. Longer reasoning reviews charge every physical API call,
including the answer phase after an incomplete bounded scratchpad. Completed
wrong answers are never retried according to correctness.

`dynamic_grounded_pomdp.py` removes masked trailing padding before each online
encoder pass. It charges the actual unpadded token slots. Its planning model
retains the conservative training-cap estimate for future encoder costs.
Development accuracy matched the padded implementation while measured input
slots fell; real-encoder regression tests check that the transformation
preserves logits. The nonlinear calibration trials did not improve accuracy.

Use `uv` for all Python commands. The working dependency pins are Modal 1.6.4,
PyTorch 2.9.0, Transformers 4.57.1, Safetensors 0.6.2, NumPy 2.5.4,
scikit-learn 1.8.0, SentencePiece 0.2.1, and Protobuf 5.29.5. Modal uses the
existing `mrtsb` profile; credentials do not belong in source or reports.

`development*.py` select policies using development data and can acquire only
missing development paths. Run collectors serially against the shared bank.
`development_audit.py` reconciles costs and removes labels and unpaid responses
before replaying decisions. `fresh_study.py`, `verify_development.py`,
`fresh_collect.py`, and `fresh_audit.py` implement freeze, independent training
refit, blind union collection, and final grading/audit. The fresh collector
archives exact reference-free neural predictions and refuses a changed freeze.
No test scoring is allowed before all 192 questions finish.

Completed model checkpoints are retained in the private Modal volume
`srr-semantic-training-20261120`, under `/verifier-v2` and
`/verifier-grounded-v1`. `fetch_verifier.py` retrieves the full model and checks
every weight shard against the training recipe. Full checkpoint assets are
large; do not silently omit their provenance or replace them with a refitted
model when reproducing a frozen result.

The larger-observer and selective deeper-review continuation is documented in
[followup/README.md](followup/README.md). It keeps the original192scores hidden
while reserving a prospective912-question validation with separate source-panel
reports. Its current figures are explicitly labeled as development results.
