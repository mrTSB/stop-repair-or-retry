# Controller B: two-state POMDP

Read [DESIGN.md](DESIGN.md) first. It records the user's selected scope, the assistant's declared defaults, and the final-project extensions that are deliberately deferred. The original design is preserved; a separate four-state diagnostic is now available below.

This standalone workflow uses only the published cached math/science dataset. It fits a new two-state model on 100 training questions, evaluates all 60 dev questions, freezes the model and evidence, and then replays all 400 test questions. No Modal account, router, API key, model inference, or older fitted controller is used. No additional one-step baseline is included.

## Model

The hidden state is incorrect/correct under the published frozen scoring rule. Confidence (low/medium/high/missing) and answer-format validity form eight observation symbols. This is an automated correctness label, not an independently verified semantic judgment. Known remaining budget is not a hidden state. A generic belief vector and state-dimensional matrices support future estimators with four states.

The training estimator uses 100 roots for initial correctness and initial observations, and 300 edges per action for repair/retry dynamics and action-conditioned successor observations. It adds a pseudocount of one per probability cell. Expected action costs are conditional on current state, with a pooled-action fallback for empty cells. The tables include sample counts for audit. Branches from one question are correlated, and correctness alone may not make the process Markov; these are model approximations.

For belief b, stopping has value b·r. An action's value is its negative expected token cost plus the expectation of the optimal next value over successor observations. Predict b′=bT, multiply by the action-conditioned observation likelihood, and normalize. At zero remaining calls, submit. Root tokens are sunk for decisions but still count in reported episode cost. Belief updating uses neither reference answers nor unseen branch responses.

## Reproducible offline commands

Use a fresh output directory; the commands refuse overwriting existing stages. These commands make no network requests:

```sh
PYTHONPATH=src python3 -m stop_repair_retry.pomdp_experiment fit --run runs/pomdp-two-state-v1
PYTHONPATH=src python3 -m stop_repair_retry.pomdp_experiment dev --run runs/pomdp-two-state-v1
# Inspect dev for correctness of implementation. Do not retune the declared experiment.
PYTHONPATH=src python3 -m stop_repair_retry.pomdp_experiment freeze --run runs/pomdp-two-state-v1
PYTHONPATH=src python3 -m stop_repair_retry.pomdp_experiment test --run runs/pomdp-two-state-v1
PYTHONPATH=src python3 -m unittest discover -s tests -p test_pomdp_experiment.py
```

The primary penalty is 0.01. All five predeclared penalties [0, 0.001, 0.01, 0.1, 1] are evaluated separately. The five fixed baselines appear throughout. Controller A appears only at its original 0.01 penalty; it was not re-prompted for this sweep. CPU time is recorded separately from research tokens. No extra LLM-token cost is assigned to the POMDP's arithmetic.

Each path file records observations, beliefs, action values, chosen actions, final path, actual tokens, scores, and CPU timing. The freeze binds dev evidence to model, configuration, source/helper, design record, dataset, and Controller A hashes. Full dev replay and summary reconstruction are required before freezing. Do not edit the frozen design or code and present a later run as the same experiment; create a new version when extending it.

## Milestone run

The [frozen milestone archive](../../results/pomdp-two-state/README.md) contains the model, full dev/test replay, sensitivity sweep, validation, and design snapshot. At the primary penalty the policy coincides with repair twice on every evaluated question. This is documented as a limitation, not an adaptive improvement.

## Follow-up lambda diagnostic

A [separate train/dev-only analysis](../../results/pomdp-lambda-diagnostic/report.md) refines the sweep to 602 points and checks model-derived switch points. It finds mixed policies and lower token cost, but no dev accuracy gain. It does not replace the frozen primary result or add a new test result. Independent offline checks are available in `verify_lambda_sweep.py` and `verify_policy_regions.py`.

## Separately frozen lambda 0.248 follow-up

The user selected lambda0.248 from the dev-only diagnostic. The [separate frozen test experiment](../../results/pomdp-lambda-0248/README.md) obtains 53.00% accuracy at 913.49 mean tokens. It beats Controller A on these metrics but exceeds repair once by only one correct answer with more tokens; it remains below repair-twice accuracy. Its freeze, raw paths, both utility weightings, audit, and prior-test-exposure caveat are preserved separately. Use `frozen_lambda.py verify` for offline reproduction.

## Known-context improvement diagnostic

A [train/dev-only context-conditioned candidate](../../results/pomdp-context-diagnostic/report.md) keeps two states and existing observations but separates dynamics by prior action. It matches repair-once dev accuracy with a small token saving at0.248; no test evaluation is performed. The report preserves the diagnostic evidence, fixed shrinkage choice, and the caveat that early stops involve unseen confidence bins. Confidence backoff and history-aware answer-change observations remain future experiments, not silent milestone changes.

## Calibration cross-validation

A [five-fold training diagnostic](../../results/pomdp-calibration-cv/report.md) compares pooled, context, and unsupported-category-calibrated models, then evaluates one training-selected variant on dev. Calibration does not yield a new accuracy gain; no answer-change features or test run were added. Fold models are fitted only on80 questions, with every question's branches kept together. This diagnostic motivates a separately authorized richer-observation experiment rather than more lambda tuning.

## Four-state latent-model diagnostic

The user requested additional hidden states. A [separate train-CV diagnostic](../../results/pomdp-latent-state-cv/report.md) compares one correct plus three latent incorrect types with a matched two-state control, keeping confidence/validity observations. Four states show no accuracy gain. Only the training-selected two-state candidate is evaluated on dev; no new test evaluation or inference spending is added. The report records initialization, weak state support, fixed iterations, model assumptions and deferred observation improvements.
