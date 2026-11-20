# Contextual POMDP and paid-review experiment

This child experiment preserves the previous `pomdp-abc` study and its frozen results. It targets held-out accuracy strictly above 65% for A and B and above 75% for C. The completed 600-question evaluation achieved A 72.50%, B 82.33%, and C 81.67%; C used 10.92% less active-parameter compute than Repair twice. [Results and replay archive](../../../../results/controller-hillclimb/pomdp-abc/guided-repair/README.md) include the complete baseline table and uncertainty.

The solver is the same Qwen3.5-4B checkpoint and quantization, identified by its complete Ollama manifest digest. The initial prompt and 128-token output cap remain unchanged. Training-only paired pilots motivated a shared 256-token continuation budget and up to 90 words of scratch work. Every baseline and controller uses the same continuation templates and decoding settings. A maximum of two additional solver calls is permitted. Retry starts a fresh context. Repair checks the current solution. Stop submits the current answer.

The baseline families are **Stop, Repair, and Retry**. The final evaluation includes all five fixed variants from the original table: Stop, Repair once, Repair twice, Retry once, and Retry twice. Development selects the primary comparator by accuracy, with lower compute breaking ties. Model names identify resources; they are not additional baselines.

**A is a POMDP.** Its hidden state is current-answer correctness. The remaining call budget and question/response features are observed. A grouped five-fold classifier produces observable confidence categories; all responses for a question belong to the same fold. Training learns action-dependent correctness transitions, response-observation likelihoods, and action costs. Contextual transition regressors estimate whether an action is likely to fix or damage the current answer, conditional on each hidden correctness state. An explicit two-step Bellman recursion plans over a 501-point belief grid. After every solver call, A predicts the next state and applies a Bayesian observation update. A never acquires an LLM-controller response.

**B is an LLM controller.** A frozen Qwen3.5-35B-A3B reviewer independently works through the problem, checks the current final answer, and returns an action and correctness assessment. Development selects its direct action or a fixed confidence/agreement rule. B's inference function does not call the POMDP or its classifier.

**C is a hybrid POMDP and LLM controller.** It uses the same reviewer as B and queries when the learned expected decision benefit exceeds its estimated cost and a development-selected threshold. The reviewer observation includes its action, confidence category, and whether its independent answer agrees with the compact answer. Empirical likelihoods condition on the already observed compact-response confidence category. C performs a Bayesian belief update and replans with review-conditioned action transitions. The acquisition calculation is a one-step estimate of information value, using the remaining solver horizon. It does not recursively optimize future review acquisition.

Development also tests reusing an already paid review after the next solver response. C learns a conditional observation model for agreement between the new compact answer and the earlier reviewer's independent answer, and incorporates it in its Bayesian filter and information-value calculation. B receives a matching reuse candidate that stops when the new answer agrees with the paid review. Reuse incurs no additional LLM call; the original review and any repair prompt containing it remain fully charged. Current contextual action estimates use observed response features, while future Bellman values use pooled stage-specific dynamics.

If B or C acquires a review at the initial checkpoint and chooses Repair, the compact solver receives that paid review through the same repair template. Later repairs use the current solution without passing another review. `guided` is a cached path label for a reviewed first repair, not a fourth action. The paired collection contains ten possible solver nodes and four possible review checkpoints per question. A deployed policy pays only for its visited nodes and acquired reviews.

Training contains 192 questions: 48 earlier training-pilot questions and 144 retired study-test questions. Development contains 120 disjoint questions from that retired study test. These retired questions are explicitly development/training evidence, not fresh validation. Auxiliary initial-answer classification examples come from earlier archives with every current training, development, and test ID excluded. Question-grouped cross-fitting prevents the same question's other responses from entering its out-of-fold predictions.

The reserved test contains 600 unused IDs and normalized question stems: 300 GSM8K official training-source questions, 150 MMLU-Pro chemistry questions, and 150 MMLU-Pro physics questions. The unused biology pool is exhausted. This is a study holdout with a stated subject mixture, rather than a result on the official GSM8K test split or a direct repeat of the previous panels.

Reviews use ordinary JSON text rather than constrained JSON mode. An early backend check found that constrained JSON can omit completion/token metadata when a review hits its generation limit. All earlier review and dependent-repair nodes were retired together and regenerated under the ordinary-text protocol. Completed malformed or truncated JSON is retained with its full token charge and the declared fallback action. The parser tolerates invalid LaTeX escaping; it does not invent missing verdict fields.

Selection maximizes development accuracy among candidates no more expensive than the strongest development-selected fixed baseline, with lower compute breaking ties. A family with no feasible candidate reports its most accurate candidate and its excess cost. A selected C must actually acquire LLM evidence on development. Matched C-without-review, C-always-review, and random-acquisition controls are archived to distinguish acquisition effects from planner-fit differences.

Compute charges all input and output tokens. The primary proxy weights solver tokens by 4B and reviewer tokens by 3B active parameters, divided by 3.09B for continuity with earlier reports. The reviewer's [official model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) specifies 35B total and 3B activated parameters. Results also retain a conservative total-parameter sensitivity using 35B for the reviewer, raw token counts, and call counts. The active-parameter proxy does not measure GPU latency, energy, or memory. Offline training and branch collection are separate from reported inference cost.

The private Modal worker serves the exact local model manifests and verifies every downloaded public model blob by SHA-256. Its pool permits at most four L40S GPUs, targeting two concurrent inputs per container, with a two-minute idle shutdown. Collection uses eight concurrent requests in total. Model-cache preparation runs on CPU. No credentials are stored in experiment files. Earlier cached responses and new local/GPU responses retain their model and prompt fingerprints; new responses also record the inference backend.

The decision models were frozen before test collection. An initial unscored seven-question run exposed missing completion/token metadata in the 0.19.0 GPU runtime, in both streamed and non-streamed responses. The 0.40.16 runtime returned complete receipts with the identical model digests. Its deployment image was the only source change in a recorded runtime amendment. The original freeze is retained, and the selected-model/settings/baseline hash remains `77e57d3bc8f7cc1055c8e73581d61ae0f3af62c5565eb7f8d46b7e47c21a570c`. All 600 test questions are recollected on the updated runtime. No test accuracy was examined and no policy was fitted or selected again during this accounting fix.

## Commands

Local runtime data lives under `runs/controller-hillclimb/pomdp-abc/guided-repair/extended` and is ignored by Git. Use `uv` for Python and the `mrtsb` Modal profile for GPU work.

```sh
code=experiments/controller-hillclimb/pomdp-abc/guided-repair
data=runs/controller-hillclimb/pomdp-abc/guided-repair/extended
out=runs/controller-hillclimb/pomdp-abc/guided-repair/selection

MODAL_PROFILE=mrtsb uv run --with modal==1.3.5 --no-project \
  modal run "$code/modal_ollama.py::prepare_weights"
MODAL_PROFILE=mrtsb uv run --with modal==1.3.5 --no-project \
  modal deploy "$code/modal_ollama.py"
MODAL_PROFILE=mrtsb uv run --with modal==1.3.5 --with numpy==2.5.4 --no-project \
  python "$code/extended_acquire.py" collect --data "$data" \
  --split all --phase full --backend modal --workers 8

uv run --no-sync python "$code/extended_study.py" select --data "$data" --output "$out"
MODAL_PROFILE=mrtsb uv run --with modal==1.3.5 --with numpy==2.5.4 --no-project \
  python "$code/extended_acquire.py" holdout --data "$data" \
  --frozen "$out/frozen.json" --backend modal --workers 8
uv run --no-sync python "$code/extended_study.py" score --data "$data" --output "$out"
uv run --no-sync python "$code/audit.py" --data "$data" --output "$out"
```

`prepare` creates the split manifest from the cached source datasets; `seed` copies unchanged initial responses from the preceding experiment. `collect` appends individual solver/reviewer observations under a process lock and can resume after interruption. `holdout` first checks frozen code and training/development evidence, then collects the union of paths required by the frozen methods. It replaces reference answers with empty strings during policy replay and prints progress only. `score` refuses an incomplete test panel. `audit` independently reconciles charges and repeats each policy using only its paid observations, with hidden labels removed.

Thirteen focused unit tests check Bayesian filtering, Bellman values, paid-review reuse, review parsing and costs, serialization, role separation, and hidden-reference invariance. Full archival replay and test audits are required before reporting validated results.
