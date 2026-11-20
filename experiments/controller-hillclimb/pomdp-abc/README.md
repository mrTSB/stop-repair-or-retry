# POMDP, LLM judge, and selective judge experiment

This is the archived 420-question study. The [guided-repair continuation](guided-repair/README.md) preserves these results and meets the requested A/B/C accuracy targets on a separate 600-question holdout.

This experiment holds the Qwen3.5-4B solver, action prompts, generation limits, question panels, and paired solver branches fixed across all methods. It evaluates Stop, Repair, and Retry as baselines. Repair and Retry each use a constant depth of one or two, selected on development data. Stop submits the initial answer. All adaptive controllers have at most two additional solver calls.

A explicitly tracks hidden answer correctness. A classifier trained on question and observed response features produces a categorical continuation observation. Development selects either a binned initial belief or a continuous probability calibrated on out-of-fold initial scores. The POMDP estimates correctness transitions, observation likelihoods, and action costs from paired branches. It performs Bayesian belief updates after each observed response and solves a two-step Bellman recursion on a 501-point belief grid. Initial classifier calibration uses archived questions, and its training predictions are cross-fitted by question. The search considers the existing linear and contextual feature bases.

B asks the installed Qwen3.5-9B judge for a terse verification, probability of current-answer correctness, and a stop, repair, or retry action. Development selects either the judge's direct action or a fixed threshold on its reported correctness probability. C uses the same judge and updates its POMDP belief with the judge observation. A learned conditional observation table accounts for the POMDP's existing belief category. The observed judge category also conditions the estimated repair and retry transitions. C consults the judge when the expected decision value exceeds the judge cost and a development-selected threshold. This information acquisition calculation uses a one-step estimate of the judge's benefit at the current decision.

The judge's full input and output tokens are charged at the 9B-to-4B parameter ratio. Solver and judge tokens are then expressed in 3.09B-equivalent units. The acquisition model estimates separate costs for the initial and subsequent checkpoint. The reported inference proxy excludes offline collection and model fitting. A random acquisition control fixes its query probability on development data. Additional controls apply C's fitted planner and policy settings while acquiring evidence at every decision or at no decision. These controls are included in the archived analysis alongside the six primary methods.

Baseline depths are selected by development accuracy, with lower compute breaking ties. All three controllers are selected by development accuracy minus 0.05 times mean solver-equivalent tokens divided by 1000. The initial grid is followed by local refinement of the four strongest planner fits for each family. Judge confidence uses five categories separated at 0.2, 0.8, 0.97, and 0.995. The high-confidence split was motivated by training verdicts that differed sharply in correctness at reported probabilities 0.95 and 1.0.

The training tree panel contains 144 archived questions. Development contains 120 earlier development questions. The fresh 420-question study holdout contains 210 unused GSM8K official training questions and 210 unused MMLU-Pro questions. Only 28 biology questions remain outside recorded earlier uses and exclusions, so the science panel contains 28 biology, 91 chemistry, and 91 physics questions. Its subject mixture differs from the earlier balanced science panels. This holdout measures generalization to new questions drawn for this study.

## Collection and reproduction

The default local collection writes to ignored `runs/controller-hillclimb/pomdp-abc`. The selection manifest freezes the question IDs, source hashes, and counts. Solver responses retain their prompt hashes, model digest, sampled decoding seed, raw answer, and token counts. Training and development prompts can be improved before freezing the selected policies.

`acquire.py prepare` creates the split files and archived initial responses. `acquire.py trees` collects seven paired responses per question. Initial answers use the existing compact prompt at temperature zero. Repair includes the previous response. Independent retries use the initial question in a fresh context. Continuations use a deterministic question-and-path seed at temperature 0.65. `acquire.py judges` collects the three possible decision checkpoints.

`acquire.py probes` recaptures missing initial token probabilities. It checks the raw text, parsed answer, finish reason, prompt hash, and both token counts against the archive. Only exact matches supply extra confidence features.

`study.py tune` fits the models and selects policy settings using training and development. `acquire.py holdout` collects the union of solver and judge calls requested by the frozen policies and controls. It replaces references with empty strings during policy replay and saves each generated solver response to a recovery journal. Completed question records contain the branches needed to replay every frozen policy. Training and development retain all seven solver branches. `study.py score` charges each method for its visited solver and acquired judge calls. Gold answers are used for fitting and grading. Policy features, solver prompts, and judge prompts use only the question and observed responses.

The packed response archive supports fitting and scoring on CPU with Python 3.12 and NumPy 2.5.4. Run from the repository root. The preceding controller experiment archives are also inputs to the initial classifier.

```sh
data=results/controller-hillclimb/pomdp-abc/data
out=runs/pomdp-abc-replay
for phase in tune score; do
  uv run --python 3.12 --with numpy==2.5.4 --no-project python \
    experiments/controller-hillclimb/pomdp-abc/study.py "$phase" \
    --data "$data" --output "$out"
done
```

Generating new observations requires the matching local model weights. Repeat an interrupted collection command with the same data and freeze files to resume. The recovery journal retains completed solver calls within partially finished questions. Regeneration can vary with the inference backend. The archived responses are the inputs to the reported paired replay.
