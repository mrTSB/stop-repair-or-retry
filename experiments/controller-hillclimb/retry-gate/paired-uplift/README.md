# Paired retry uplift gate

The initial compact call and short retry use the previous Qwen3.5-4B weights, prompts, decoding limits, and answer parsers. The classifier estimates a retry fix as an initially wrong answer followed by a correct short answer. A second model chooses between the observed answers after a retry. The controller estimates the rates of accepted fixes and regressions using out-of-fold answer comparisons. It sees only the question and compact response when deciding to retry.

The fit uses 480 new paired responses plus the 920 previously recorded pairs. The older test pairs are training data in this experiment. The 120 development questions and 300 test questions are new to the controller studies. The selection manifest records source hashes and the exact question IDs and texts. Question sampling excludes earlier study IDs and normalized stems.

The allocator protects part of the expected gain from retrying on invalid answers. It spends the remaining budget on the largest predicted net gains per token. It credits a fraction of the invalid baseline’s retry input tokens and debits the maximum possible input and output tokens for each planned retry. The compact and short prompt inputs differ by a fixed 8 math tokens or 26 science tokens with the pinned tokenizer. The runner verifies those offsets against actual responses. All policies pay the same compact first-call cost.

## Reproduce the experiment

Run the three stages from the repository root with Python 3.12 and NumPy 2.5.4. The archive supplies the full paired training and development responses, test first responses, and test retries required by the frozen plan. The scoring step charges each policy for its own requests, even when a retry is shared across policies for evaluation.

```sh
data=results/controller-hillclimb/retry-gate/paired-uplift/data
out=runs/paired-uplift-reproduction
for phase in tune plan score; do
  uv run --python 3.12 --with numpy==2.5.4 --no-project python \
    experiments/controller-hillclimb/retry-gate/paired-uplift/experiment.py "$phase" \
    --data "$data" --output "$out"
done
```

The live collection helper uses `pairs` for training and development and `first` for the unseen test questions. After `tune` and `plan`, `retries` takes `--plan` to collect only the requests needed for the frozen test comparison. Outputs are resumable by question ID. Each endpoint call is made to the local Ollama model and checks its weight digest. [Evaluation results](../../../../results/controller-hillclimb/retry-gate/paired-uplift/README.md)
