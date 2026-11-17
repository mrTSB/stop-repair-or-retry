# Retry gate with token margins and arithmetic checks

The initial compact solver and short retry use the same Qwen3.5-4B weights, prompt strings, decoding limits, and parsers as the preceding study. The training probes and new development pairs record the five highest probabilities for each generated token. The acquisition script checks that adding these observations preserves the raw answer and token counts on every probed archived question. The fresh test uses the original compact request and its selected-token confidence fields.

The learned gate starts with the paired fix and regression estimators from the parent experiment. A teacher uses the gap between the two most likely tokens, the probability mass of the five recorded tokens, and the token positions around the answer. Cross-fitted teacher predictions are blended with the observed labels to train an optional probability update that uses only the original confidence fields at inference. The updates use out-of-fold base predictions and inverse sampling weights to account for the enrichment of useful training events. When both candidates have been observed, a calibrated comparison chooses which to submit.

The controller can also check a complete arithmetic equality in the compact scratch work. It submits the recomputed value when the scratch work reports the same value as the compact final answer and all terms preceding that reported value agree on a different exact result. This checks arithmetic within the solver’s explanation. The test includes a controller with the arithmetic check and retry-on-invalid, as well as the new learned gate with the check disabled.

The scheduler selects the set of retries with the highest total predicted benefit under an integer token budget derived from retry-on-invalid. It charges an upper bound for each retry, including the full prompt and the maximum possible output. The earlier frozen gate remains a fixed control. The selected policy and all comparator behavior are frozen before the fresh candidate replies are graded.

The independently frozen tight-budget gate fits on the 1,520 archive pairs and the 300 newly completed earlier pairs. It predicts accepted fixes and regressions directly and uses its own answer comparator. Fresh development accuracy and token savings shortlist candidates. The final selection requires a gain in each of the two cross-fitted archive groups. This gate uses the original answer parser and its own retry decisions on the same fresh test. A separate fixed variant submits the exact arithmetic correction whenever one is available, then skips the scheduled retry for that question. Every other routing and answer selection decision comes from the tight-budget gate.

## Data and replay

The base fit uses 1,520 archived pairs. It includes 249 re-probed training questions with probability margins. The previous 300-question test serves as secondary development in this round after collecting its 267 missing retries. The new development split has 180 fully paired questions. The new 600-question test excludes earlier question IDs and normalized stems. The test archive contains all first responses and the union of retries requested by the frozen policies. The solver never receives a reference answer.

Run the archived replay from the repository root with Python 3.12 and NumPy 2.5.4.

```sh
data=results/controller-hillclimb/retry-gate/paired-uplift/margin-gate/data
out=runs/margin-gate-reproduction
for phase in tune plan score; do
  uv run --python 3.12 --with numpy==2.5.4 --no-project python \
    experiments/controller-hillclimb/retry-gate/paired-uplift/margin-gate/margin_study.py "$phase" \
    --data "$data" --output "$out"
  uv run --python 3.12 --with numpy==2.5.4 --no-project python \
    experiments/controller-hillclimb/retry-gate/paired-uplift/margin-gate/tight-budget/study.py "$phase" \
    --data "$data" --output "$out/tight-budget"
  uv run --python 3.12 --with numpy==2.5.4 --no-project python \
    experiments/controller-hillclimb/retry-gate/paired-uplift/margin-gate/tight-budget/arithmetic_check.py "$phase" \
    --data "$data" --output "$out/tight-budget/arithmetic-check"
done
```

The previous `paired-uplift` and `retry-gate` archives are inputs to the fit and control. `study_data.py` can prepare splits, collect full pairs and compact first responses, capture training probes, complete the previous responses, and run the planned test retries. Mean compute counts all paid model tokens per policy, scaled to equivalent 3B tokens. [Test report](../../../../../results/controller-hillclimb/retry-gate/paired-uplift/margin-gate/README.md)
