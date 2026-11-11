# Frozen LLM controller evaluation

Read [report.md](report.md) for the full 60-dev/400-test results and interpretation, [examples.md](examples.md) for selected full paths, and [billing.json](billing.json) for spending and confirmed app shutdown. This artifact is separate from the published dataset characterization and earlier pilot. No POMDP is included.

- `dev/` and `test/`: all raw requests/responses, exact formatted prompts, token usage, per-question paths, aggregate reports, and bootstrap intervals.
- `frozen.json`: binds all 60 completed development questions to the exact experiment before test inference.
- `protocol.json`, `config.json`, `backend.json`, source snapshots: predeclared sequence, model revision, limits, and reproduction provenance.
- `validation.json`: independent offline reconstruction of actions, scores, costs, freeze chronology, split disjointness, and confidence interval endpoints.
- `tests.txt`: 36 relevant passing tests; no cloud calls are needed for these tests.
- `artifact-sha256.json`: hashes every archived file except itself.

The 10-question pilot was repeated as part of all 60 dev questions. No responses were selected among attempts, no prompt settings were changed, and no test-driven tuning was performed. Existing baseline test outcomes had previously been inspected; this is not an untouched benchmark claim. Model output is an action only; no controller explanations are fabricated.

## Reproduce verification without spending

From this repository revision, with Python 3.10+ and no model credentials:

```sh
python3 experiments/llm-controller/verify_full_evaluation.py \
  --run results/llm-controller-full-evaluation
```

The verifier resolves dev/test/freeze inside this bundle and does not follow the original absolute provenance paths in the manifests. It checks those original evidence hashes, reconstructs the actions against the published cache, independently scores the selected answers, sums actual recorded token usage, and reproduces summary and interval arithmetic. It makes no model or network calls. Core code, prompt, reward, and dataset hashes must match the frozen revision. Bootstrap reproduction validates arithmetic, not population coverage or independence assumptions.

`events.jsonl` contains request, raw-response, and accepted-response entries: raw and accepted entries are two records of one call, not two billable calls. Token usage is recorded from actual model tokenization and includes prompt templates and EOS; token IDs were not archived for independent retokenization. Actual monetary billing is separate from research token costs.

Do not rerun paid inference to reproduce these files. The collector has an explicit execution flag, fixed gancone profile, cumulative reservation checks, a fresh-directory guard, disabled function retries, and a separate 30-minute app shutdown watchdog. None of these replaces reconciliation of the provider ledger before another paid run.
