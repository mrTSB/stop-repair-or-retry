# LLM controller dev pilot

Read [pilot-report.md](pilot-report.md) for results and limitations; [walkthroughs.md](walkthroughs.md) shows all ten complete executed paths. [billing.json](billing.json) distinguishes posted charges from estimated/reserved spending. [validation.json](validation.json) contains the independent audit.

This is an exploratory development artifact, separate from the frozen solver dataset and final baseline characterization. The data and code snapshots preserve real inference; no POMDP or older controller outcomes are used. The original collection directory is retained under ignored `runs/llm-controller/modal-dev-pilot-qwen25-14b-20261110-v2`.

`events.jsonl` includes request, raw response, and accepted response records. Raw and accepted entries refer to the same call: never charge them twice. Model responses contain token usage and exact formatted prompts. The failed-startup folder preserves the separate failed launch; it contains no model decisions. The config endpoint is a descriptive Modal app address, not an HTTP inference endpoint.

Offline re-evaluation from this repository revision (no Modal calls or keys):

```sh
PYTHONPATH=src python3 -m stop_repair_retry.llm_experiment evaluate --run results/llm-controller-dev-pilot
```

The evaluator rejects changed core code, prompt, reward, or dataset hashes. Frozen source snapshots are archival evidence; the current repository code supplies imports and dataset paths. Run re-evaluation on a copy to preserve the archived report files. `artifact-sha256.json` covers every archived file except itself. Do not rerun paid collection to reproduce these results. Any future paid run must first reconcile cumulative billing and use a new output directory.
