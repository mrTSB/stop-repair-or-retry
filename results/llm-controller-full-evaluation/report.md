# Frozen LLM controller: full development and test evaluation

The same Qwen2.5-14B-Instruct controller was run on all 60 development questions, frozen, and then run on all 400 test questions. Model weights, prompt, reward, decoding, action parsing, and observation fields were unchanged from the small pilot. No POMDP was fitted or evaluated, and no earlier controller results were imported.

## Test results

| Method | Correct | Accuracy | Mean total tokens | Mean utility |
|---|---:|---:|---:|---:|
| LLM controller | 177/400 | 44.25% | 1,616.0 | 0.42634 |
| Submit initial | 145/400 | 36.25% | 340.1 | 0.35910 |
| Repair once | 211/400 | 52.75% | 833.0 | 0.51917 |
| Repair twice | 219/400 | 54.75% | 1,358.3 | 0.53392 |
| Retry once | 153/400 | 38.25% | 679.6 | 0.37570 |
| Retry twice | 161/400 | 40.25% | 1,020.7 | 0.39229 |

| Domain | Questions | LLM accuracy | LLM mean tokens | Repair-once accuracy | Repair-once mean tokens |
|---|---:|---:|---:|---:|---:|
| math | 200 | 52.50% | 1,334.1 | 65.50% | 652.2 |
| science | 200 | 36.00% | 1,897.8 | 40.00% | 1,013.7 |

Observed action sequences: repair → repair: 3; repair → retry: 16; repair → stop: 139; retry → repair: 1; stop: 241.

Valid/invalid decisions: 559/0. Initially wrong → finally correct: 36. Initially correct → finally wrong: 4.

Mean solver tokens: 571.3; mean controller tokens: 1,044.7.

### Paired utility comparisons on test

Positive differences favor the LLM controller. Utility is correctness minus 0.01 × total research tokens / 1,000. Intervals use 2,000 paired question bootstrap resamples, conditional on this fixed solver cache and single controller run; they are not adjusted for five comparisons.

| Comparator | Mean LLM minus baseline utility | 95% paired bootstrap interval |
|---|---:|---|
| Submit initial | +0.06724 | [+0.03719, +0.09745] |
| Repair once | -0.09283 | [-0.12545, -0.06265] |
| Repair twice | -0.10758 | [-0.14196, -0.07534] |
| Retry once | +0.05064 | [+0.00223, +0.09846] |
| Retry twice | +0.03405 | [-0.01369, +0.08134] |

## What the test result shows

The LLM controller improved on submitting the initial answer by 8 percentage points, but repair once was 8.5 points more accurate and used about 783 fewer tokens per question. Repair twice was also more accurate and used fewer tokens than the LLM controller. Thus both repair baselines dominated this controller on these two measured outcomes. This is a result for this fixed prompt, model, cache, and scoring setup, not a claim about all LLM controllers.

The controller stopped immediately on 241 test questions; 117 of those submitted answers were wrong. On 37 of these wrong stops, the cached repair-once answer was correct. Across all 400 questions, the LLM controller was correct and repair once wrong on 5, while the reverse held on 39. These are post-evaluation diagnostic comparisons, not information available during decisions. They identify premature stopping as a substantial source of missed improvements. The diagnostic examples are selected by explicit rules in `examples.md`.

## Development results

| Method | Correct | Accuracy | Mean total tokens | Mean utility |
|---|---:|---:|---:|---:|
| LLM controller | 30/60 | 50.00% | 1,544.4 | 0.48456 |
| Submit initial | 25/60 | 41.67% | 325.6 | 0.41341 |
| Repair once | 33/60 | 55.00% | 779.2 | 0.54221 |
| Repair twice | 32/60 | 53.33% | 1,253.0 | 0.52080 |
| Retry once | 25/60 | 41.67% | 650.8 | 0.41016 |
| Retry twice | 24/60 | 40.00% | 975.4 | 0.39025 |

| Domain | Questions | LLM accuracy | LLM mean tokens | Repair-once accuracy | Repair-once mean tokens |
|---|---:|---:|---:|---:|---:|
| math | 30 | 66.67% | 1,261.6 | 76.67% | 609.0 |
| science | 30 | 33.33% | 1,827.2 | 33.33% | 949.4 |

Observed action sequences: repair → retry: 1; repair → stop: 25; stop: 34.

Valid/invalid decisions: 86/0. Initially wrong → finally correct: 6. Initially correct → finally wrong: 1.

Mean solver tokens: 522.9; mean controller tokens: 1,021.5.

## Protocol and interpretation

The project has 100 training, 60 development (30 math/30 science), and 400 test (200 math/200 science) questions. IDs are disjoint across project splits. The original ten-question pilot is a subset of dev and was recollected once as part of this full run; no response was selected among repetitions. Training was not used to fit this prompt-only controller. Existing fixed-baseline test outcomes were inspected before this experiment, so this test is not represented as a previously unseen benchmark. No controller settings were selected using test outcomes.

Controller model: `Qwen/Qwen2.5-14B-Instruct`, revision `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, bfloat16, greedy decoding, output cap 128 tokens. Cached solver: Qwen2.5-3B-Instruct. The controller emits only a JSON action; it does not supply a replacement answer. There is no constrained decoding or JSON repair. Invalid outputs count as stopping and remain reported.

Actions are stop (submit current answer), repair (revise the latest response), or retry (fresh attempt). At most two additional solver calls are allowed; each replaces the current answer. The controller sees the question, remaining call budget, declared penalty, and only the visited responses. Gold answers, correctness labels, source labels, and unvisited branches are withheld. All visited solver and controller input/output tokens count, including stopping, prompt templates, and EOS. The same five fixed baselines replay the same questions. Research token totals combine differently sized models and are not dollar costs.

The operational config permits up to 800 requests per split, with a global maximum of 920. One private Modal app serves both splits; a separate watchdog process stops it after 1,800 seconds if needed. GPU resources are one A100-80GB, four CPU cores, and 48 GiB RAM, with function retries disabled. Endpoint text in the config is descriptive: inference uses authenticated Modal RPC, not an HTTP endpoint.

Frozen evidence binds the completed dev journal and manifest to the exact config, prompt, model identity, code, dataset manifest, and reward design. Test collection and evaluation enforce this freeze. The driver froze settings immediately after dev evaluation, with no tuning or outcome-dependent branching. The core source and driver are archived for provenance.

Controller RPC latency includes transport; GPU generation time is also recorded. Model startup is a collection cost, not per-question inference. Cached solver batch timing cannot be interpreted as serial response latency. No solver inference was purchased for this evaluation.

## Evidence and verification

See `dev/` and `test/` for raw journals, manifests, summaries, and per-question outputs; `frozen.json` for the dev freeze; `validation.json` for independent verification; and `billing.json` for cumulative posted charges and reservations. `artifact-sha256.json` hashes the archived evidence. Detailed evaluation metrics must be interpreted alongside finite-sample uncertainty and the single cached solver realization. No claim of a trained or tuned POMDP is made.


## Spending and shutdown

The full dev/test app finished in 509.24 seconds (8.49 minutes), with 645 controller calls. Modal confirms it and both pilot apps are stopped with zero tasks. The watchdog process was canceled after normal completion and is no longer running.

Posted cumulative project charges are **$3.0006**. The latest full-run charge and earlier successful-pilot charge have not yet posted. Using full app lifetimes and requested resources gives approximately **$0.43** for this full run and **$3.51 cumulative compute** including prior work and the pilot. These are estimates, not final billed totals. The retained cumulative spending reservation is **$7.65**, below the $20 cap; reconcile pending charges before any future paid run. The failed pilot startup currently has zero posted CPU/GPU/RAM charges. `billing-before.json`, `billing.json`, and `protocol.json` retain the distinctions among posted cost, estimates, and reservations.
