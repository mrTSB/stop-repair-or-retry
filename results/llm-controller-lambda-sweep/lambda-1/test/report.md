# LLM controller and five fixed baselines

Split: test; questions: 400; token penalty: 1.0.

| Method | Accuracy | Mean total tokens | Mean utility |
|---|---:|---:|---:|
| llm_controller | 43.75% | 1613.9 | -1.1764 |
| stop | 36.25% | 340.1 | 0.0224 |
| repair_once | 52.75% | 833.0 | -0.3054 |
| repair_twice | 54.75% | 1358.3 | -0.8108 |
| retry_once | 38.25% | 679.6 | -0.2971 |
| retry_twice | 40.25% | 1020.7 | -0.6182 |

The controller pays for every executed input/output token, including a decision to stop. Baselines use only the same visited solver cache. No fitted controller or POMDP is loaded. Invalid action outputs submit the current answer and remain counted.

Intervals are paired question bootstrap estimates conditional on one fixed cache and controller run. Development results are exploratory; comparisons against five baselines are not adjusted for multiple testing. This report does not establish a confirmatory win.

Only actual controller request latency is measured. Cached solver batch times are not serial serving latency.
Token-price estimates are not invoices. Compute/idle/startup charges require the provider ledger. Budgets are per run, not account-wide.
Existing test baseline outcomes have been inspected; use fresh data if those outcomes informed redesign.
