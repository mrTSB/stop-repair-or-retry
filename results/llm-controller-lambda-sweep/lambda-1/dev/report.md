# LLM controller and five fixed baselines

Split: dev; questions: 60; token penalty: 1.0.

| Method | Accuracy | Mean total tokens | Mean utility |
|---|---:|---:|---:|
| llm_controller | 50.00% | 1543.0 | -1.0430 |
| stop | 41.67% | 325.6 | 0.0911 |
| repair_once | 55.00% | 779.2 | -0.2292 |
| repair_twice | 53.33% | 1253.0 | -0.7197 |
| retry_once | 41.67% | 650.8 | -0.2341 |
| retry_twice | 40.00% | 975.4 | -0.5754 |

The controller pays for every executed input/output token, including a decision to stop. Baselines use only the same visited solver cache. No fitted controller or POMDP is loaded. Invalid action outputs submit the current answer and remain counted.

Intervals are paired question bootstrap estimates conditional on one fixed cache and controller run. Development results are exploratory; comparisons against five baselines are not adjusted for multiple testing. This report does not establish a confirmatory win.

Only actual controller request latency is measured. Cached solver batch times are not serial serving latency.
Token-price estimates are not invoices. Compute/idle/startup charges require the provider ledger. Budgets are per run, not account-wide.
Existing test baseline outcomes have been inspected; use fresh data if those outcomes informed redesign.
