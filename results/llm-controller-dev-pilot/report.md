# LLM controller and five fixed baselines

Split: dev; questions: 10; token penalty: 0.01.

| Method | Accuracy | Mean total tokens | Mean utility |
|---|---:|---:|---:|
| llm_controller | 80.00% | 1657.3 | 0.7834 |
| stop | 50.00% | 326.5 | 0.4967 |
| repair_once | 80.00% | 793.6 | 0.7921 |
| repair_twice | 80.00% | 1279.8 | 0.7872 |
| retry_once | 30.00% | 678.6 | 0.2932 |
| retry_twice | 60.00% | 1011.6 | 0.5899 |

The controller pays for every executed input/output token, including a decision to stop. Baselines use only the same visited solver cache. No fitted controller or POMDP is loaded. Invalid action outputs submit the current answer and remain counted.

Intervals are paired question bootstrap estimates conditional on one fixed cache and controller run. Development results are exploratory; comparisons against five baselines are not adjusted for multiple testing. This report does not establish a confirmatory win.

Only actual controller request latency is measured. Cached solver batch times are not serial serving latency.
Token-price estimates are not invoices. Compute/idle/startup charges require the provider ledger. Budgets are per run, not account-wide.
Existing test baseline outcomes have been inspected; use fresh data if those outcomes informed redesign.
