Exploratory development-only budget/information ablation. Not a fresh result.

| Method | Accuracy (%) | Compute proxy |
|---|---:|---:|
| Stop | 55.000 | 326.9 |
| Repair once | 67.500 | 864.8 |
| Repair twice | 74.583 | 1498.6 |
| Retry once | 66.250 | 772.9 |
| Retry twice | 69.583 | 1216.5 |
| A | 76.250 | 1109.9 |
| B | 82.500 | 1503.1 |
| C | 83.333 | 1347.5 |
| B action-only, 1 continuation(s) | 67.917 | 1131.2 |
| B action-only, 2 continuation(s) | 70.000 | 1497.6 |

One-continuation fixed baseline panel: Stop, Repair once, Retry once.

A/C and the full-strength references retain their frozen policies. Restricted B follows only the LLM action and uses plain compact-solver continuations. All original review tokens are charged.

The restricted conditions are explicit resource/information ablations. They do not replace the primary 800-question table or establish equal-budget superiority.

All requested criteria passed in any restricted scenario: **False**.

Full criteria and audit receipts: `summary.json`.
