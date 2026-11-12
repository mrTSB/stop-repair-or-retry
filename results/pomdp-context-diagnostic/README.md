# Context-conditioned two-state diagnostic

Read [report.md](report.md). This separate train/dev-only candidate conditions dynamics on observed action history; it does not add hidden states, domain-specific models, or answer-change observations. The original milestone and frozen lambda0.248 test result are unchanged.

At lambda0.248, it gets 33/60 dev correct at 760.85 mean tokens, matching repair-once accuracy with a small token saving. The report explains why the two early stops and unseen confidence categories make that saving fragile. No test result is claimed.

Reproduce with a fresh directory (offline):

```sh
python3 -B experiments/pomdp-controller/context_diagnostic.py --output runs/context-diagnostic-v1
```

The separate `verify_context.py` script independently recounts training estimates and reconstructs results. All branches stay in their original train/dev split. No paid inference or API credentials are required. `artifact-sha256.json` hashes the archived evidence. The script snapshot preserves the exact collected version.
