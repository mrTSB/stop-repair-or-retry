# Two-state POMDP milestone archive

Read [report.md](report.md) for results and limitations. [DESIGN.md](../../experiments/pomdp-controller/DESIGN.md) distinguishes the user's decisions, assistant defaults, and final-project extensions; `DESIGN.frozen.md` preserves the exact design used for this run.

The primary POMDP policy is repair twice on all 60 dev and 400 test questions. It matches that fixed baseline; do not claim an adaptive-policy improvement. No four-state model or one-step ablation was evaluated.

All computation is offline on the existing solver cache. No API credentials, Modal resources, or new inference spending are needed. The independent audit can be rerun from the frozen repository revision:

```sh
python3 experiments/pomdp-controller/verify.py --run results/pomdp-two-state
```

The `dev/` and `test/` folders each contain one path file and summary per predeclared penalty, plus a status record. Do not modify dev files after freeze. `model.json` is fitted from training only. `manifest.json` and `frozen.json` bind model/source/design/data and dev evidence. `artifact-sha256.json` hashes every archived file except itself. CPU timings are measurements and are excluded from deterministic path equality; all decisions, beliefs, costs, scores, and summary arithmetic are reproducible.

Changes to the future model, design, or dependencies require a new experiment version. Preserve this archive and use its matching Git revision for verification. The main workflow instructions are in [experiments/pomdp-controller/README.md](../../experiments/pomdp-controller/README.md).
