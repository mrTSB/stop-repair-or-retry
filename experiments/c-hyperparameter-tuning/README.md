# C hyperparameter-only experiments

These scripts extend the frozen `abc-runtime-2026-11-25` cached runtime without editing its code. Copy them into `<runtime>/runs/extensions/` and invoke through the original `scripts/reproduction/run.py` adapter, with a fresh Python 3.12.11 environment matching the release's hash-locked requirements and `PYTHONDONTWRITEBYTECODE=1`.

- `cost_constrained.py OUTPUT`: original 1,800-setting grid, retaining original settings, exact subject-composition selection, fold sensitivity and full outcomes.
- `audit.py OUTPUT`: verify artifact hashes, reload the saved model, prove that only existing settings/prior weights differ, and directly/blindly replay all questions with independent correctness and billing checks.

See [results and reproduction instructions](../../results/c-hyperparameter-tuning/README.md). These are exploratory development searches. No test reads, model forwards, refitting or architectural changes are permitted by these scripts.
