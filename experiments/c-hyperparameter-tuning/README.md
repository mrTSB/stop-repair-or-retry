# C hyperparameter-only experiments

These scripts extend the frozen `abc-runtime-2026-11-25` cached runtime without editing its code. Copy them into `<runtime>/runs/extensions/` and invoke through the original `scripts/reproduction/run.py` adapter, with a fresh Python 3.12.11 environment matching the release's hash-locked requirements and `PYTHONDONTWRITEBYTECODE=1`.

- `cost_constrained.py OUTPUT`: original 1,800-setting grid, retaining original settings, exact subject-composition selection, fold sensitivity and full outcomes.
- `accuracy_first.py OUTPUT`: 864 penalty/prior pairs with label-independent gate intervals, accuracy-first selection without a compute ceiling, constrained alternatives and fold sensitivity. See [the protocol](ACCURACY_PROTOCOL.md).
- `analyze_accuracy.py OUTPUT`: independent frontier reconstruction and PNG/PDF figures; requires Matplotlib for rendering only.
- `document_accuracy.py OUTPUT`: generate the accuracy report after the audit and analysis.
- `audit.py OUTPUT`: verify artifact hashes, reload the saved model, prove that only existing settings/prior weights differ, and directly/blindly replay all questions with independent correctness and billing checks.

See [results and reproduction instructions](../../results/c-hyperparameter-tuning/README.md). These are exploratory development searches. No test reads, model forwards, refitting or architectural changes are permitted by these scripts.

For published artifacts, decompress only filenames listed by each result directory’s `packaging.json` into a new copy before auditing or analyzing. Preserve the original bytes to satisfy `complete.json`. Plots and reports are post-completion artifacts, with a separate publication manifest.
