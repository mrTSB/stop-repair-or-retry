# Four-state latent-model diagnostic

Read [report.md](report.md). Four states did not improve held-out training accuracy over the matched two-state control. Only the selected two-state candidate was evaluated on dev; no test evaluation or paid inference was performed.

Reproduce offline in a fresh directory:

```sh
python3 -B experiments/pomdp-controller/latent_states.py --output runs/pomdp-latent-state-cv
```

See protocol.json for the fixed state counts, penalties, fitting iterations, selection rule and source hashes. Fold models and paths preserve question-level separation. The original milestone and frozen lambda-0.248 results remain unchanged.

Independent verification passed: exact small-tree inference tests, held-out label invariance, matched two-state parameters, all ten fold-model refits, twenty held-out replays, and the selected dev replay. Run `python3 -B experiments/pomdp-controller/verify_latent_states.py` to repeat the archive audit and `python3 -B -m unittest discover -s tests -p test_latent_states.py` for the three mathematical/policy-input tests. See [validation.json](validation.json).
