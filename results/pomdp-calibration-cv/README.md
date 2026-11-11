# Calibration cross-validation diagnostic

Read [report.md](report.md). This is a train-only five-fold model comparison followed by a single selected dev evaluation. It finds no new accuracy improvement; the selected context model at0.01 matches repair twice. No answer-change feature or test evaluation was added.

Offline reproduction with a fresh directory:

```sh
python3 -B experiments/pomdp-controller/calibration_cv.py --output runs/calibration-cv-v1
```

`verify_calibration_cv.py` reconstructs the fold models, held-out paths, summaries and selection without inference. Each question and all its branches stay in one fold. Protocol/hash records and `artifact-sha256.json` preserve provenance. The initial three variants and two penalties were declared before their OOF results; dev did not select the candidate. Cross-validation and dev are not untouched evidence after iterative project development.
