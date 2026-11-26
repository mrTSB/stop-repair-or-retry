# Complete A/B/C reproduction handoff

Download the full runtime from the [2026-11-25 release](https://github.com/mrTSB/stop-repair-or-retry/releases/tag/abc-runtime-2026-11-25). The release contains one zstd tar archive split into numbered parts, `SHA256SUMS`, `archive.json`, a file-by-file SHA-256 manifest, and the bundle inventory. Download every part. The compressed archive is 69,249,674,958 bytes across 35 parts; the payload occupies about 101 GB before filesystem hard-link savings.

This handoff includes the reserved test questions **and reference labels**, all local solver/reviewer response banks, intermediate and selected controllers, neural weights and tokenizers, feature arrays and prediction caches, raw dataset sources, retrieval corpora and indexes, configurations, registrations, completion receipts, source, and dependencies. Every file in the local `runs/`, `.cache/`, and historical `.venv/` is included, as are every installed Ollama model's manifests/blobs and the original GSM8K source caches outside the checkout. Identical file contents share tar hard links; each original path is retained. Existing experiment artifacts and frozen source remain byte-identical.

The latest experiment is `experiments/controller-semantic-hillclimb/followup/round2`, through the audited Bayesian C checkpoint. Its 512 training, 256 development and 1,024 reserved test questions are disjoint by ID. Latest development results are A 153/256, B 179/256, C 177/256. Development has been repeatedly used for selection. The 1,024-question set has no collected responses or grades; its empty response journals are included. Exporting and checking its labels does not produce a new evaluation.

## Repository files and archive layout

`latest/` contains the latest split files with labels, compressed ordinary train/dev response banks, selected `models/A.pkl` and `models/C.pkl`, B's rules, and selected configuration records. The complete model and evidence lineage is in the archive; the small directory alone does not supply all observer assets needed by these pickles. The original tracked experiment code and older published results remain in their existing locations.

The extracted archive has one top-level `stop-repair-or-retry/` directory. `reproduction-bundle.json` records coverage, split counts, original source commit and automatic path mappings. `reproduction-files.json` lists the size and SHA-256 of every payload file. The same inventories and replay receipts are also published alongside this README and as release attachments.

| Location inside the extracted bundle | Contents |
| --- | --- |
| `runs/controller-semantic-followup/round2/data` | Latest splits, labels, ordinary paired train/dev responses and collection contracts |
| `runs/controller-semantic-followup/round2` | All later paired banks, raw requests/receipts, candidates, intermediate/recovery checkpoints, observer models/features, registrations and analyses |
| `runs/controller-semantic-followup` | Earlier 912-question studies, observer folds and preceding model/data lineages |
| `runs/controller-semantic-hillclimb`, `runs/controller-hillclimb` | Preceding semantic, fresh-800, guided-repair and original A/B/C runs |
| `.cache` | Original dataset downloads, model assets and retrieval source documents |
| `external/ollama/models` | All installed inference model blobs, configuration, license layers and manifests |
| `external/huggingface` | Original external GSM8K Arrow and downloaded dataset source files |
| `.venv` | Historical environment files, including dependency files named in frozen hashes; install the portable environment below for execution |

## Download and setup

Install `uv`, GitHub CLI (`gh`), and `zstd`. Python 3.12.11 and the Python dependencies are obtained by `uv`; the dependency lock includes hashes and platform markers for macOS and Linux. Downloading/installing dependencies needs internet. Cached replay requires no inference service or credentials.

For downloading without a GitHub account or GitHub CLI, download the release's `download.py` attachment and run `uv run --python 3.12.11 --no-project python download.py --destination abc-download`. It verifies every part and the combined archive. Then use the extraction/setup commands below, starting with `cat`.

```sh
mkdir abc-download
gh release download abc-runtime-2026-11-25 \
  --repo mrTSB/stop-repair-or-retry \
  --pattern 'abc-runtime.tar.zst.part-*' \
  --pattern SHA256SUMS --pattern archive.json --dir abc-download
(cd abc-download && shasum -a 256 -c SHA256SUMS)
cat abc-download/abc-runtime.tar.zst.part-* | zstd -d | tar -xf -
cd stop-repair-or-retry

uv run --python 3.12.11 --no-project python scripts/reproduction/verify.py
uv run --python 3.12.11 \
  --with-requirements scripts/reproduction/requirements.txt --no-project \
  python scripts/reproduction/run.py scripts/reproduction/replay_latest.py \
  --output runs/replay-latest
```

`run.py` redirects archived logical paths to the extracted bundle. It preserves the original strings used in registrations, serialized models and checksums. It also blocks filesystem operations that would accidentally use the producer's original checkout. No mount point, home-directory alias, or edits to frozen files are required. Use Python 3.12 for this adapter. Original directory strings in provenance are evidence, not paths the recipient must create.

The latest replay checks all eight methods and repeats A/B/C decisions with references blanked and only paid solver/reviewer observations. All used observer outputs still incur their original inference charge. It forbids new model calls and test reads. Its summary and per-question output are written to a new directory.

To check that the archived weights and tokenizer support new CPU feature extraction, use the same command prefix with `python scripts/reproduction/run.py scripts/reproduction/smoke_observer.py`. It runs one training input with its reference removed.

Independently audit the latest 36 new C candidates, their component bills and the 192-candidate subject selection:

```sh
mkdir -p runs/replay-audits
uv run --python 3.12.11 \
  --with-requirements scripts/reproduction/requirements.txt --no-project \
  python scripts/reproduction/run.py \
  experiments/controller-semantic-hillclimb/followup/round2/audit_bayesian_hybrid.py \
  --run runs/controller-semantic-followup/round2/development-bayesian-model-hybrid-v3 \
  --output runs/replay-audits/bayesian.json
```

The earlier fresh-800 study remains separately reproducible:

```sh
old=results/controller-hillclimb/pomdp-abc/hybrid-margin
mkdir -p runs/replay-800
cp "$old/frozen.json" runs/replay-800/
cp -R "$old/models" runs/replay-800/
uv run --python 3.12.11 \
  --with-requirements scripts/reproduction/requirements.txt --no-project \
  python scripts/reproduction/run.py \
  experiments/controller-hillclimb/pomdp-abc/hybrid-margin/fresh_study.py score \
  --data "$old/data" --output runs/replay-800
uv run --python 3.12.11 \
  --with-requirements scripts/reproduction/requirements.txt --no-project \
  python scripts/reproduction/run.py \
  experiments/controller-hillclimb/pomdp-abc/hybrid-margin/margin_audit.py \
  --data "$old/data" --output runs/replay-800
```

## Refit and extend

Every original training recipe, fold assignment, feature file and intermediate model remains under its original relative runtime path. Use new output directories. This example repeats the selected math forecaster's registered training fit using the archived excluded-question features and diagnostic. `extend.py` obtains input paths from the archived registration, registers the new output, computes its registration digest, and invokes the fitting phase:

```sh
uv run --python 3.12.11 \
  --with-requirements scripts/reproduction/requirements.txt --no-project \
  python scripts/reproduction/extend.py \
  --recipe runs/controller-semantic-followup/round2/math-specialized-model-v1/registration.json \
  --script experiments/controller-semantic-hillclimb/followup/round2/fit_math_specialization.py \
  --output runs/extensions/math-refit
```

For a new prepared training diagnostic, add `--set diagnostic=runs/your-new-diagnostic`. The original script enforces its source, whole-question folds and training-only evidence checks. A corresponding new development search can be repeated with `--phase replay`, the archived `development-math-specialization-v1/registration.json`, `development_math_specialization.py`, and a new output directory. Original scripts document their other training and collection parameters; invoke them through `run.py` when using archived inputs.

To serve the included generation weights on a compatible machine, use Ollama 0.40.16, the version recorded by the experiment worker:

```sh
OLLAMA_MODELS="$PWD/external/ollama/models" \
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

The selected solver is `qwen3.5:4b`, manifest SHA-256 `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`; the reviewer is `qwen3.5:35b`, `3460ffeede5453ead027dbd2f821b12ad0aa3de54630971993babdb2165221f7`. All other local installed tags are also included. Cloud collection code and pinned worker images are preserved; using Modal requires the recipient's own configured account. Cached replay and refitting above need no Modal account. Source licenses and attribution remain in their dataset/model manifests and included license files.

The complete export can be rebuilt with `scripts/reproduction/build.py` followed by `pack.py`. Keep the destination outside the source checkout. Neither export nor replay retunes a controller on the reserved test set.
