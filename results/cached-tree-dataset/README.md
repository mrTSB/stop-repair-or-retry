# Cached tree dataset and characterization

This bundle contributes **560 cached solver-response trees and their empirical characterization**, using exactly five fixed baselines. The baselines describe the landscape created by these questions, the frozen solver, its prompts and decoding settings; they are not proposed controller methods. No adaptive-controller results are included.

Read [the report](report.md) for accuracy, token costs, confidence intervals and math/science breakdowns.

| Split | Math | Science | Questions | Solver responses |
|---|---:|---:|---:|---:|
| Training | 50 | 50 | 100 | 700 |
| Development | 30 | 30 | 60 | 420 |
| Test | 200 | 200 | 400 | 2,800 |
| Total | 280 | 280 | 560 | 3,920 |

Every question and all seven of its response nodes remain in a single split. IDs are disjoint across splits. The five fixed baselines need neither fitting nor development tuning; reported results use only test questions. Training and development data are available for future work. Test outcomes have already been inspected: if they inform a redesign, use a fresh held-out set for confirmation.

## Reproduce the characterization

From the repository root, with Python 3.10 or newer and no additional dependencies:

```sh
python3 results/cached-tree-dataset/validate.py
python3 results/cached-tree-dataset/run_baselines.py
python3 results/cached-tree-dataset/validate.py
```

The first command checks the shipped files and dataset invariants. The second recomputes `report.md`, `results.json` and `per-question.jsonl` from the compressed solver trees, without reading or training any controller. The last checks that reproduction preserves the file manifest. No API, GPU, account credentials, model download or paid generation is needed.

| Fixed baseline | Test accuracy | Mean executed tokens |
|---|---:|---:|
| Submit initial answer | 36.25% | 340 |
| Repair once | 52.75% | 833 |
| Repair twice | 54.75% | 1,358 |
| Retry once | 38.25% | 680 |
| Retry twice | 40.25% | 1,021 |

Repair checks the immediate preceding response; retry independently solves the original question. Each baseline submits its latest response, with no best-answer selection. Tokens count the initial call and every executed solver input/output, but not unused cached branches. Confidence intervals condition on one sampled tree per question; this bundle does not measure variation across generation seeds. Answer correctness does not certify reasoning correctness.

## Files and schema

- `data/{train,dev,test}-questions.jsonl`: frozen questions, source IDs, gold answers, domain/subject and source provenance. Math rows also retain the upstream reference solution.
- `data/{train,dev,test}-trees.jsonl.gz`: gzip-compressed JSONL with one complete tree per question. Standard `gzip.open(path, 'rt', encoding='utf-8')` can read them without extraction.
- `provenance/selection-manifest.json`: sampling seed, upstream snapshot hashes, question-file hashes, historical exclusions, mechanical quality exclusions and duplicate-stem correction history.
- `provenance/generation.json`: exact solver/repair prompts, pinned model and software, decoding settings, batch seed rule and checkpoint collection procedure.
- `scoring.py`: frozen extraction and answer-scoring functions only. This does not modify the repository's existing scoring/controller implementation.
- `per-question.jsonl`: 2,000 question/baseline results, including visited paths, submitted answers, accuracy and executed token counts.
- `results.json`: full-precision summaries and bootstrap intervals for all, math and science questions.
- `FILE_MANIFEST.json`: SHA256 and size for each published bundle file, excluding itself.

Each tree has paths `""` (initial), `repair`, `retry`, `repair/repair`, `repair/retry`, `retry/repair`, and `retry/retry`. Nodes preserve exact raw response text, pre-template and formatted prompts, parsed answers/confidence, correctness labels, input/output tokens, finish reasons and original batch seed/position/size. Output counts include EOS when present; input counts exclude batch padding. Missing confidence is separate from answer validity. Invalid/truncated responses remain in the denominator.

Gold answers, reference solutions, correctness labels and unvisited branches are **evaluation data**, not permitted inference inputs. A future decision system must see only the question and visited history/observations. The bundle contains no cached judge outputs or fitted controller objects.

`batch_wall_seconds` measures shared batch generation time, not an independent request's latency. Do not sum it to claim interactive latency. Recovery retained all saved responses and generated only missing original batches with unchanged settings, seeds and parents; recovery was not based on correctness. Changing batching, hardware or software can change sampled outputs. The shipped cache, rather than fresh sampling, is the reproducible input for these results.

## Source selection and limits

Math questions come from official [GSM8K](https://github.com/openai/grade-school-math): this study's training/development partitions use upstream training data, and test uses upstream test data. Biology, chemistry and physics questions come from [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), repartitioned from its official test source. These are study-specific splits, not an official MMLU-Pro leaderboard score.

Selection used seed `2026111042`, recoverable historical/pilot exclusions and mechanical source screening. Historical GSM8K pilot IDs were recovered and excluded. Six candidate science test stems duplicated historical questions under different IDs; they were replaced deterministically within subject before test generation. Two known historical duplicate stems remain in training and are documented in the selection manifest; none of those exact historical stem matches remain in current development/test. No generated outcome determined question inclusion. Mechanical checks do not establish semantic uniqueness, expert answer-key validity or freedom from pretraining contamination.

The exact frozen question files and source snapshot hashes identify the release. Full upstream corpora and historical runs are not duplicated here. The corpus is balanced equally between math and science; its aggregate score is specific to that mixture. There is one model and one sampled tree per question. This is a dataset characterization, not evidence of a new controller or general superiority of any reasoning strategy.

## Attribution

Underlying benchmark questions are not newly authored by this project. This subset formats science questions with option letters and a JSON answer instruction, trims option whitespace, and adds generated solver trajectories and evaluation labels; raw model responses are preserved without rewriting. GSM8K is released under the [upstream MIT notice](licenses/GSM8K-MIT.txt). MMLU-Pro's [dataset card](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) declares MIT for the dataset; the associated GitHub source repository separately carries [Apache 2.0](licenses/MMLU-Pro-source-Apache-2.0.txt), retained here with its actual designation. These two upstream declarations are recorded separately rather than treating the source-repository license as the dataset's MIT notice. Model outputs were sampled from [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), at the revision recorded in generation provenance; no model weights are distributed. The scorer derives from this repository's baseline commit `490762fc4621e543ee413ee9e433254744ed38b1`, with this dataset's answer-validity handling preserved. Upstream notices remain applicable to their respective material.
