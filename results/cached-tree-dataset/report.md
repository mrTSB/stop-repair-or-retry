# Cached tree dataset and empirical characterization

We present a reproducible dataset of cached reasoning trajectories and characterize its accuracy and token-cost landscape using five fixed baselines on **400 test questions: 200 math and 200 science**. The baselines are descriptive reference points, not proposed methods. This report makes no controller contribution or controller-performance claim.

## Dataset and split

The contribution at this stage is the reproducible cached-tree dataset and its empirical characterization. Underlying questions come from GSM8K and MMLU-Pro; the new artifact is their generated response trees, provenance, scoring and controlled split, not newly authored benchmark questions.

| Split | Math | Science | Total | Purpose |
|---|---:|---:|---:|---|
| Training | 50 | 50 | 100 | Future controller fitting |
| Development | 30 | 30 | 60 | Future tuning and design choices |
| Test | 200 | 200 | 400 | Reported baseline evaluation |

Splits are disjoint by question ID, with frozen source hashes and documented duplicate checks. Every question and all seven of its cached solver nodes stay together in one split. These five fixed baselines require no training or development tuning. There are 560 complete trees and 3,920 solver responses in the contribution dataset. This bundle contains solver responses only. No fitted controllers, judge decisions or controller-result files are used in this analysis.

## Dataset construction and scope

The cache uses Qwen/Qwen2.5-3B-Instruct at pinned revision aa8e72537993ba99e69dfaafa59ed015b17504d1, fixed action prompts, temperature 0.7, top-p 0.9 and a 768-token solver output cap. Each question has an initial response and every repair/retry branch through two additional calls. Repair sees the question and immediate preceding response; retry solves the original question in a fresh context. All five baselines share these same cached solver outputs.

The measured landscape depends jointly on the sampled questions, frozen solver, prompts, decoding settings and one sampled tree per question. These findings are not properties of the questions alone, nor evidence about adaptive decision making. Math questions come from GSM8K; science questions cover MMLU-Pro biology, chemistry and physics. Exact-ID and normalized-stem duplicate checks and source hashes are documented in the dataset manifest; these checks do not establish absence of semantic duplicates or model pretraining exposure.

## Main characterization

Repair once raises observed accuracy from 36.25% to 52.75%; a second repair reaches 54.75%, with another 525 tokens per question on average. Fresh retries produce smaller aggregate gains on this cache. These descriptive differences establish an accuracy–cost reference for future work and do not introduce a new algorithm.

## All questions

| Baseline | Accuracy | 95% accuracy interval | Mean tokens | Mean utility |
|---|---:|---:|---:|---:|
| Submit initial | 36.25% | 31.8–41.0% | 340 | 0.3591 |
| Repair once | 52.75% | 47.8–57.8% | 833 | 0.5192 |
| Repair twice | 54.75% | 49.5–60.0% | 1358 | 0.5339 |
| Retry once | 38.25% | 33.2–43.0% | 680 | 0.3757 |
| Retry twice | 40.25% | 35.8–45.2% | 1021 | 0.3923 |

## Math questions

| Baseline | Accuracy | 95% accuracy interval | Mean tokens | Mean utility |
|---|---:|---:|---:|---:|
| Submit initial | 38.00% | 31.0–44.5% | 259 | 0.3774 |
| Repair once | 65.50% | 59.0–72.0% | 652 | 0.6485 |
| Repair twice | 68.50% | 62.0–75.0% | 1059 | 0.6744 |
| Retry once | 40.00% | 33.0–47.0% | 521 | 0.3948 |
| Retry twice | 47.00% | 40.0–54.0% | 782 | 0.4622 |

## Science questions

| Baseline | Accuracy | 95% accuracy interval | Mean tokens | Mean utility |
|---|---:|---:|---:|---:|
| Submit initial | 34.50% | 28.5–41.0% | 422 | 0.3408 |
| Repair once | 40.00% | 33.5–47.0% | 1014 | 0.3899 |
| Repair twice | 41.00% | 34.5–48.0% | 1657 | 0.3934 |
| Retry once | 36.50% | 30.0–43.0% | 838 | 0.3566 |
| Retry twice | 33.50% | 27.5–40.0% | 1260 | 0.3224 |

## Definitions and interpretation

Repair once checks the initial solution once and submits that response. Repair twice checks it twice and submits the latest response. Retry once/twice independently solve the original question and submit the latest response; they do not select the best answer. All baselines allow at most two extra solver calls.

Tokens include every visited solver input and output, including the initial response; there are no judge tokens. Utility is correctness minus 0.01 times tokens/1,000. Intervals use 2,000 question bootstrap resamples on one fixed cache. They do not cover repeated model-generation seeds. The source questions and cache remain unchanged; malformed responses stay in the denominator.

Controller development and evaluation are deferred. These five fixed strategies describe the existing dataset landscape and require no fitting or tuning. The test outcomes have already been examined; use train/development data for future choices and a fresh held-out set if test results inform redesign.

Per-question paths and costs: `per-question.jsonl`. Full precision summaries and source hashes: `results.json`. Reproduce locally with `python3 results/cached-tree-dataset/run_baselines.py`.
