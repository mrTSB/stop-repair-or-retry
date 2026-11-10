# Benchmark baseline index

- [Cached math/science dataset and five fixed baseline characterization](../cached-tree-dataset/report.md)

- [Large GSM8K baseline](../gsm8k-large/report.md)
- [Interactive ALFWorld baseline and ScienceWorld feasibility gate](../interactive/report.md)
- [MMLU-Pro science baseline](../mmlu-pro-science/report.md)
- [ALFWorld Qwen3.5 9B fixed-policy follow-up](../alfworld-qwen35-9b/report.md)

The MMLU-Pro study evaluates biology, chemistry, and physics with local Qwen3.5 9B. On 300 held-out study items, stop scored 56.3%, repair 65.0% (+8.7 pp, paired 95% CI +5.0 to +12.3), and a dev-tuned confidence threshold scored 66.3% (+10.0 pp vs stop, +6.7 to +13.3). Against fixed repair, threshold was +1.3 pp (+0.3 to +2.7) while using 29.2% fewer tokens. Its score is specific to disjoint partitions sampled from the public test source and is not an official MMLU-Pro leaderboard result.

The original 134-task ALFWorld 3B run had low absolute completion (3% stop, 6% repair). A disjoint 40-task Qwen3.5 9B follow-up reached 2.5% stop and 40% for both repair and retry, with paired gains of 37.5 pp (95% bootstrap CI +22.5 to +52.5). This is a promising fixed-policy result, but the sample was selected after a separate pilot and has no controller train/dev evaluation; treat it as exploratory. An earlier 3B ScienceWorld gate failed; revised action-menu pruning still needs a fresh ScienceWorld evaluation.
