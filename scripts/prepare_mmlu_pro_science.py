#!/usr/bin/env python3
"""Prepare disjoint science-only controller splits from official MMLU-Pro items."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


SCIENCE_CATEGORIES = ("biology", "chemistry", "physics")
LETTERS = "ABCDEFGHIJ"


def _read(path: Path) -> tuple[dict[str, list[dict]], str]:
    rows = pq.read_table(path).to_pylist()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = {category: [] for category in SCIENCE_CATEGORIES}
    for row in rows:
        category = row["category"].strip().lower()
        if category not in groups:
            continue
        options = list(row["options"] or [])
        answer = str(row["answer"]).strip().upper()
        if len(options) < 2 or len(options) > len(LETTERS) or answer not in LETTERS[:len(options)]:
            continue
        choices = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(options))
        question = (
            f"Question: {row['question']}\n\nOptions:\n{choices}\n\n"
            "Choose the best answer. In the JSON answer field, return only one option letter."
        )
        groups[category].append({
            "id": f"mmlu-pro-{category}-{int(row['question_id']):05d}",
            "question": question,
            "answer": answer,
        })
    return groups, digest


def _stratified_take(groups: dict[str, list[dict]], count: int, rng: random.Random) -> list[dict]:
    categories = list(SCIENCE_CATEGORIES)
    rng.shuffle(categories)
    selected = []
    while len(selected) < count:
        progressed = False
        for category in categories:
            if len(selected) == count:
                break
            if groups[category]:
                selected.append(groups[category].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"Requested {count} from only {len(selected)} remaining science questions")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(".cache/mmlu-pro/data/test-00000-of-00001.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/mmlu-pro/prepared"))
    parser.add_argument("--pilot-size", type=int, default=25)
    parser.add_argument("--train-size", type=int, default=150)
    parser.add_argument("--dev-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()

    groups, digest = _read(args.input)
    eligible_counts = {category: len(rows) for category, rows in groups.items()}
    id_to_category = {row["id"]: category for category, rows in groups.items() for row in rows}
    rng = random.Random(args.seed)
    for rows in groups.values():
        rng.shuffle(rows)
    prepared = {
        "pilot": _stratified_take(groups, args.pilot_size, rng),
        "train": _stratified_take(groups, args.train_size, rng),
        "dev": _stratified_take(groups, args.dev_size, rng),
        "test": _stratified_take(groups, args.test_size, rng),
    }
    id_sets = [set(row["id"] for row in rows) for rows in prepared.values()]
    if any(id_sets[i] & id_sets[j] for i in range(len(id_sets)) for j in range(i + 1, len(id_sets))):
        raise ValueError("Prepared MMLU-Pro science splits overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in prepared.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": "MMLU-Pro",
        "source": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "paper": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/ad236edc564f3e3156e1b2feafb99a24-Abstract-Datasets_and_Benchmarks_Track.html",
        "license": "MIT",
        "source_file_sha256": digest,
        "source_split": "official test; disjoint question IDs partitioned for controller fit, tuning, and held-out evaluation",
        "categories": list(SCIENCE_CATEGORIES),
        "question_filter": "biology, chemistry, physics only; cot_content excluded from model input",
        "seed": args.seed,
        "counts": {split: len(rows) for split, rows in prepared.items()},
        "category_counts": {},
    }
    for split, rows in prepared.items():
        manifest["category_counts"][split] = dict(Counter(id_to_category[row["id"]] for row in rows))
    manifest["eligible_counts"] = eligible_counts
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "category_counts": manifest["category_counts"], "eligible": manifest["eligible_counts"]}, indent=2))


if __name__ == "__main__":
    main()
