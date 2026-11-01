"""Prepare reproducible, disjoint subsets of the official GSM8K parquet files."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from .core import read_jsonl, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-parquet', required=True)
    p.add_argument('--test-parquet', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-size', type=int, default=500)
    p.add_argument('--dev-size', type=int, default=200)
    p.add_argument('--test-size', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude-test-jsonl', action='append', default=[], help='Exclude previously inspected questions from the new test set')
    args = p.parse_args()
    import pyarrow.parquet as pq
    original_train = pq.read_table(args.train_parquet, columns=['question', 'answer']).to_pylist()
    original_test = pq.read_table(args.test_parquet, columns=['question', 'answer']).to_pylist()
    excluded = {row['question'] for file in args.exclude_test_jsonl for row in read_jsonl(file)}
    available_test = [i for i, row in enumerate(original_test) if row['question'] not in excluded]
    if min(args.train_size, args.dev_size, args.test_size) <= 0 or args.train_size + args.dev_size > len(original_train) or args.test_size > len(available_test):
        p.error('Invalid requested split sizes')
    def convert(original, indices, split):
        rows = []
        for i in indices:
            sample = original[i]
            if '####' not in sample['answer']:
                raise ValueError(f'Missing GSM8K final-answer marker at {split} row {i}')
            answer = sample['answer'].rsplit('####', 1)[1].strip().replace(',', '')
            rows.append({'id': f'gsm8k-{split}-{i:05d}', 'question': sample['question'], 'answer': answer})
        return rows
    rng = random.Random(args.seed)
    train_indices = rng.sample(range(len(original_train)), args.train_size + args.dev_size)
    test_indices = rng.sample(available_test, args.test_size)
    splits = {'train': convert(original_train, train_indices[:args.train_size], 'train'), 'dev': convert(original_train, train_indices[args.train_size:], 'train'), 'test': convert(original_test, test_indices, 'test')}
    out = Path(args.output)
    for split, rows in splits.items(): write_jsonl(out / (split + '.jsonl'), rows)
    manifest = {'dataset': 'openai/gsm8k main', 'source': 'https://huggingface.co/datasets/openai/gsm8k', 'seed': args.seed, 'counts': {k: len(v) for k, v in splits.items()}, 'excluded_test_questions': len(excluded), 'exclusion_sha256': {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.exclude_test_jsonl}, 'source_sha256': {split: hashlib.sha256(Path(path).read_bytes()).hexdigest() for split, path in [('train', args.train_parquet), ('test', args.test_parquet)]}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
