"""Paired comparisons of accuracy, token use, and utility against a baseline."""
import argparse
import json
import statistics
from pathlib import Path
from .core import read_jsonl
from .cli import bootstrap_interval


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_dir')
    p.add_argument('--baseline', default='repair')
    p.add_argument('--penalty', type=float, default=.01)
    args = p.parse_args()
    root = Path(args.result_dir)
    summary = json.loads((root / 'summary.json').read_text())
    if summary['penalty'] != args.penalty: p.error('Penalty does not match the evaluated results')
    groups = {}
    for r in read_jsonl(root / 'per_question.jsonl'):
        group = groups.setdefault(r['policy'], {})
        if r['id'] in group: raise ValueError('Duplicate policy/question result')
        group[r['id']] = r
    base = groups[args.baseline]
    results = []
    for policy, rows in groups.items():
        if rows.keys() != base.keys(): raise ValueError(f'Question set differs for {policy}')
        accuracy = [rows[i]['correct'] - base[i]['correct'] for i in base]
        tokens = [rows[i]['tokens'] - base[i]['tokens'] for i in base]
        utility = [a - args.penalty * t / 1000 for a,t in zip(accuracy, tokens)]
        results.append({'policy':policy, 'baseline':args.baseline, 'n':len(base), 'accuracy_difference':statistics.mean(accuracy), 'accuracy_difference_ci_95':bootstrap_interval(accuracy), 'token_difference':statistics.mean(tokens), 'token_difference_ci_95':bootstrap_interval(tokens), 'utility_difference':statistics.mean(utility), 'utility_difference_ci_95':bootstrap_interval(utility)})
    dest = root / f'paired-vs-{args.baseline}.json'
    dest.write_text(json.dumps(results, indent=2) + '\n')
    print(dest)

if __name__ == '__main__': main()
