"""Compare two generation seeds on the repeat sample, paired by question."""
import argparse
import json
import statistics
from pathlib import Path
from .core import canonical_answer, read_jsonl
from .cli import bootstrap_interval


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('primary_result_dir')
    p.add_argument('repeat_result_dir')
    args = p.parse_args()
    def collect(path):
        groups = {}
        for row in read_jsonl(Path(path) / 'per_question.jsonl'):
            group = groups.setdefault(row['policy'], {})
            if row['id'] in group: raise ValueError('Duplicate policy/question result')
            group[row['id']] = row
        return groups
    first, second = collect(args.primary_result_dir), collect(args.repeat_result_dir)
    if first.keys() != second.keys(): raise ValueError('Policy sets do not match')
    results = []
    for policy, repeat in second.items():
        initial = first[policy]
        if not repeat.keys() <= initial.keys(): raise ValueError('Repeat questions not present in primary run')
        diffs = [repeat[i]['correct'] - initial[i]['correct'] for i in repeat]
        changes = [canonical_answer(repeat[i]['answer']) != canonical_answer(initial[i]['answer']) for i in repeat]
        results.append({'policy':policy, 'n':len(repeat), 'primary_accuracy_on_repeat_questions':statistics.mean(initial[i]['correct'] for i in repeat), 'repeat_accuracy':statistics.mean(repeat[i]['correct'] for i in repeat), 'accuracy_difference_repeat_minus_primary':statistics.mean(diffs), 'accuracy_difference_ci_95':bootstrap_interval(diffs), 'answer_change_rate':statistics.mean(changes)})
    out=Path(args.repeat_result_dir)/'paired-seed-comparison.json'
    out.write_text(json.dumps(results, indent=2)+'\n')
    print(out)

if __name__ == '__main__': main()
