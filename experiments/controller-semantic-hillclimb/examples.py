"""Prepare question-grouped semantic observations and masked paired-transition labels."""

import argparse
import hashlib
import json
from pathlib import Path

import semantic_data as data
from semantic_data import read, write
from difficulty_pomdp import FIXED, visible
from extended_acquire import SCIENCE_SUFFIX
from late_pairs import merged_records
from scoring import correct
import re


def text_pair(row, node):
    question, separator, options = row['question'].removesuffix(SCIENCE_SUFFIX).partition('\n\nOptions:\n')
    candidate = str(node['answer'])
    if separator and len(candidate) == 1 and candidate in 'ABCDEFGHIJ':
        found = re.search(r'^' + candidate + r'\.\s*(.*?)(?=^[A-J]\.\s|\Z)', options, re.M | re.S)
        if found:
            candidate += ': ' + found[1].strip()
    return question, 'Proposed answer: ' + candidate + '\nSolution:\n' + node['text']


def examples(source, split):
    rows, trees, judges = merged_records(source, split, False)
    # Original stage-one reviews remain useful supervised examples but are never
    # substituted for a paid late review during policy evaluation.
    _, _, original = data.records(source, split, False)
    result = []
    for row in rows:
        qid = row['id']
        tree = trees[qid]
        hard = int(sum(correct(tree[p], row['answer']) for p in FIXED) / len(FIXED) < .6)
        fold = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 3
        entries = [('solver', p, n) for p, n in tree.items()]
        entries += [('original-review', p, n) for (i, p), n in original.items() if i == qid]
        entries += [('late-review', p, n) for (i, p), n in judges.items() if i == qid and p]
        extra_review = min((p for i, p in original if i == qid and p),
                           key=lambda p: hashlib.sha256(f'{qid}:{p}'.encode()).digest(), default=None)
        for kind, path, receipt in entries:
            node = receipt
            current = tree
            if kind != 'solver':
                node = {**receipt, 'text':str(receipt['verdict'].get('solution', '')) + '\nFinal: ' + receipt['answer'],
                        'confidence':None}
                current = {**tree, path:node}
            state = 2 * hard + int(correct(node, row['answer']))
            left, right = text_pair(row, node)
            targets = {'plain':[-1.] * 8, 'guided':[-1.] * 8}
            if kind == 'solver' and path in ('', 'repair', 'retry', 'guided'):
                for action, name in enumerate(('repair', 'retry')):
                    child = f'{path}/{name}' if path else name
                    if child in tree:
                        targets['plain'][action * 4 + state] = float(correct(tree[child], row['answer']))
                    guided = ('guided' if action == 0 else 'retry') if not path else (
                        path + '/reviewed' if action == 0 else child)
                    if guided in tree and (qid, path) in judges:
                        targets['guided'][action * 4 + state] = float(correct(tree[guided], row['answer']))
            train = (kind == 'solver' and path in ('', 'repair', 'retry', 'guided')) or (
                kind == 'original-review' and path in ('', extra_review))
            result.append({'id':qid, 'kind':kind, 'path':path, 'fold':fold, 'train':train,
                           'left':left, 'right':right, 'numeric':visible(row, path, current),
                           'label':state, 'valid':bool(node['answer']), **targets})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for split in ('train', 'dev'):
        values = examples(args.data, split)
        write(args.output / (split + '.jsonl'), values)
        report[split] = {'questions':len({r['id'] for r in values}), 'all_nodes':len(values),
                         'training_nodes':sum(r['train'] for r in values),
                         'sha256':hashlib.sha256((args.output / (split + '.jsonl')).read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
