"""Create reference-free observations for every completed root-medium training pair."""

import argparse
import hashlib
import json
from pathlib import Path

from banks import ordinary, committed
from root_bank import view
from semantic_data import write
from difficulty_pomdp import FIXED, visible
from scoring import correct
from examples import text_pair
from evidence import Retriever, left_text
from semantic_pomdp import review_node


def run(args):
    contract = json.loads((args.root / 'contract.json').read_bytes())
    observations = committed(args.root / 'train.jsonl')
    expected = {(r['id'], kind, path) for r in contract['selection']
                for kind, paths in [('judge', ['']), ('solver', ['guided', 'guided/repair', 'guided/retry'])]
                for path in paths}
    if len(observations) != len(expected) or {(r['id'], r['kind'], r['path']) for r in observations} != expected:
        raise ValueError('Complete all192root training pairs before preparing observations')
    rows, trees, _, _ = ordinary(args.data, 'train')
    local, reviews = view(rows, trees, observations, validate=True)
    by_id = {r['id']: r for r in rows}
    retriever = Retriever()
    inputs, metadata = [], []
    for item in contract['selection']:
        qid = item['id']
        row = by_id[qid]
        safe = {**row, 'answer': ''}
        background, _ = retriever.context(safe)
        hard = int(sum(correct(trees[qid][p], row['answer']) for p in FIXED) / len(FIXED) < .6)
        fold = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 3
        for kind, path in [('original-review', ''), ('solver', 'guided'),
                           ('solver', 'guided/repair'), ('solver', 'guided/retry')]:
            node = review_node(reviews[qid, path]) if kind == 'original-review' else local[qid][path]
            current = {**local[qid], path: node}
            _, right = text_pair(safe, node)
            fields = {'left': left_text(safe, background), 'right': right, 'numeric': visible(safe, path, current)}
            inputs.append({'id': qid, 'kind': kind, 'path': path, 'fold': fold, **fields,
                           'input_sha256': hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()})
            metadata.append({'id': qid, 'kind': kind, 'path': path, 'fold': fold, **fields,
                             'label': 2 * hard + int(correct(node, row['answer'])), 'valid': bool(node['answer']),
                             'train': kind == 'solver' and path == 'guided',
                             'plain': [-1.] * 8, 'guided': [-1.] * 8})
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'inputs.jsonl', inputs)
    write(args.output / 'metadata.jsonl', metadata)
    receipt = {'questions': 192, 'nodes': len(inputs), 'fresh_data_used': False,
               'inputs_sha256': hashlib.sha256((args.output / 'inputs.jsonl').read_bytes()).hexdigest(),
               'metadata_sha256': hashlib.sha256((args.output / 'metadata.jsonl').read_bytes()).hexdigest(),
               'corpus_sha256': retriever.sha256, 'rule': 'Reference labels remain local and outside neural inputs.'}
    (args.output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
