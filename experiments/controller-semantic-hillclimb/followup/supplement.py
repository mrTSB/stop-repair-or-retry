"""Prepare blinded fold-specific neural inputs for the new paired training observations."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from semantic_data import read, write
from examples import text_pair
from evidence import Retriever, left_text
from semantic_pomdp import review_node
from difficulty_pomdp import FIXED, visible
from scoring import correct
from banks import ordinary, variant_observations, medium_view


def run(args):
    contract = json.loads((args.medium / 'contract.json').read_bytes())
    observations = variant_observations(args.medium, args.variants, args.variant)
    known = {(r['id'], r['kind'], r['path']): r['response'] for r in observations}
    expected = {(r['id'], kind, r['path'] if kind == 'judge' else r['path'] + '/reviewed')
                for r in contract['selection'] for kind in ('judge', 'solver')}
    if set(known) != expected:
        raise ValueError('Finish the full prespecified paired training collection first')
    rows, trees, original, _ = ordinary(args.data, 'train')
    local, reviews = medium_view(rows, trees, original, observations, validate=True)
    by_id = {r['id']: r for r in rows}
    retriever = Retriever()
    inputs = []
    metadata = []
    for item in contract['selection']:
        qid, parent = item['id'], item['path']
        row = by_id[qid]
        safe = {**row, 'answer': ''}
        context, _ = retriever.context(safe)
        hard = int(sum(correct(trees[qid][p], row['answer']) for p in FIXED) / len(FIXED) < .6)
        fold = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 3
        for kind, path in (('late-review', parent), ('solver', parent + '/reviewed')):
            node = review_node(reviews[qid, path]) if kind == 'late-review' else local[qid][path]
            current = {**local[qid], path: node}
            _, right = text_pair(safe, node)
            fields = {'left': left_text(safe, context), 'right': right,
                      'numeric': visible(safe, path, current)}
            inputs.append({'id': qid, 'kind': kind, 'path': path, 'fold': fold, **fields,
                           'input_sha256': hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()})
            metadata.append({'id': qid, 'kind': kind, 'path': path, 'fold': fold, 'train': False,
                             **fields, 'valid': bool(node['answer']),
                             'label': 2 * hard + int(correct(node, row['answer'])),
                             'plain': [-1.] * 8, 'guided': [-1.] * 8})
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'inputs.jsonl', inputs)
    write(args.output / 'metadata.jsonl', metadata)
    manifest = {'questions': len(contract['selection']), 'nodes': len(inputs), 'variant': args.variant,
                'folds': dict((str(k), sum(r['fold'] == k for r in inputs)) for k in (0, 1, 2)),
                'corpus_sha256': retriever.sha256,
                'inputs_sha256': hashlib.sha256((args.output / 'inputs.jsonl').read_bytes()).hexdigest(),
                'metadata_sha256': hashlib.sha256((args.output / 'metadata.jsonl').read_bytes()).hexdigest(),
                'rule': 'Metadata labels stay local; GPU inputs contain only visible text/features and question-grouped fold IDs.',
                'fresh_data_used': False}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--medium', type=Path, required=True)
    parser.add_argument('--variants', type=Path)
    parser.add_argument('--variant', choices=('compressed','medium-short','medium-full'), default='medium-short')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
