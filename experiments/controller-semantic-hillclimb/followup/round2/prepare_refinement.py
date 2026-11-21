"""Build local, question-grouped observer training inputs from complete new pairs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary
from examples import examples
from evidence import Retriever, left_text
from semantic_data import read, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Do not overwrite training inputs')
    completed = json.loads((args.data / 'train-ordinary-complete.json').read_bytes())
    if completed['base_bank_sha256'] != sha(args.data / 'train-observations.jsonl') or completed['late_bank_sha256'] != sha(args.data / 'train-late-observations.jsonl'):
        raise ValueError('Finish and verify all new paired training data first')
    rows, trees, _, _ = ordinary(args.data, 'train')
    if len(rows) != 512 or any(len(trees[r['id']]) != 13 for r in rows):
        raise ValueError('Use all512new paired questions and every solver branch')
    old = read(args.legacy)
    if {r['id'] for r in rows} & {r['id'] for r in old}:
        raise ValueError('New and old training questions overlap')
    retriever = Retriever()
    by_id = {r['id']: r for r in rows}
    contexts = {r['id']: retriever.context({**r, 'answer': ''})[0] for r in rows}
    new = examples(args.data, 'train')
    for item in new:
        item['left'] = left_text(by_id[item['id']], contexts[item['id']])
    all_rows = old + new
    for item in all_rows:
        expected = int(hashlib.sha256(item['id'].encode()).hexdigest()[:8], 16) % 3
        if item['fold'] != expected:
            raise ValueError('Preserve the original excluded-question folds')
    args.output.mkdir(parents=True)
    path = args.output / 'train.jsonl'
    write(path, all_rows)
    manifest = {'source_sha256': sha(__file__), 'legacy_inputs_sha256': sha(args.legacy),
                'new_base_bank_sha256': completed['base_bank_sha256'],
                'new_late_bank_sha256': completed['late_bank_sha256'],
                'corpus_sha256': retriever.sha256, 'input_sha256': sha(path),
                'old_questions': len({r['id'] for r in old}), 'new_questions': len(rows),
                'questions': len({r['id'] for r in all_rows}), 'nodes': len(all_rows),
                'new_nodes': len(new), 'grouped_folds': 3,
                'development_or_test_used': False, 'observation_text_uploaded': False}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'legacy', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
