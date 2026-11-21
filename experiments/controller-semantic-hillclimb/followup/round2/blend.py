"""Materialize completed fitting banks without changing either original study."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import ordinary
from semantic_data import read, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    complete = json.loads((args.new / 'train-ordinary-complete.json').read_bytes())
    for key, file in (('base_bank_sha256', 'train-observations.jsonl'), ('late_bank_sha256', 'train-late-observations.jsonl')):
        if complete[key] != sha(args.new / file):
            raise ValueError('Changed or incomplete new paired bank')
    if args.output.exists():
        raise ValueError('Do not overwrite a fitting view')
    old_rows, _, _, _ = ordinary(args.old, 'train')
    new_rows, _, _, _ = ordinary(args.new, 'train')
    if {r['id'] for r in old_rows} & {r['id'] for r in new_rows}:
        raise ValueError('Old and new fitting sets overlap')
    args.output.mkdir(parents=True)
    (args.output / 'splits').mkdir()
    (args.output / 'provenance').mkdir()
    selected = json.loads((args.new / 'splits/selection.json').read_bytes())
    selected['original_reservation_sha256'] = sha(args.new / 'splits/selection.json')
    selected['legacy_fitting_banks'] = {n: sha(args.old / n) for n in ('train-observations.jsonl', 'train-late-observations.jsonl')}
    selected['fitting_view'] = 'Original1472plus new512training questions; only the new256development and new1024test questions.'
    for split in ('train', 'dev', 'test'):
        path = args.output / 'splits' / (split + '-questions.jsonl')
        rows = old_rows + new_rows if split == 'train' else read(args.new / 'splits' / path.name)
        write(path, rows)
        selected['splits'][split] = {'n': len(rows), 'sha256': sha(path)}
        for suffix in ('observations', 'late-observations'):
            target = args.output / (split + '-' + suffix + '.jsonl')
            parts = ([args.old / target.name, args.new / target.name] if split == 'train' else [args.new / target.name])
            observations = [r for p in parts for r in read(p)]
            keys = {(r['id'], r['kind'], r['path']) for r in observations}
            if len(keys) != len(observations):
                raise ValueError('Duplicate paired records')
            write(target, observations)
    selected['strata'] = {'new-source': [r['id'] for r in read(args.output / 'splits/test-questions.jsonl')]}
    (args.output / 'splits/selection.json').write_text(json.dumps(selected, indent=2) + '\n')
    shutil.copy2(args.new / 'provenance/historical-exclusions.json', args.output / 'provenance/historical-exclusions.json')
    for split in ('train',):
        rows, _, _, _ = ordinary(args.output, split)
        assert len(rows) == 1984
    print(json.dumps({'train_questions': 1984, 'dev_questions': 256, 'fresh_test_questions': 1024,
                      'original_sources_unchanged': True, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'new', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
