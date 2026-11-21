"""Preregister a joint, still-blind panel and retain its source-specific strata."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from semantic_data import read, write, previous


def run(args):
    if (args.original.parent / 'confirmation-192/test/summary.json').exists():
        raise ValueError('Earlier test grades already opened; do not call the joint panel fresh')
    first = read(args.original / 'splits/test-questions.jsonl')
    second = read(args.transfer / 'splits/test-questions.jsonl')
    rows = first + second
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({previous.stem(r['question']) for r in rows}) == len(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('splits', 'provenance'):
        (args.output / name).mkdir()
    for split in ('train', 'dev'):
        shutil.copy2(args.original / 'splits' / f'{split}-questions.jsonl',
                     args.output / 'splits' / f'{split}-questions.jsonl')
        for suffix in ('observations', 'late-observations'):
            shutil.copy2(args.original / f'{split}-{suffix}.jsonl', args.output / f'{split}-{suffix}.jsonl')
    write(args.output / 'splits/test-questions.jsonl', rows)
    shutil.copy2(args.original / 'provenance/historical-exclusions.json',
                 args.output / 'provenance/historical-exclusions.json')
    history = json.loads((args.output / 'provenance/historical-exclusions.json').read_bytes())
    assert not set(history['ids']) & {r['id'] for r in rows}
    assert not set(history['normalized_stem_sha256']) & {
        hashlib.sha256(previous.stem(r['question']).encode()).hexdigest() for r in rows}
    manifest = {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                'questions': len(rows), 'subjects': dict(Counter(r['subject'] for r in rows)),
                'strata': {'same-source': [r['id'] for r in first],
                           'transfer': [r['id'] for r in second]},
                'scores_opened': False, 'selection': 'Both question panels fixed before any test grading. Report each source panel as well as the joint result; no outcome-based dropping or source selection.',
                'raw_examples_publishable': False,
                'training_rule': 'Only the existing1472training and240development questions may train or select models; neither ungraded confirmation panel may enter fitting.',
                'response_replay': 'Earlier label-free solver observations may be reused at exact model/prompt/seed settings. Every policy use remains charged. Freeze any new controller before its test replay, and never inspect test labels during selection.'}
    manifest['splits'] = {split: {'n': len(read(args.output / 'splits' / f'{split}-questions.jsonl')),
                                  'sha256': hashlib.sha256((args.output / 'splits' / f'{split}-questions.jsonl').read_bytes()).hexdigest()}
                          for split in ('train', 'dev', 'test')}
    (args.output / 'splits/selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('registered_at_utc', 'questions', 'subjects', 'scores_opened')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--transfer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
