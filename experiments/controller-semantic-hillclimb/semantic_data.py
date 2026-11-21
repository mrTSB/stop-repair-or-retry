"""Retire exposed panels, keep exact receipts, and reserve unused same-source questions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS_CODE = ROOT / 'experiments/controller-hillclimb/pomdp-abc/hybrid-margin'
sys.path.insert(0, str(PREVIOUS_CODE))
import margin_data as previous
from extended_study import records
from late_pairs import merged_records
from study_common import sha
sys.path.insert(0, str(HERE))

PRIMARY = ROOT / 'results/controller-hillclimb/pomdp-abc/hybrid-margin'
SEED = 2026112021


def read(path):
    return previous.read(path)


def choose(rows, counts, label):
    selected = []
    for subject, n in counts.items():
        group = sorted((r for r in rows if r['subject'] == subject),
                       key=lambda r: hashlib.sha256(f'{SEED}:{label}:{r["id"]}'.encode()).digest())
        if len(group) < n:
            raise ValueError(f'Only {len(group)} unused {subject} questions, need {n}')
        selected.extend(group[:n])
    return sorted(selected, key=lambda r: hashlib.sha256(f'{SEED}:{label}:shuffle:{r["id"]}'.encode()).digest())


def write(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def prepare(destination):
    all_rows = [r for split in ('train', 'dev', 'test')
                for r in read(PRIMARY / 'data/splits' / f'{split}-questions.jsonl')]
    retired = read(PRIMARY / 'data/splits/test-questions.jsonl')
    dev = choose(retired, {'math': 120, 'chemistry': 60, 'physics': 60}, 'new-development')
    dev_ids = {r['id'] for r in dev}
    train = [r for r in all_rows if r['id'] not in dev_ids]
    history = json.loads((PRIMARY / 'data/provenance/historical-exclusions.json').read_bytes())
    excluded_ids = set(history['ids']) | {r['id'] for r in all_rows}
    excluded_stems = set(history['normalized_stem_sha256'])
    for base in (ROOT / 'results', ROOT / 'runs'):
        for p in base.rglob('*-questions.jsonl'):
            for r in read(p):
                excluded_ids.add(r['id'])
                excluded_stems.add(hashlib.sha256(previous.stem(r['question']).encode()).hexdigest())
    pool = previous.gsm8k(previous.MATH, 'train')
    pool += [r for group in previous.science(previous.SCIENCE).values() for r in group]
    available = []
    for r in sorted(pool, key=lambda r: hashlib.sha256(f'{SEED}:pool:{r["id"]}'.encode()).digest()):
        stem = hashlib.sha256(previous.stem(r['question']).encode()).hexdigest()
        if r['id'] not in excluded_ids and stem not in excluded_stems:
            available.append(r)
            excluded_ids.add(r['id'])
            excluded_stems.add(stem)
    counts = dict(Counter(r['subject'] for r in available))
    # Keep the original 50/25/25 task mixture; do not switch datasets for a desired ordering.
    quarters = min(48, counts.get('chemistry', 0), counts.get('physics', 0), counts.get('math', 0) // 2)
    if quarters < 30:
        raise ValueError('Too few historically unused same-source questions; choose a new validation source explicitly')
    test = choose(available, {'math': 2 * quarters, 'chemistry': quarters, 'physics': quarters}, 'untouched-test')
    groups = {'train': train, 'dev': dev, 'test': test}
    for name, rows in groups.items():
        assert len({r['id'] for r in rows}) == len(rows)
        assert len({previous.stem(r['question']) for r in rows}) == len(rows)
        for other, values in groups.items():
            if other != name:
                assert not {r['id'] for r in rows} & {r['id'] for r in values}
                assert not {previous.stem(r['question']) for r in rows} & {previous.stem(r['question']) for r in values}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'splits').mkdir()
    (destination / 'provenance').mkdir()
    manifest = {'seed': SEED, 'retirement': 'All earlier study panels, including the reported800, are exposed and retired.',
                'train_questions': len(train), 'development_questions': len(dev),
                'unused_pool_before_reservation': counts,
                'fresh_panel_reason': 'Same source and 50/25/25 mixture; remaining chemistry pool limits panel size.',
                'sources': {'math': sha(previous.MATH), 'science': sha(previous.SCIENCE)}, 'splits': {}}
    for split, rows in groups.items():
        p = destination / 'splits' / f'{split}-questions.jsonl'
        write(p, rows)
        manifest['splits'][split] = {'n': len(rows), 'subjects': dict(Counter(r['subject'] for r in rows)), 'sha256': sha(p)}
    (destination / 'splits/selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    exclusions = {'ids': sorted(set(history['ids']) | {r['id'] for r in all_rows}),
                  'note': 'Fresh selection additionally checks all recorded historical IDs and normalized stems.'}
    (destination / 'provenance/retirement.json').write_text(json.dumps(exclusions, indent=2) + '\n')
    for split, rows in (('train', train), ('dev', dev)):
        ids = {r['id'] for r in rows}
        for suffix in ('observations', 'late-observations'):
            receipts = [r for old_split in ('train', 'dev', 'test')
                        for r in read(PRIMARY / 'data' / f'{old_split}-{suffix}.jsonl.gz') if r['id'] in ids]
            assert len({(r['id'], r['kind'], r['path']) for r in receipts}) == len(receipts)
            write(destination / f'{split}-{suffix}.jsonl', receipts)
        records(destination, split, False)
        merged_records(destination, split, False)
    print(json.dumps(manifest, indent=2), flush=True)


def seal(destination):
    history = json.loads((PRIMARY / 'data/provenance/historical-exclusions.json').read_bytes())
    ids = set(history['ids']); stems = set(history['normalized_stem_sha256'])
    for base in (ROOT / 'results', ROOT / 'runs'):
        for p in base.rglob('*-questions.jsonl'):
            if destination.resolve() in p.resolve().parents:
                continue
            for row in read(p):
                ids.add(row['id'])
                stems.add(hashlib.sha256(previous.stem(row['question']).encode()).hexdigest())
    fresh = read(destination / 'splits/test-questions.jsonl')
    assert not ids & {r['id'] for r in fresh}
    assert not stems & {hashlib.sha256(previous.stem(r['question']).encode()).hexdigest() for r in fresh}
    p = destination / 'provenance/historical-exclusions.json'
    p.write_text(json.dumps({'ids':sorted(ids), 'normalized_stem_sha256':sorted(stems)}, indent=2) + '\n')
    p = destination / 'splits/selection.json'; manifest = json.loads(p.read_bytes())
    manifest['history_sha256'] = sha(destination / 'provenance/historical-exclusions.json')
    p.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'fresh_questions':len(fresh), 'excluded_ids':len(ids), 'excluded_stems':len(stems)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    (seal if args.seal else prepare)(args.output)
