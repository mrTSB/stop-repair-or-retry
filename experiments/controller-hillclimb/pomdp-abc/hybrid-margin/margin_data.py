"""Retire prior tests and reserve a new, historically disjoint holdout."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREVIOUS_CODE = HERE.parent / 'guided-repair'
sys.path.insert(0, str(PREVIOUS_CODE))
from extended_acquire import (DATA, digest, existing_questions, gsm8k, paired_archive,
                              read, science, stem)
# Earlier experiment modules adjust their own search paths. Prefer this area's
# modules after importing those dependencies (notably its distinct review_pilot).
sys.path.insert(0,str(HERE))

PREVIOUS = ROOT / 'results/controller-hillclimb/pomdp-abc/guided-repair/data'
SEED = 2026111921
MATH = Path('/Users/tanvirb/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/'
            'e53f048856ff4f594e959d75785d2c2d37b678ee/gsm8k-train.arrow')
SCIENCE = ROOT / '.cache/mmlu-pro/data/test-00000-of-00001.parquet'


def choose(rows, counts, label):
    selected = []
    for subject, count in counts.items():
        group = sorted((r for r in rows if r['subject'] == subject),
                       key=lambda r: hashlib.sha256(f'{SEED}:{label}:{r["id"]}'.encode()).digest())
        if len(group) < count:
            raise ValueError(f'Insufficient unused {subject}: {len(group)} < {count}')
        selected.extend(group[:count])
    return sorted(selected, key=lambda r: hashlib.sha256(f'{SEED}:shuffle:{r["id"]}'.encode()).digest())


def historical():
    rows = existing_questions() + paired_archive()
    for area in ('results/controller-hillclimb', 'runs/controller-hillclimb'):
        for p in (ROOT / area).rglob('*-questions.jsonl'):
            if 'hybrid-margin' not in p.parts:
                rows.extend(read(p))
    ids = {r['id'] for r in rows}
    texts = {stem(r['question']) for r in rows}
    excluded = json.loads((DATA.parent / 'provenance/selection-manifest.json').read_bytes())
    ids.update(f'gsm8k-train-{i:05d}' for i in excluded['historical_math_training_exclusions'])
    ids.update(excluded['historical_math_test_exclusions'])
    ids.update(excluded['historical_science_exclusions'])
    ids.update(r['id'] for r in excluded['source_quality_exclusions'])
    return ids, texts


def prepare(destination):
    retired = read(PREVIOUS / 'splits/test-questions.jsonl')
    dev = choose(retired, {'math': 120, 'chemistry': 60, 'physics': 60}, 'dev')
    dev_ids = {r['id'] for r in dev}
    train = read(PREVIOUS / 'splits/train-questions.jsonl')
    train += read(PREVIOUS / 'splits/dev-questions.jsonl')
    train += [r for r in retired if r['id'] not in dev_ids]
    ids, texts = historical()
    pool = gsm8k(MATH, 'train') + [r for g in science(SCIENCE).values() for r in g]
    available = []
    for row in sorted(pool, key=lambda r: hashlib.sha256(f'{SEED}:pool:{r["id"]}'.encode()).digest()):
        text = stem(row['question'])
        if row['id'] not in ids and text not in texts:
            available.append(row)
            ids.add(row['id'])
            texts.add(text)
    test = choose(available, {'math': 400, 'chemistry': 200, 'physics': 200}, 'fresh-test')
    groups = {'train': train, 'dev': dev, 'test': test}
    for key, rows in groups.items():
        assert len({r['id'] for r in rows}) == len(rows), key
        for other, second in groups.items():
            if key != other:
                assert not {r['id'] for r in rows} & {r['id'] for r in second}
                assert not {stem(r['question']) for r in rows} & {stem(r['question']) for r in second}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'splits').mkdir()
    manifest = {'seed': SEED, 'sources': {'math': digest(MATH), 'science': digest(SCIENCE)},
                'retirement': 'All preceding study tests are training/development only.',
                'unused_pool_before_reservation': dict(Counter(r['subject'] for r in available)),
                'splits': {}}
    for split, rows in groups.items():
        path = destination / 'splits' / f'{split}-questions.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest['splits'][split] = {'n': len(rows), 'subjects': dict(Counter(r['subject'] for r in rows)),
                                     'sha256': digest(path)}
    (destination / 'splits/selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def seed(destination):
    observations = []
    for split in ('train', 'dev', 'test'):
        path = PREVIOUS / f'{split}-observations.jsonl.gz'
        observations.extend(read(path))
    for split in ('train', 'dev'):
        ids = {r['id'] for r in read(destination / 'splits' / f'{split}-questions.jsonl')}
        selected = [r for r in observations if r['id'] in ids]
        keys = {(r['id'], r['kind'], r['path']) for r in selected}
        assert len(keys) == len(selected)
        target = destination / f'{split}-observations.jsonl'
        if target.exists():
            current = {(r['id'],r['kind'],r['path']):r['response'] for r in read(target)}
            assert all(current.get((r['id'],r['kind'],r['path'])) == r['response'] for r in selected)
        else:
            target.write_text(''.join(json.dumps(r) + '\n' for r in selected))
        print(f'{split}: retained {len(selected)} exact archived observations')


def save_history(destination):
    ids,texts=historical()
    test=read(destination/'splits/test-questions.jsonl')
    assert not ids & {r['id'] for r in test}
    assert not texts & {stem(r['question']) for r in test}
    folder=destination/'provenance';folder.mkdir(exist_ok=True)
    target=folder/'historical-exclusions.json'
    value={'ids':sorted(ids),'normalized_stem_sha256':sorted(hashlib.sha256(s.encode()).hexdigest() for s in texts),
           'note':'Reconstructed from the unchanged historical archives before any new holdout inference.'}
    encoded=json.dumps(value,indent=2)+'\n'
    if target.exists() and target.read_text()!=encoded:raise ValueError('Historical exclusion snapshot changed')
    target.write_text(encoded)
    path=destination/'splits/selection.json';manifest=json.loads(path.read_bytes())
    manifest['history_sha256']=digest(target);path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'excluded_ids':len(ids),'excluded_stems':len(texts),'fresh_questions':len(test),'sha256':digest(target)}))


def wait_complete(data,split,timeout):
    from extended_study import records
    deadline=time.monotonic()+timeout
    n=len(read(data/'splits'/f'{split}-questions.jsonl'))
    original=data/f'{split}-observations.jsonl'
    size=-1
    while True:
        path=original if original.exists() else original.with_suffix('.jsonl.gz')
        if path.exists() and path.stat().st_size!=size:
            size=path.stat().st_size
            raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
            if raw.endswith(b'\n') and raw.count(b'\n')==n*14:
                return records(data,split)
        if time.monotonic()>deadline:raise TimeoutError('Paired '+split+' data incomplete')
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'seed','history'))
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    {'prepare':prepare,'seed':seed,'history':save_history}[args.action](args.data)
