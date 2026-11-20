"""Package a completed, frozen study for offline scoring and independent audit."""

import argparse
import gzip
import json
from pathlib import Path
import shutil

from extended_acquire import digest
from extended_study import evidence
from hillclimb import new_run, save


def pack(source, output, destination):
    frozen = json.loads((output / 'frozen.json').read_bytes())
    if frozen['evidence'] != evidence(source):
        raise ValueError('Frozen source or development evidence changed')
    summary = json.loads((output / 'test/summary.json').read_bytes())
    count = summary['methods']['C']['all']['n']
    metadata = json.loads((source / 'test-collection.json').read_bytes())
    if metadata['completed'] != count or metadata['questions'] != count:
        raise ValueError('The full held-out panel has not completed')
    with new_run(destination) as stage:
        shutil.copytree(source / 'splits', stage / 'data/splits')
        for split in ('train', 'dev', 'test'):
            path = source / f'{split}-observations.jsonl'
            (stage / 'data' / (path.name+'.gz')).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        cache = source / 'initial-cache.jsonl'
        (stage / 'data' / (cache.name+'.gz')).write_bytes(gzip.compress(cache.read_bytes(), mtime=0))
        shutil.copy2(source / 'test-collection.json', stage / 'data/test-collection.json')
        for name in ('frozen.json', 'development-search.json'):
            shutil.copy2(output / name, stage / name)
        shutil.copytree(output / 'test', stage / 'test')
        manifest = {str(p.relative_to(stage)): {'sha256': digest(p), 'size_bytes': p.stat().st_size}
                    for p in sorted(stage.rglob('*')) if p.is_file()}
        save(stage / 'archive-manifest.json', manifest)
    print(f'Packed {count} held-out questions into {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    pack(args.data, args.output, args.destination)


if __name__ == '__main__':
    main()
