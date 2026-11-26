"""Download and verify the complete release without a GitHub account."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

BASE = 'https://github.com/mrTSB/stop-repair-or-retry/releases/download/abc-runtime-2026-11-25'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('abc-download'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be between 1 and 16')
    args.destination.mkdir(parents=True, exist_ok=True)
    with urlopen(BASE+'/archive.json', timeout=120) as response:
        raw = response.read()
    record = json.loads(raw)
    (args.destination/'archive.json').write_bytes(raw)

    def download(part):
        if not re.fullmatch(r'abc-runtime\.tar\.zst\.part-\d+', part['name']):
            raise ValueError('Unexpected archive part name')
        path = args.destination/part['name']
        if path.exists():
            if path.stat().st_size != part['size_bytes'] or digest(path) != part['sha256']:
                raise ValueError('Existing file differs; use a fresh destination: '+str(path))
        else:
            temporary = path.with_name(path.name+'.partial')
            value = hashlib.sha256(); count = 0
            with urlopen(BASE+'/'+part['name'], timeout=300) as source, temporary.open('wb') as output:
                while block := source.read(8*1024**2):
                    output.write(block); value.update(block); count += len(block)
            if count != part['size_bytes'] or value.hexdigest() != part['sha256']:
                raise ValueError('Download size or SHA-256 differs: '+part['name'])
            temporary.replace(path)
        print('Verified '+part['name'], flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, record['parts']))
    whole = hashlib.sha256()
    for part in record['parts']:
        with (args.destination/part['name']).open('rb') as stream:
            while block := stream.read(8*1024**2):
                whole.update(block)
    if whole.hexdigest() != record['archive_sha256']:
        raise ValueError('Combined archive SHA-256 differs')
    (args.destination/'SHA256SUMS').write_text(''.join(f'{p["sha256"]}  {p["name"]}\n' for p in record['parts']))
    print(json.dumps({'passed': True, 'parts': len(record['parts']), 'bytes': record['archive_size_bytes'],
                      'archive_sha256': record['archive_sha256']}), flush=True)


if __name__ == '__main__':
    main()
