"""Create a zstd tar with release-sized parts and checksums from a snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--part-mib', type=int, default=1900)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / 'abc-runtime.tar.zst'
    if archive.exists() or list(args.output.glob('abc-runtime.tar.zst.part-*')):
        parser.error('Archive output already exists; preserve it and use a new directory')
    with archive.open('xb') as output:
        process = subprocess.Popen(['zstd', '-T0', '-3', '-c'], stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                def progress(info):
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    if info.size > 1024**3:
                        print(json.dumps({'packing': info.name, 'bytes': info.size}), flush=True)
                    return info
                tar.add(snapshot, arcname='stop-repair-or-retry', filter=progress)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('zstd compression failed')
        except BaseException:
            process.kill()
            process.wait()
            raise
    subprocess.run(['zstd', '-t', str(archive)], check=True)
    parts = []
    whole = hashlib.sha256()
    with archive.open('rb') as source:
        remaining = archive.stat().st_size
        while remaining:
            name = f'abc-runtime.tar.zst.part-{len(parts):03d}'
            count, digest = 0, hashlib.sha256()
            with (args.output / name).open('xb') as output:
                while count < args.part_mib * 1024**2:
                    block = source.read(min(8 * 1024**2, args.part_mib * 1024**2 - count))
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    whole.update(block)
                    count += len(block)
            remaining -= count
            parts.append({'name': name, 'size_bytes': count, 'sha256': digest.hexdigest()})
            print(json.dumps({'part': name, 'size_bytes': count}), flush=True)
    record = {'schema': 'abc-runtime-release-parts-v1', 'archive_sha256': whole.hexdigest(),
              'archive_size_bytes': archive.stat().st_size, 'parts': parts,
              'source_commit': json.loads((snapshot / 'reproduction-bundle.json').read_bytes())['source_commit']}
    (args.output / 'archive.json').write_text(json.dumps(record, indent=2) + '\n')
    (args.output / 'SHA256SUMS').write_text(''.join(f'{part["sha256"]}  {part["name"]}\n' for part in parts))
    print(json.dumps({'archive_bytes': record['archive_size_bytes'], 'parts': len(parts),
                      'archive_sha256': record['archive_sha256']}), flush=True)


if __name__ == '__main__':
    main()
