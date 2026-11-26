"""Prepare the private bytes for cloud assembly of the complete runtime archive.

Already-public, content-addressed model files are fetched by the cloud assembler.
Every other file is transferred in this seed. The final archive contains both.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--public-map', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = json.loads((args.snapshot/'reproduction-files.json').read_bytes())
    public = json.loads(args.public_map.read_bytes())
    for name in ('reproduction-files.json', 'reproduction-bundle.json'):
        raw = (args.snapshot/name).read_bytes()
        files[name] = {'size_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    objects, entries = {}, {}
    for name, receipt in sorted(files.items()):
        stat = (args.snapshot/name).stat()
        entries[name] = {**receipt, 'mode': stat.st_mode & 0o777, 'mtime': stat.st_mtime}
        key = receipt['sha256']
        if key not in objects:
            objects[key] = {'size_bytes': receipt['size_bytes'], 'source': name}
            if key in public:
                objects[key]['url'] = public[key]
    recipe = {'source_commit': json.loads((args.snapshot/'reproduction-bundle.json').read_bytes())['source_commit'],
              'files': entries, 'objects': objects}
    raw = (json.dumps(recipe, sort_keys=True) + '\n').encode()
    (args.output/'transport-recipe.json').write_bytes(raw)
    archive = args.output/'abc-transport.tar.zst'
    with archive.open('xb') as out:
        compressor = subprocess.Popen(['zstd', '-T2', '-7', '--long=31', '-c'], stdin=subprocess.PIPE, stdout=out)
        with tarfile.open(fileobj=compressor.stdin, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            info = tarfile.TarInfo('recipe.json'); info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
            for key, obj in objects.items():
                if 'url' in obj:
                    continue
                info = tarfile.TarInfo('objects/'+key); info.size = obj['size_bytes']; info.mode = 0o600
                with (args.snapshot/obj['source']).open('rb') as stream:
                    tar.addfile(info, stream)
        compressor.stdin.close()
        if compressor.wait():
            raise RuntimeError('Transport compression failed')
    parts = []
    with archive.open('rb') as source:
        while chunk := source.read(128*1024**2):
            name = f'abc-transport.part-{len(parts):03d}'
            (args.output/name).write_bytes(chunk)
            parts.append({'name': name, 'size_bytes': len(chunk), 'sha256': hashlib.sha256(chunk).hexdigest()})
    record = {'source_commit': recipe['source_commit'], 'parts': parts,
              'recipe_sha256': hashlib.sha256(raw).hexdigest(),
              'public_bytes': sum(o['size_bytes'] for o in objects.values() if 'url' in o),
              'private_bytes': sum(o['size_bytes'] for o in objects.values() if 'url' not in o)}
    (args.output/'transport-ready.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({**{k: v for k, v in record.items() if k != 'parts'},
                      'parts': len(parts), 'transfer_bytes': sum(p['size_bytes'] for p in parts)}), flush=True)


if __name__ == '__main__':
    main()
