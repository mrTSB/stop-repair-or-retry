"""Verify every exported file, including the reserved question/reference sets."""

import argparse
import hashlib
import json
from pathlib import Path

from cloud_runtime import immutable_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    files = json.loads((root / 'reproduction-files.json').read_bytes())
    checked = 0
    for name, receipt in files.items():
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError('File escapes the bundle: ' + name)
        status = path.stat()
        if status.st_size != receipt['size_bytes']:
            raise ValueError('File size differs: ' + name)
        if status.st_nlink > 1 and not immutable_weight(name, status.st_size):
            raise ValueError('Mutable file must have an independent inode: ' + name)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != receipt['sha256']:
            raise ValueError('SHA-256 differs: ' + name)
        if name.startswith('external/ollama/models/blobs/sha256-') and actual != path.name.removeprefix('sha256-'):
            raise ValueError('Ollama blob differs from its content address: ' + name)
        checked += receipt['size_bytes']
    metadata = json.loads((root / 'reproduction-bundle.json').read_bytes())
    splits = {}
    for name, receipt in metadata['latest_splits'].items():
        rows = [json.loads(line) for line in (root / receipt['file']).read_bytes().splitlines()]
        if len(rows) != receipt['n'] or any(not row.get('answer') for row in rows):
            raise ValueError('Split or reference labels incomplete: ' + name)
        splits[name] = {row['id'] for row in rows}
        if len(splits[name]) != len(rows):
            raise ValueError('Duplicate question ID: ' + name)
    if any(splits[a] & splits[b] for a in splits for b in splits if a != b):
        raise ValueError('Latest train/dev/test IDs overlap')
    for tag, digest in metadata['selected_inference_manifests'].items():
        path = root / 'external/ollama/models/manifests/registry.ollama.ai/library' / tag.replace(':', '/')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Selected inference model manifest differs: ' + tag)
    print(json.dumps({'passed': True, 'files': len(files), 'bytes': checked,
                      'latest_splits': {name: len(ids) for name, ids in splits.items()},
                      'reserved_test_labels_included': True, 'test_inference_or_grading': False}), flush=True)


if __name__ == '__main__':
    main()
