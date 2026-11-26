"""Snapshot all experiment runtime files and create a content-verified bundle.

Use uv run --python 3.12 --no-project python scripts/reproduction/build.py --help.
The snapshot is outside the checkout so it cannot recursively include itself.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root, destination = args.root.resolve(), args.destination.resolve()
    if destination.exists() or destination.is_relative_to(root):
        parser.error('Use a new destination outside the checkout')
    destination.mkdir(parents=True)
    origins = {str(root): '.'}
    roots = [(root / 'runs', 'runs'), (root / '.cache', '.cache'), (root / '.venv', '.venv'),
             (Path.home() / '.ollama/models', 'external/ollama/models'),
             (Path.home() / '.cache/huggingface/datasets/openai___gsm8k',
              'external/huggingface/datasets/openai___gsm8k'),
             (Path.home() / '.cache/huggingface/hub/datasets--openai--gsm8k',
              'external/huggingface/hub/datasets--openai--gsm8k')]
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    sources = {name: root / name for name in tracked if name}
    for path in (root / 'scripts/reproduction').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            sources[str(path.relative_to(root))] = path
    for source, relative in roots:
        if not source.is_dir():
            raise FileNotFoundError('Required runtime root missing: ' + str(source))
        if not source.is_relative_to(root):
            origins[str(source)] = relative
        for folder, dirs, files in os.walk(source):
            dirs.sort()
            for name in sorted(files):
                path = Path(folder) / name
                sources[str(Path(relative) / path.relative_to(source))] = path
    entries = {}
    content = {}
    counts, sizes = Counter(), Counter()
    for index, (relative, source) in enumerate(sorted(sources.items()), 1):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved = source.resolve(strict=True)
        before = resolved.stat()
        digest = sha(resolved)
        after = resolved.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Runtime changed while snapshotting: ' + str(source))
        key = digest, before.st_size
        # Copies are independent of the original checkout; identical bytes in
        # the snapshot share an inode and a tar hard-link entry.
        if key in content:
            os.link(content[key], target)
        else:
            shutil.copy2(resolved, target)
            content[key] = target
        entries[relative] = {'sha256': digest, 'size_bytes': before.st_size}
        group = relative.split('/')[0]
        counts[group] += 1
        sizes[group] += before.st_size
        if index % 250 == 0 or before.st_size > 1024**3:
            print(json.dumps({'files': index, 'of': len(sources), 'logical_bytes': sum(sizes.values()),
                              'unique_bytes': sum(k[1] for k in content), 'latest': relative}), flush=True)
    splits = {}
    for name in ('train', 'dev', 'test'):
        relative = f'runs/controller-semantic-followup/round2/data/splits/{name}-questions.jsonl'
        rows = [json.loads(line) for line in (destination / relative).read_bytes().splitlines()]
        if any('answer' not in r or not r['answer'] for r in rows):
            raise ValueError('Missing requested reference labels: ' + relative)
        splits[name] = {'file': relative, 'n': len(rows), 'subjects': dict(Counter(r['subject'] for r in rows)),
                        **entries[relative]}
    record = {'schema': 'complete-abc-reproduction-bundle-v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip(),
              'path_mappings': origins, 'files': len(entries), 'logical_bytes': sum(sizes.values()),
              'unique_bytes': sum(k[1] for k in content), 'groups': {k: {'files': counts[k], 'bytes': sizes[k]} for k in counts},
              'latest_splits': splits, 'reserved_test_included_with_labels': True,
              'selected_inference_manifests': {
                  'qwen3.5:4b': '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd',
                  'qwen3.5:35b': '3460ffeede5453ead027dbd2f821b12ad0aa3de54630971993babdb2165221f7'},
              'reserved_test_inference_or_grading_performed_by_export': False,
              'coverage': ['All tracked repository files and reproduction tools',
                           'Every file under runs/, .cache/ and .venv/, including selected, intermediate and failed candidates',
                           'Every locally installed Ollama model manifest and referenced blob',
                           'Original GSM8K Arrow and Hugging Face dataset caches'],
              'file_manifest': 'reproduction-files.json'}
    write(destination / 'reproduction-files.json', entries)
    write(destination / 'reproduction-bundle.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('path_mappings', 'coverage')}), flush=True)


if __name__ == '__main__':
    main()
