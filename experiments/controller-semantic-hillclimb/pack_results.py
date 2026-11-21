"""Archive owned experiment receipts, excluding large weights with explicit provenance."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from semantic_data import ROOT


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pack(args):
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = ROOT / 'runs/controller-semantic-hillclimb'
    corpus = ROOT / '.cache/semantic-textbooks'
    files = [p for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [p for p in corpus.iterdir() if p.is_file()
              and (p.name in ('passages.jsonl', 'tfidf.pkl', 'manifest.json')
                   or p.suffix == '.json' or 'LICENSE' in p.name)]
    files += [p for p in corpus.rglob('LICENSE*') if p.is_file() and p.parent != corpus]
    manifest = {'format': 1, 'files': {}, 'excluded_weight_files': {},
                'weight_provenance': {'private_modal_volume': 'srr-semantic-training-20261120',
                                      'remote_models': ['/verifier-v2', '/verifier-grounded-v1'],
                                      'fetch_command': 'uv run --with modal==1.6.4 --no-project python experiments/controller-semantic-hillclimb/fetch_verifier.py --remote REMOTE --output LOCAL'},
                'note': 'Every archived byte is checksummed. Large weight shards stay in the named private volume; model recipes contain their exact digests. Missing checkpoints must be fetched and verified before regenerating neural predictions.'}
    for source in sorted(files):
        relative = str(source.relative_to(ROOT))
        raw = source.read_bytes()
        if source.suffix == '.safetensors':
            manifest['excluded_weight_files'][relative] = {'sha256': digest(raw), 'bytes': len(raw)}
            continue
        if source.suffix == '.partial':
            continue
        compress = len(raw) > 16_384 and source.suffix not in ('.npz', '.gz')
        destination = args.output / 'archive' / (relative + ('.gz' if compress else ''))
        destination.parent.mkdir(parents=True, exist_ok=True)
        archived = gzip.compress(raw, compresslevel=9, mtime=0) if compress else raw
        destination.write_bytes(archived)
        manifest['files'][relative] = {'archive': str(destination.relative_to(args.output)),
                                       'compression': 'gzip' if compress else 'none',
                                       'raw_sha256': digest(raw), 'archive_sha256': digest(archived),
                                       'raw_bytes': len(raw), 'archive_bytes': len(archived)}
    manifest['source_files'] = {str(p.relative_to(ROOT)): digest(p.read_bytes())
                                for p in sorted((ROOT / 'experiments/controller-semantic-hillclimb').glob('*.py'))}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'archived_files': len(manifest['files']),
                      'archived_bytes': sum(v['archive_bytes'] for v in manifest['files'].values()),
                      'excluded_weight_files': len(manifest['excluded_weight_files'])}), flush=True)


def restore(args):
    manifest = json.loads((args.output / 'manifest.json').read_bytes())
    checked = 0
    for relative, receipt in manifest['files'].items():
        archived = (args.output / receipt['archive']).read_bytes()
        if digest(archived) != receipt['archive_sha256']:
            raise ValueError('Changed archive: ' + relative)
        raw = gzip.decompress(archived) if receipt['compression'] == 'gzip' else archived
        if digest(raw) != receipt['raw_sha256']:
            raise ValueError('Changed decoded bytes: ' + relative)
        destination = ROOT / relative
        if not args.verify_only:
            if destination.exists() and digest(destination.read_bytes()) != receipt['raw_sha256']:
                raise FileExistsError('Refusing to replace a different local file: ' + str(destination))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(raw)
        checked += 1
    print(json.dumps({'archive_checks_passed': checked, 'restored': not args.verify_only}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pack', 'restore'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    (pack if args.phase == 'pack' else restore)(args)
