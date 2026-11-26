"""Assemble and independently verify the entire runtime on a GitHub runner."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import io
import json
from pathlib import Path
import subprocess
import tarfile
import threading
import time
from urllib.request import Request, urlopen

REPO = 'mrTSB/stop-repair-or-retry'
TAG = 'abc-runtime-2026-11-25'
RELEASE = 458824938


def immutable_weight(name, size):
    return size > 0 and (name.endswith(('.safetensors', '.gguf', '/pytorch_model.bin'))
                         or name.startswith('external/ollama/models/blobs/')
                         or (name.startswith('.cache/') and '/blobs/' in name))


def gh(*args):
    return subprocess.check_output(['gh', *args], text=True)


def upload(path):
    for attempt in range(5):
        try:
            gh('release', 'upload', TAG, str(path), '--repo', REPO, '--clobber')
            print(json.dumps({'uploaded': path.name, 'bytes': path.stat().st_size}), flush=True)
            return
        except subprocess.CalledProcessError:
            if attempt == 4:
                raise
            time.sleep(5)


class Remote:
    def __init__(self, url, size):
        self.url, self.size, self.offset = url, size, 0
        self.response = None

    def read(self, size):
        result = bytearray()
        failures = 0
        while len(result) < size and self.offset < self.size:
            if self.response is None:
                headers = {'Accept-Encoding': 'identity'}
                if self.offset:
                    headers['Range'] = f'bytes={self.offset}-'
                self.response = urlopen(Request(self.url, headers=headers), timeout=300)
                if self.offset and self.response.status != 206:
                    raise ValueError('Range request was not honored')
            try:
                chunk = self.response.read(min(size-len(result), self.size-self.offset))
            except http.client.IncompleteRead as e:
                chunk = e.partial
                self.response.close(); self.response = None
            except (OSError, http.client.HTTPException):
                self.response.close(); self.response = None
                chunk = b''
            if not chunk:
                failures += 1
                if failures > 5:
                    raise OSError('Public model stream incomplete')
                if self.response is not None:
                    self.response.close(); self.response = None
                continue
            self.offset += len(chunk); result.extend(chunk)
        return bytes(result)

    def close(self):
        if self.response is not None:
            self.response.close()


class Hashed:
    def __init__(self, source):
        self.source = source; self.digest = hashlib.sha256(); self.size = 0

    def read(self, size):
        chunk = self.source.read(size); self.digest.update(chunk); self.size += len(chunk)
        return chunk


def initializations(recipe, work):
    """Include the pinned upstream checkpoints used to initialize all observers."""
    files, objects = recipe['files'], recipe['objects']
    additions, inventories = {}, []

    def local(name, raw):
        key = hashlib.sha256(raw).hexdigest()
        (work/'objects'/key).write_bytes(raw)
        objects[key] = {'size_bytes': len(raw), 'source': name}
        files[name] = {'sha256': key, 'size_bytes': len(raw), 'mode': 0o644, 'mtime': 0}
        return key

    for repository, revision in (
        ('microsoft/deberta-v3-base', '8ccc9b6f36199bec6961081d44eb72fb3f7353f3'),
        ('MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'),
        ('MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli', 'b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7'),
    ):
        with urlopen(f'https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true', timeout=120) as stream:
            metadata = json.load(stream)
        if metadata['sha'] != revision:
            raise ValueError('Pretrained initialization revision changed')
        siblings = {s['rfilename']: s for s in metadata['siblings']}
        weight = 'model.safetensors' if 'model.safetensors' in siblings else 'pytorch_model.bin'
        if weight not in siblings:
            raise ValueError('Pinned initialization has no supported checkpoint')
        prefix = '.cache/semantic-encoder/models--'+repository.replace('/', '--')+'/snapshots/'+revision+'/'
        record = {'repository': repository, 'revision': revision, 'cache': prefix, 'files': {}}
        for name, sibling in siblings.items():
            if '/' in name or not (name == weight or name.endswith('.json') or name in
                                    ('spm.model', 'vocab.txt', 'merges.txt', 'README.md', 'LICENSE')):
                continue
            target = prefix+name
            url = f'https://huggingface.co/{repository}/resolve/{revision}/{name}'
            if target not in files:
                if name == weight:
                    key, size = sibling['lfs']['sha256'], sibling['lfs']['size']
                    objects.setdefault(key, {'url': url, 'size_bytes': size, 'source': target})
                    files[target] = {'sha256': key, 'size_bytes': size, 'mode': 0o644, 'mtime': 0}
                else:
                    with urlopen(url, timeout=120) as stream:
                        local(target, stream.read())
                additions[target] = {k: files[target][k] for k in ('sha256', 'size_bytes')}
            record['files'][name] = {k: files[target][k] for k in ('sha256', 'size_bytes')}
        inventories.append(record)
    payload = (json.dumps(inventories, indent=2, sort_keys=True)+'\n').encode()
    name = 'results/reproduction/pretrained-initializations.json'
    local(name, payload)
    additions[name] = {k: files[name][k] for k in ('sha256', 'size_bytes')}
    original = json.loads((work/'objects'/files['reproduction-files.json']['sha256']).read_bytes())
    original.update(additions)
    local('reproduction-files.json', (json.dumps(original, indent=2, sort_keys=True)+'\n').encode())
    meta = json.loads((work/'objects'/files['reproduction-bundle.json']['sha256']).read_bytes())
    meta['source_snapshot_files'] = meta['files']
    meta['files'] = len(original)
    meta['logical_bytes'] = sum(v['size_bytes'] for v in original.values())
    meta['unique_bytes'] = sum(v['size_bytes'] for v in {v['sha256']: v for v in original.values()}.values())
    meta['pretrained_initializations'] = inventories
    for name, value in additions.items():
        group = meta['groups'].setdefault(name.split('/')[0], {'files': 0, 'bytes': 0})
        group['files'] += 1; group['bytes'] += value['size_bytes']
    meta['coverage'].append('Exact pinned upstream initialization checkpoints for all three DeBERTa observer families')
    local('reproduction-bundle.json', (json.dumps(meta, indent=2, sort_keys=True)+'\n').encode())
    return inventories


def main():
    work = Path('runtime-assembly'); work.mkdir()
    seed = work/'seed'; seed.mkdir()
    ready = json.loads(Path('transport-ready.json').read_bytes())
    gh('release', 'download', TAG, '--repo', REPO, '--pattern', 'abc-transport.part-*', '--dir', str(seed))
    decoder = subprocess.Popen(['zstd', '-d', '--long=31', '-c'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    errors = []
    def feed():
        try:
            for part in ready['parts']:
                path = seed/part['name']
                if path.stat().st_size != part['size_bytes']:
                    raise ValueError('Transport size mismatch')
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    while chunk := stream.read(8*1024**2):
                        digest.update(chunk); decoder.stdin.write(chunk)
                if digest.hexdigest() != part['sha256']:
                    raise ValueError('Transport digest mismatch')
                path.unlink()
        except BaseException as e:
            errors.append(e)
        finally:
            decoder.stdin.close()
    feeder = threading.Thread(target=feed); feeder.start()
    with tarfile.open(fileobj=decoder.stdout, mode='r|') as tar:
        for member in tar:
            if not member.isfile() or '..' in Path(member.name).parts or Path(member.name).is_absolute():
                raise ValueError('Unexpected transport entry')
            tar.extract(member, work, filter='data')
    feeder.join()
    if decoder.wait() or errors:
        raise RuntimeError('Transport unpacking failed: '+str(errors))
    raw = (work/'recipe.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != ready['recipe_sha256']:
        raise ValueError('Transport recipe changed')
    recipe = json.loads(raw)
    added_initializations = initializations(recipe, work)
    files = recipe['files']
    compressor = subprocess.Popen(['zstd', '-T2', '-3', '-c'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    checker = subprocess.Popen(['zstd', '-d', '-c'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    report = {}
    def produce():
        try:
            seen = {}
            with tarfile.open(fileobj=compressor.stdin, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for name, entry in sorted(files.items()):
                    key = entry['sha256']; obj = recipe['objects'][key]
                    info = tarfile.TarInfo('stop-repair-or-retry/'+name)
                    info.mode = entry['mode']; info.mtime = entry['mtime']
                    shared = immutable_weight(name, entry['size_bytes'])
                    if shared and key in seen:
                        info.type = tarfile.LNKTYPE; info.linkname = seen[key]; tar.addfile(info)
                        continue
                    if shared:
                        seen[key] = info.name
                    info.size = entry['size_bytes']
                    source = Remote(obj['url'], info.size) if 'url' in obj else (work/'objects'/key).open('rb')
                    hashed = Hashed(source)
                    try:
                        tar.addfile(info, hashed)
                    finally:
                        source.close()
                    if hashed.size != info.size or hashed.digest.hexdigest() != key:
                        raise ValueError('Source content hash mismatch: '+name)
                    if info.size > 1024**3:
                        print(json.dumps({'verified_input': name, 'bytes': info.size}), flush=True)
        except BaseException as e:
            errors.append(e)
        finally:
            compressor.stdin.close()
    def verify():
        try:
            seen = {}
            with tarfile.open(fileobj=checker.stdout, mode='r|') as tar:
                for member in tar:
                    name = member.name.removeprefix('stop-repair-or-retry/')
                    if name in seen or name not in files:
                        raise ValueError('Unexpected or duplicate final archive file')
                    if member.islnk():
                        if not immutable_weight(name, files[name]['size_bytes']):
                            raise ValueError('Mutable files must retain independent inodes: '+name)
                        actual = seen[member.linkname.removeprefix('stop-repair-or-retry/')]
                    elif member.isfile():
                        actual = {'sha256': hashlib.file_digest(tar.extractfile(member), 'sha256').hexdigest(),
                                  'size_bytes': member.size}
                    else:
                        raise ValueError('Unexpected archive entry type')
                    expected = {k: files[name][k] for k in ('sha256', 'size_bytes')}
                    if actual != expected:
                        raise ValueError('Final archive hash mismatch: '+name)
                    seen[name] = actual
            if set(seen) != set(files):
                raise ValueError('Final archive omissions')
            report.update(passed=True, archive_file_entries=len(seen), payload_files=len(seen)-2,
                          all_payload_sha256_checked=True, missing_files=0,
                          mutable_files_are_independent=True,
                          logical_bytes_checked=sum(v['size_bytes'] for v in seen.values()))
        except BaseException as e:
            errors.append(e)
    producer = threading.Thread(target=produce); verifier = threading.Thread(target=verify)
    producer.start(); verifier.start()
    pool = ThreadPoolExecutor(max_workers=3); pending = []; parts = []; whole = hashlib.sha256()
    def retire(path, future):
        future.result(); path.unlink()
    part = None; output = None; digest = None; count = 0
    try:
        while chunk := compressor.stdout.read(8*1024**2):
            checker.stdin.write(chunk); whole.update(chunk)
            view = memoryview(chunk)
            while view:
                if output is None:
                    part = work/f'abc-runtime.tar.zst.part-{len(parts):03d}'
                    output = part.open('xb'); digest = hashlib.sha256(); count = 0
                n = min(len(view), 1900*1024**2-count)
                output.write(view[:n]); digest.update(view[:n]); count += n; view = view[n:]
                if count == 1900*1024**2:
                    output.close(); output = None
                    parts.append({'name': part.name, 'size_bytes': count, 'sha256': digest.hexdigest()})
                    pending.append((part, pool.submit(upload, part)))
                    if len(pending) >= 4:
                        retire(*pending.pop(0))
        if output is not None:
            output.close(); parts.append({'name': part.name, 'size_bytes': count, 'sha256': digest.hexdigest()})
            pending.append((part, pool.submit(upload, part)))
    finally:
        checker.stdin.close()
    producer.join(); verifier.join()
    if compressor.wait() or checker.wait() or errors:
        raise RuntimeError('Final archive construction/verification failed: '+str(errors))
    for item in pending:
        retire(*item)
    pool.shutdown()
    record = {'schema': 'abc-runtime-release-parts-v1', 'archive_sha256': whole.hexdigest(),
              'archive_size_bytes': sum(p['size_bytes'] for p in parts), 'parts': parts,
              'source_commit': recipe['source_commit']}
    pages = json.loads(gh('api', '--paginate', '--slurp', f'repos/{REPO}/releases/{RELEASE}/assets?per_page=100'))
    assets = {a['name']: a for page in pages for a in page}
    for part in parts:
        actual = assets[part['name']]
        if actual['size'] != part['size_bytes'] or actual.get('digest') != 'sha256:'+part['sha256']:
            raise ValueError('GitHub upload hash/size mismatch')
    for name, value in (('archive.json', record), ('archive-verification.json', report)):
        path = work/name; path.write_text(json.dumps(value, indent=2)+'\n'); upload(path)
    path = work/'SHA256SUMS'; path.write_text(''.join(f'{p["sha256"]}  {p["name"]}\n' for p in parts)); upload(path)
    for name in ('reproduction-files.json', 'reproduction-bundle.json', 'results/reproduction/pretrained-initializations.json'):
        path = work/Path(name).name
        path.write_bytes((work/'objects'/files[name]['sha256']).read_bytes())
        upload(path)
    result = {'passed': True, 'source_commit': recipe['source_commit'], 'parts': len(parts),
              'all_remote_part_hashes_verified': True, 'archive_sha256': record['archive_sha256'],
              'archive_size_bytes': record['archive_size_bytes'], 'payload_verification': report,
              'pretrained_initializations': added_initializations}
    path = work/'cloud-complete.json'; path.write_text(json.dumps(result, indent=2)+'\n'); upload(path)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
