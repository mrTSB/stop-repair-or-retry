"""Small local durability/source helpers for the new expanded-fit variants."""
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists(): raise ValueError('Preserved temporary output requires inspection: ' + str(temporary))
    with temporary.open('wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def jsonl(path, rows):
    atomic_bytes(path, ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows).encode())


def append(path, value):
    with Path(path).open('ab') as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode())
        stream.flush(); os.fsync(stream.fileno())


def npz(path, arrays):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    if temporary.exists(): raise ValueError('Preserved temporary output requires inspection: ' + str(temporary))
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


@contextmanager
def lock(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('Registered evidence/source changed: ' + str(path))


def sources(*roots):
    """Static project import closure; loaded unrelated tests are not roots."""
    def allowed(filename):
        if not filename: return None
        path = Path(filename).resolve()
        if (not path.is_file() or path.suffix != '.py' or not path.is_relative_to(PROJECT)
            or any(p.startswith('.') or p in ('site-packages', '__pycache__') for p in path.relative_to(PROJECT).parts)):
            return None
        return path
    def locate(name):
        if not name: return None
        if name in sys.modules:
            found = allowed(getattr(sys.modules[name], '__file__', None))
            if found: return found
        relative = Path(*name.split('.'))
        for folder in dict.fromkeys([str(HERE), *sys.path]):
            for path in (Path(folder or '.')/relative.with_suffix('.py'), Path(folder or '.')/relative/'__init__.py'):
                if path.is_file(): return allowed(path)
        return None
    pending = [(Path(module.__file__).resolve(), module.__name__) for module in roots]
    found = set()
    while pending:
        path, module = pending.pop()
        if path in found: continue
        if allowed(path) is None: raise ValueError('Source root outside project')
        found.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import): names = [v.name for v in node.names]
            elif isinstance(node, ast.ImportFrom):
                package = module.split('.')[:-node.level] if node.level else []
                name = '.'.join([*package, *([node.module] if node.module else [])])
                names = [name, *[name+'.'+v.name for v in node.names if v.name != '*']]
            for name in names:
                dependency = locate(name)
                if dependency is not None: pending.append((dependency, name))
    return {str(path): sha(path) for path in sorted(found)}


def structural_digest(value):
    """Saved learned values, excluding transient runtime caches; no byte claim."""
    caches = {'_model', '_tokenizer', '_recipe', '_cache', '_retriever', '_backgrounds',
              '_alphas', '_observed', '_values', '_information', '_asset_pool', '_current_context', '_review_context'}
    active = set()
    def encode(value):
        if value is None or type(value) in (bool, int, str): return [type(value).__name__, value]
        if isinstance(value, np.str_): return ['str', str(value)]
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value): raise ValueError('Nonfinite learned scalar')
            return ['float', float(value).hex()]
        if isinstance(value, np.integer): return ['int', int(value)]
        if isinstance(value, np.bool_): return ['bool', bool(value)]
        if isinstance(value, np.ndarray):
            if value.dtype.fields:
                return ['structured', list(value.shape), [[n, encode(value[n])] for n in value.dtype.names]]
            if value.dtype.hasobject: return ['object-array', list(value.shape), encode(value.tolist())]
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError('Nonfinite learned array')
            return ['array', value.dtype.str, list(value.shape), hashlib.sha256(value.tobytes(order='C')).hexdigest()]
        if isinstance(value, np.dtype): return ['dtype', value.str]
        if isinstance(value, Path): return ['path', str(value)]
        if isinstance(value, type): return ['type', value.__module__, value.__qualname__]
        marker = id(value)
        if marker in active: raise ValueError('Cyclic learned state')
        active.add(marker)
        try:
            if isinstance(value, dict):
                pairs = [[encode(k), encode(v)] for k, v in value.items()]
                return ['dict', sorted(pairs, key=lambda item: json.dumps(item[0], sort_keys=True))]
            if isinstance(value, (list, tuple)): return [type(value).__name__, [encode(v) for v in value]]
            if hasattr(value, '__dict__'):
                return ['object', type(value).__module__, type(value).__qualname__,
                    [[name, encode(item)] for name, item in sorted(vars(value).items()) if name not in caches]]
            raise ValueError('Unsupported learned value: ' + str(type(value)))
        finally: active.remove(marker)
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True).encode()).hexdigest()
