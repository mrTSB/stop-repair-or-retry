"""Replay unmodified experiment receipts at a new filesystem location.

Archived path strings remain logical identifiers, so source/evidence hashes and
registration equality checks retain their original meaning. Only filesystem and
import operations are redirected. This adapter is installed in the replay
process, never in the stored models, questions, receipts, or frozen source.
"""

import builtins
import functools
import io
import os
import pathlib
import sys


def install(root, mappings, *, deny_original=True):
    root = os.path.realpath(root)
    pairs = sorted(((old.rstrip('/'), os.path.normpath(os.path.join(root, new))) for old, new in mappings.items()),
                   key=lambda pair: len(pair[0]), reverse=True)
    inverse = sorted(((new, old) for old, new in pairs), key=lambda pair: len(pair[0]), reverse=True)

    def replace(value, table):
        if not isinstance(value, (str, bytes)):
            return value
        binary = isinstance(value, bytes)
        text = os.fsdecode(value)
        for source, target in table:
            if text == source or text.startswith(source + '/'):
                result = target + text[len(source):]
                return os.fsencode(result) if binary else result
        return value

    def physical(value):
        return replace(value, pairs)

    def logical(value):
        return replace(value, inverse)

    original_init = pathlib.PurePath.__init__

    def path_init(self, *parts):
        original_init(self, *(logical(p) if isinstance(p, str) else p for p in parts))

    pathlib.PurePath.__init__ = path_init
    pathlib.PurePath.__fspath__ = lambda self: physical(str(self))

    def wrap(function, positions=(0,)):
        @functools.wraps(function)
        def redirected(*args, **kwargs):
            args = list(args)
            for index in positions:
                if index < len(args):
                    args[index] = physical(args[index])
            for key in ('file', 'path', 'filename', 'src', 'dst'):
                if key in kwargs:
                    kwargs[key] = physical(kwargs[key])
            return function(*args, **kwargs)
        return redirected

    builtins.open = wrap(builtins.open)
    io.open = wrap(io.open)
    for name in ('stat', 'lstat', 'access', 'listdir', 'scandir', 'open', 'mkdir',
                 'rmdir', 'unlink', 'remove', 'chdir', 'readlink', 'chmod', 'utime'):
        setattr(os, name, wrap(getattr(os, name)))
    for name in ('rename', 'replace', 'link'):
        setattr(os, name, wrap(getattr(os, name), (0, 1)))

    class ImportPaths(list):
        def insert(self, index, value):
            super().insert(index, physical(value))

        def append(self, value):
            super().append(physical(value))

        def extend(self, values):
            super().extend(physical(v) for v in values)

        def __setitem__(self, key, value):
            super().__setitem__(key, [physical(v) for v in value] if isinstance(key, slice) else physical(value))

    sys.path = ImportPaths(physical(p) for p in sys.path)
    os.chdir(root)
    # These extension modules accept filename strings directly in Rust/C++.
    # Map their public loading boundary as well as Python's filesystem calls.
    try:
        import safetensors
        import safetensors.torch
        safetensors.safe_open = wrap(safetensors.safe_open)
        safetensors.torch.safe_open = wrap(safetensors.torch.safe_open)
        safetensors.torch.load_file = wrap(safetensors.torch.load_file)
    except ImportError:
        pass
    try:
        import transformers
        for name in ('AutoTokenizer', 'AutoConfig', 'AutoModel',
                     'AutoModelForSequenceClassification', 'AutoModelForCausalLM'):
            cls = getattr(transformers, name)
            original = cls.from_pretrained
            def pretrained(current, filename, *args, _original=original, **kwargs):
                filename = physical(os.fspath(filename)) if isinstance(filename, os.PathLike) else physical(filename)
                return _original(filename, *args, **kwargs)
            cls.from_pretrained = classmethod(pretrained)
    except ImportError:
        pass
    if deny_original:
        # An uncaught legacy string must fail instead of silently using a file
        # from the producer's checkout during a relocation test.
        def audit(event, args):
            if event not in ('open', 'os.listdir', 'os.scandir', 'os.chdir') or not args:
                return
            value = args[0]
            if isinstance(value, (str, bytes)) and physical(value) != value:
                raise RuntimeError('Unmapped original path reached filesystem: ' + os.fsdecode(value))
        sys.addaudithook(audit)
    return physical, logical
