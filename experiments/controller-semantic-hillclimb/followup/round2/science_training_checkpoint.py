"""Atomic, bounded-copy exact training-state checkpoints; CPU/MPS RNG included."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
from safetensors.torch import load_file, save_file
import torch

SCHEMA = 'science-exact-training-state-v1'
CHUNK_BYTES = 32 * 1024 * 1024


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def sync_directory(path):
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, payload):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path); sync_directory(path.parent)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_mps': torch.mps.get_rng_state() if device == 'mps' else None}


def restore_rng(value, device):
    if (value['torch_mps'] is None) != (device != 'mps'):
        raise ValueError('Resume must retain the registered CPU/MPS RNG domain')
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch_cpu'])
    if device == 'mps':
        torch.mps.set_rng_state(value['torch_mps'])


class Writer:
    def __init__(self, directory, chunk_bytes=CHUNK_BYTES):
        self.directory, self.chunk_bytes, self.serial = Path(directory), chunk_bytes, 0
        self.files = {}; self.max_cpu_copy_bytes = 0

    def tensor(self, value):
        if value.layout != torch.strided or not value.is_contiguous():
            raise ValueError('Checkpoint requires contiguous strided tensors; never materialize a large hidden copy')
        count = max(1, self.chunk_bytes // value.element_size()); parts = []
        flat = value.detach().view(-1)
        for start in range(0, value.numel(), count):
            # This is the only device-to-CPU tensor copy, bounded by chunk_bytes.
            cpu = flat[start:start+count].to(device='cpu', copy=True).contiguous()
            self.max_cpu_copy_bytes = max(self.max_cpu_copy_bytes, cpu.numel() * cpu.element_size())
            name = f'tensor-{self.serial:06d}.safetensors'; self.serial += 1
            save_file({'value': cpu}, str(self.directory / name))
            with (self.directory / name).open('rb') as stream:
                os.fsync(stream.fileno())
            self.files[name] = sha(self.directory / name)
            parts.append({'file': name, 'start': start, 'count': cpu.numel()}); del cpu
        return {'kind': 'tensor', 'shape': list(value.shape), 'dtype': str(value.dtype),
                'device': str(value.device), 'parts': parts}

    def encode(self, value):
        if isinstance(value, torch.Tensor):
            return self.tensor(value)
        if isinstance(value, np.ndarray):
            if value.size > 8192:
                raise ValueError('Large NumPy training-state arrays need explicit streamed tensors')
            return {'kind': 'numpy', 'dtype': value.dtype.str, 'shape': list(value.shape), 'value': value.tolist()}
        if isinstance(value, dict):
            return {'kind': 'dict', 'items': [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {'kind': type(value).__name__, 'items': [self.encode(v) for v in value]}
        if value is None or type(value) in (str, bool, int, float):
            return {'kind': 'scalar', 'value': value}
        raise ValueError('Unsupported checkpoint state type: ' + str(type(value)))


class Reader:
    def __init__(self, directory):
        self.directory = Path(directory)

    def tensor(self, specification, destination=None):
        dtype_name = specification['dtype'].removeprefix('torch.')
        dtype = getattr(torch, dtype_name)
        if not isinstance(dtype, torch.dtype):
            raise ValueError('Invalid checkpoint tensor dtype')
        if destination is None:
            destination = torch.empty(specification['shape'], dtype=dtype, device=specification['device'])
        if (list(destination.shape) != specification['shape'] or destination.dtype != dtype
                or not destination.is_contiguous()):
            raise ValueError('Resume tensor shape/dtype/layout changed')
        flat = destination.detach().view(-1); offset = 0
        for part in specification['parts']:
            values = load_file(str(self.directory / part['file']), device='cpu')['value']
            if (part['start'] != offset or part['count'] != values.numel() or values.dtype != dtype
                    or values.ndim != 1):
                raise ValueError('Changed checkpoint tensor partition')
            with torch.no_grad():
                flat[offset:offset+values.numel()].copy_(values)
            offset += values.numel(); del values
        if offset != destination.numel():
            raise ValueError('Incomplete checkpoint tensor coverage')
        return destination

    def decode(self, value):
        kind = value['kind']
        if kind == 'tensor':
            return self.tensor(value)
        if kind == 'numpy':
            return np.asarray(value['value'], dtype=np.dtype(value['dtype'])).reshape(value['shape'])
        if kind == 'dict':
            return {self.decode(k): self.decode(v) for k, v in value['items']}
        if kind in ('tuple', 'list'):
            items = [self.decode(v) for v in value['items']]
            return tuple(items) if kind == 'tuple' else items
        if kind == 'scalar':
            return value['value']
        raise ValueError('Unrecognized checkpoint value kind')


def parameter_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [[names[id(parameter)] for parameter in group['params']] for group in optimizer.param_groups]


def load_manifest(root, binding, device):
    root = Path(root); pointer = json.loads((root / 'LATEST.json').read_bytes())
    directory = root / pointer['directory']
    if directory.parent != root or not directory.name.startswith('step-'):
        raise ValueError('Invalid checkpoint location')
    path = directory / 'manifest.json'
    if sha(path) != pointer['manifest_sha256']:
        raise ValueError('Checkpoint manifest differs from atomic latest pointer')
    manifest = json.loads(path.read_bytes())
    if manifest['schema'] != SCHEMA or manifest['binding'] != binding or manifest['device'] != device:
        raise ValueError('Checkpoint recipe/input/device changed')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(directory / name) != expected:
            raise ValueError('Checkpoint tensor shard changed')
    return directory, manifest


def save(root, model, optimizer, scheduler, progress, binding, device, history_path,
         chunk_bytes=CHUNK_BYTES, keep=2):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if device == 'mps':
        torch.mps.synchronize()
    if progress['step'] != scheduler.last_epoch:
        raise ValueError('Checkpoint optimizer/scheduler position differs')
    states = model.state_dict()
    if any(tensor.is_floating_point() and tensor.dtype != torch.float32 for tensor in states.values()):
        raise ValueError('Training checkpoint must preserve the full FP32 model, without half conversion')
    name = f"step-{progress['step']:06d}-{uuid.uuid4().hex[:12]}"
    temporary = root / (name + '.partial'); temporary.mkdir()
    writer = Writer(temporary, chunk_bytes)
    # Snapshot RNG first; serialization must not change the continuation state.
    random_state = rng_state(device)
    model_state = {name: writer.tensor(tensor) for name, tensor in states.items()}
    optimizer_state = writer.encode(optimizer.state_dict())
    scheduler_state = writer.encode(scheduler.state_dict())
    random_state = writer.encode(random_state)
    history = Path(history_path).read_bytes()
    if history and not history.endswith(b'\n'):
        raise ValueError('Do not checkpoint a partial optimization-history line')
    manifest = {'schema': SCHEMA, 'binding': binding, 'device': device, 'progress': progress,
        'model_class': type(model).__module__ + '.' + type(model).__qualname__,
        'optimizer_class': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
        'scheduler_class': type(scheduler).__module__ + '.' + type(scheduler).__qualname__,
        'optimizer_parameter_names': parameter_names(model, optimizer),
        'model_training': model.training, 'model': model_state, 'optimizer': optimizer_state,
        'scheduler': scheduler_state, 'rng': random_state, 'files': writer.files,
        'history': {'bytes': len(history), 'sha256': hashlib.sha256(history).hexdigest()},
        'maximum_tensor_cpu_copy_bytes': writer.max_cpu_copy_bytes, 'chunk_bytes': chunk_bytes,
        'checkpoint_selection': 'Latest recovery point only; final deployment remains fixed2160steps/epoch3'}
    atomic_json(temporary / 'manifest.json', manifest)
    directory = root / name; temporary.replace(directory)
    atomic_json(root / 'LATEST.json', {'directory': name, 'manifest_sha256': sha(directory / 'manifest.json')})
    # Retention is operational, never metric-based. Keep two complete recovery
    # generations and retain every retired manifest as provenance.
    generations = sorted((p for p in root.glob('step-*') if p.is_dir() and not p.name.endswith('.partial')),
                         key=lambda p: (json.loads((p/'manifest.json').read_bytes())['progress']['step'], p.stat().st_mtime_ns))
    for obsolete in generations[:-keep]:
        retired = root / 'retired-manifests'; retired.mkdir(exist_ok=True)
        atomic_bytes(retired/(obsolete.name+'.json'), (obsolete/'manifest.json').read_bytes())
        shutil.rmtree(obsolete)
    return {'directory': str(directory), 'manifest_sha256': sha(directory/'manifest.json'),
            'step': progress['step'], 'maximum_tensor_cpu_copy_bytes': writer.max_cpu_copy_bytes}


def restore(root, model, optimizer, scheduler, binding, device, history_path):
    root = Path(root)
    directory, manifest = load_manifest(root, binding, device)
    classes = [('model_class', model), ('optimizer_class', optimizer), ('scheduler_class', scheduler)]
    if any(manifest[key] != type(value).__module__ + '.' + type(value).__qualname__ for key, value in classes):
        raise ValueError('Resume model/optimizer/scheduler classes changed')
    if parameter_names(model, optimizer) != manifest['optimizer_parameter_names']:
        raise ValueError('Resume parameter group order changed')
    states = model.state_dict()
    if set(states) != set(manifest['model']):
        raise ValueError('Resume model tensors changed')
    reader = Reader(directory)
    for name, tensor in states.items():
        reader.tensor(manifest['model'][name], tensor)
    # Create the scheduler before restoring both states, as the installed
    # torch.optim.Optimizer.load_state_dict documentation requires.
    scheduler.load_state_dict(reader.decode(manifest['scheduler']))
    optimizer.load_state_dict(reader.decode(manifest['optimizer']))
    if scheduler.last_epoch != manifest['progress']['step']:
        raise ValueError('Restored optimizer/scheduler position differs')
    history_path = Path(history_path); history = history_path.read_bytes()
    prefix = history[:manifest['history']['bytes']]
    if len(prefix) != manifest['history']['bytes'] or hashlib.sha256(prefix).hexdigest() != manifest['history']['sha256']:
        raise ValueError('Checkpoint optimization-history prefix changed')
    if len(history) > len(prefix):
        # Preserve later uncheckpointed progress before rolling back the active
        # journal to the exact recovery state. Local updates may be replayed.
        abandoned = root / 'abandoned-attempt-history'; abandoned.mkdir(exist_ok=True)
        target = abandoned / (hashlib.sha256(history).hexdigest()+'.jsonl')
        if not target.exists():
            atomic_bytes(target, history)
        elif target.read_bytes() != history:
            raise ValueError('Preserved abandoned history changed')
        atomic_bytes(history_path, prefix)
    model.train(manifest['model_training'])
    restore_rng(reader.decode(manifest['rng']), device)
    return manifest['progress']


def save_deployable(model, tokenizer, directory, recipe, chunk_bytes=CHUNK_BYTES):
    """Same final FP16 deployment values as network.save, streamed without model.cpu()."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Never overwrite an existing final deployment archive')
    temporary = directory.with_name(directory.name+'.partial-'+uuid.uuid4().hex[:12]); temporary.mkdir()
    model.encoder.config.save_pretrained(temporary); tokenizer.save_pretrained(temporary)
    names, files, serial, max_copy = {}, {}, 0, 0
    for key, value in model.state_dict().items():
        if not value.is_contiguous():
            raise ValueError('Final streaming archive requires contiguous source tensors')
        rows = value.shape[0] if value.ndim else 1
        row_values = value.numel() // rows if rows else 0
        if row_values * value.element_size() > chunk_bytes:
            raise ValueError('A deployment tensor row exceeds the bounded CPU copy size')
        width = max(1, chunk_bytes // max(1, row_values * value.element_size()))
        names[key] = []
        for index, start in enumerate(range(0, rows, width)):
            part = value.detach()[start:start+width] if value.ndim else value.detach()
            cpu = part.to(device='cpu', copy=True).contiguous()
            max_copy = max(max_copy, cpu.numel()*cpu.element_size())
            if key not in ('numeric_mean','numeric_std'):
                cpu = cpu.half()
            part_name = key + f'__part_{index}'; names[key].append(part_name)
            filename = f'weights-{serial:04d}.safetensors'; serial += 1
            save_file({part_name: cpu}, str(temporary / filename)); del cpu
            with (temporary / filename).open('rb') as stream:
                os.fsync(stream.fileno())
            files[filename] = sha(temporary / filename)
    metadata = {**recipe, 'numeric_width':len(model.numeric_mean),
        'parameters':sum(p.numel() for p in model.parameters()), 'tensor_parts':names, 'weights':files,
        'streaming_archive': {'helper_sha256':sha(__file__), 'maximum_tensor_cpu_copy_bytes':max_copy,
                              'float_conversion':'FP32 training -> FP16 deployment, numeric buffers remain FP32'}}
    atomic_json(temporary/'recipe.json',metadata); temporary.replace(directory); sync_directory(directory.parent)
    return metadata
