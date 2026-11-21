"""NLI-initialized discriminative observation model with answer-independent textbook evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

BASE_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'


class Network(nn.Module):
    def __init__(self, encoder, width, mean=None, std=None):
        super().__init__()
        self.encoder = encoder
        self.register_buffer('numeric_mean', torch.zeros(width) if mean is None else torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('numeric_std', torch.ones(width) if std is None else torch.tensor(std, dtype=torch.float32))
        self.norm = nn.LayerNorm(encoder.config.hidden_size)
        self.project = nn.Sequential(nn.Linear(encoder.config.hidden_size + width, 256), nn.GELU(), nn.Dropout(.1))
        self.state = nn.Linear(256, 4)
        self.plain = nn.Linear(256, 8)
        self.guided = nn.Linear(256, 8)

    def forward(self, input_ids, attention_mask, numeric, token_type_ids=None):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        text = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
        numbers = ((numeric - self.numeric_mean) / self.numeric_std).clamp(-6, 6)
        joint = self.project(torch.cat((self.norm(text), numbers), dim=1))
        return self.state(joint), self.plain(joint), self.guided(joint)


def encode(tokenizer, rows, length):
    result = tokenizer([r['left'] for r in rows], [r['right'] for r in rows], padding='max_length',
                       truncation='longest_first', max_length=length, return_tensors='pt')
    result['numeric'] = torch.tensor(np.asarray([r['numeric'] for r in rows]), dtype=torch.float32)
    return result


def predict(model, tensors, device='cpu', batch=32):
    outputs = [[], [], []]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors['input_ids']), batch):
            values = model(**{k:v[start:start+batch].to(device) for k,v in tensors.items()})
            for destination, value in zip(outputs, values):
                destination.append(value.float().cpu().numpy())
    return tuple(np.concatenate(values) for values in outputs)


def save(model, tokenizer, directory, recipe):
    directory.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.encoder.config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    # Split even the large embedding tensor, keeping every artifact below 80 MB.
    shards, names, current, size = [], {}, {}, 0
    for key, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        if key not in ('numeric_mean', 'numeric_std'):
            tensor = tensor.half()
        chunks = tensor.chunk(max(1, (tensor.numel() * tensor.element_size() + 69_999_999) // 70_000_000), dim=0) if tensor.ndim else [tensor]
        names[key] = []
        for i, chunk in enumerate(chunks):
            name = key + f'__part_{i}'
            chunk = chunk.contiguous()
            amount = chunk.numel() * chunk.element_size()
            if size + amount > 75_000_000 and current:
                shards.append(current); current, size = {}, 0
            current[name] = chunk; names[key].append(name); size += amount
    if current:
        shards.append(current)
    fingerprints = {}
    for i, tensors in enumerate(shards):
        name = f'weights-{i:02d}.safetensors'
        save_file(tensors, str(directory / name))
        fingerprints[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    metadata = {**recipe, 'numeric_width':len(model.numeric_mean),
                'parameters':sum(p.numel() for p in model.parameters()), 'tensor_parts':names, 'weights':fingerprints}
    (directory / 'recipe.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def load(directory):
    directory = Path(directory)
    recipe = json.loads((directory / 'recipe.json').read_bytes())
    tensors = {}
    for name, digest in recipe['weights'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Observation-model weight shard changed')
        tensors.update(load_file(str(directory / name)))
    state = {key:torch.cat([tensors[n] for n in parts], dim=0) if len(parts) > 1 else tensors[parts[0]]
             for key, parts in recipe['tensor_parts'].items()}
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(directory, local_files_only=True), trust_remote_code=False)
    model = Network(encoder, recipe['numeric_width'])
    model.load_state_dict(state); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_fast=False)
    return model, tokenizer, recipe
