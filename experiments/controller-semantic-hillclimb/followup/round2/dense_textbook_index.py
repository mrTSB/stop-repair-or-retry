"""Question-independent, local-only BGE index of the frozen textbook corpus.

Uses the model author's CLS-pooling/normalization recipe. The index is an
offline artifact; future query forwards must be billed separately at inference.
No benchmark questions, answers, trajectories, or labels are read by this file.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = 'BAAI/bge-small-en-v1.5'
REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
FILES = ('README.md', 'config.json', 'model.safetensors', 'special_tokens_map.json',
         'tokenizer.json', 'tokenizer_config.json', 'vocab.txt')
INSTRUCTION = 'Represent this sentence for searching relevant passages: '
SCHEMA = 'question-independent-BGE-small-textbook-index-v1'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm < 1e-8):
        raise ValueError('Invalid or zero passage embedding')
    return values/norm


def contract(args):
    complete = read(args.corpus/'complete.json')
    manifest = read(args.corpus/'manifest.json')
    if (complete['schema'] != 'question-independent-broader-OCW-lecture-corpus-v2'
            or complete['benchmark_records_read'] or manifest['benchmark_records_read']
            or sha(args.corpus/'passages.jsonl') != manifest['sha256']):
        raise ValueError('Require the complete question-independent frozen broad corpus')
    for name, digest in complete['digests'].items():
        if sha(args.corpus/name) != digest: raise ValueError('Changed frozen corpus artifact')
    return {'schema': SCHEMA, 'source_sha256': sha(__file__),
        'corpus': str(args.corpus.resolve()), 'corpus_sha256': manifest['sha256'],
        'corpus_complete_sha256': sha(args.corpus/'complete.json'), 'passages': manifest['passages'],
        'model': MODEL, 'revision': REVISION, 'download_files': list(FILES), 'model_license': 'MIT',
        'model_card': 'https://huggingface.co/BAAI/bge-small-en-v1.5',
        'pooling': 'CLS followed by L2 normalization', 'max_length': 256,
        'passage_format': 'Title followed by newline and full passage text; no query instruction',
        'query_instruction': INSTRUCTION, 'batch_size': 16, 'device': 'cpu', 'threads': 2,
        'dtype': 'float32', 'cloud_calls': 0, 'training': False, 'benchmark_records_read': False,
        'inference_billing': 'Each actual query-token slot times measured model parameters; offline passage-index cost separately disclosed',
        'output': str(args.output.resolve())}


class Encoder:
    def __init__(self, assets):
        self.tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(assets, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True).to('cpu').eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def encode(self, texts, query=False):
        texts = [INSTRUCTION+v if query else v for v in texts]
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.inference_mode():
            values = self.model(**batch).last_hidden_state[:, 0].cpu().numpy()
        return normalize(values), {'actual_forward_token_slots': int(batch['input_ids'].numel()),
            'nonpadding_tokens': int(batch['attention_mask'].sum()), 'model_parameters': self.parameters}


def build(args, record):
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, revision=REVISION, allow_patterns=list(FILES), local_dir=args.output/'assets')
    if any(not (args.output/'assets'/name).is_file() for name in FILES):
        raise ValueError('Missing pinned retrieval-model asset')
    torch.set_num_threads(2)
    encoder = Encoder(args.output/'assets')
    rows = [json.loads(line) for line in (args.corpus/'passages.jsonl').open()]
    if len(rows) != record['passages'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Changed corpus row coverage')
    vectors, receipts = [], []
    for start in range(0, len(rows), 16):
        batch = rows[start:start+16]
        values, receipt = encoder.encode([r['title']+'\n'+r['text'] for r in batch])
        vectors.append(values)
        receipts.append({'start': start, 'rows': len(batch), **receipt})
        if len(receipts) % 32 == 0 or start+16 >= len(rows):
            print({'indexed_passages': min(start+16, len(rows)), 'total': len(rows)}, flush=True)
    values = np.concatenate(vectors)
    if values.shape != (len(rows), 384) or not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-5):
        raise ValueError('Unexpected normalized corpus embedding shape')
    np.save(args.output/'vectors.npy', values, allow_pickle=False)
    write(args.output/'passage-identities.json', [{'id': r['id'], 'subject': r['subject']} for r in rows])
    write(args.output/'offline-forward-receipts.json', receipts)
    write(args.output/'summary.json', {'schema': SCHEMA, 'passages': len(rows), 'dimensions': 384,
        'model_parameters': encoder.parameters, 'forward_calls': len(receipts),
        'offline_forward_token_slots': sum(r['actual_forward_token_slots'] for r in receipts),
        'offline_nonpadding_tokens': sum(r['nonpadding_tokens'] for r in receipts),
        'cloud_calls': 0, 'training': False, 'benchmark_records_read': False})
    if contract(args) != record: raise ValueError('Changed source or corpus during build')
    files = ['registration.json', 'vectors.npy', 'passage-identities.json', 'offline-forward-receipts.json', 'summary.json']
    files += ['assets/'+name for name in FILES]
    write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': sha(args.output/'registration.json'),
        'digests': {name: sha(args.output/name) for name in files}, 'cloud_calls': 0, 'benchmark_records_read': False})
    print(read(args.output/'summary.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'build'))
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    record = contract(args)
    if args.phase == 'register':
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output/'registration.json', record)
        print({'registration_sha256': sha(args.output/'registration.json'), 'model_calls': 0}, flush=True)
        return
    if (not args.registration_sha256 or sha(args.output/'registration.json') != args.registration_sha256
            or read(args.output/'registration.json') != record):
        raise ValueError('Changed registered model/corpus/indexing recipe')
    if {p.name for p in args.output.iterdir()} != {'registration.json'}:
        raise ValueError('Preserve previous index-build outputs; no silent restart')
    build(args, record)


if __name__ == '__main__': main()
