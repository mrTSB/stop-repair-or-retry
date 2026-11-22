"""Subject-filtered dense/lexical textbook retrieval with explicit query bills.

Returns factual excerpts separately from their compute receipts. Query caching
can accelerate offline replay, but each used query is charged one full forward
when comparing controllers. This module alone makes no controller gain claim.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dense_textbook_index as index
import active_textbook_retrieval_v2 as parsing

ROOT = Path(__file__).resolve().parents[4]
INDEX = ROOT/'runs/controller-semantic-followup/round2/dense-textbook-index-v1'
CORPUS = ROOT/'runs/controller-semantic-followup/round2/broad-textbooks-v2'


def rank(passages, dense, lexical, subject):
    dense, lexical = np.asarray(dense), np.asarray(lexical)
    if (subject not in ('chemistry', 'physics') or dense.shape != (len(passages),)
            or lexical.shape != dense.shape or not np.isfinite(dense).all() or not np.isfinite(lexical).all()):
        raise ValueError('Aligned finite subject-specific retrieval scores required')
    eligible = [i for i, row in enumerate(passages) if row['subject'] == subject]
    dense_order = sorted(eligible, key=lambda i: (-float(dense[i]), passages[i]['id']))[:60]
    lexical_order = sorted((i for i in eligible if lexical[i] > 0),
                           key=lambda i: (-float(lexical[i]), passages[i]['id']))[:60]
    fused = {}
    for order in (dense_order, lexical_order):
        for position, i in enumerate(order, 1): fused[i] = fused.get(i, 0.)+1/(60+position)
    selected, sources = [], set()
    for i in sorted(fused, key=lambda i: (-fused[i], passages[i]['id'])):
        row = passages[i]
        if row['source'] in sources: continue
        sources.add(row['source'])
        selected.append({'title': row['title'], 'text': ' '.join(row['text'].split()[:85]),
                         'source': row['source'], 'retrieval_score': fused[i]})
        if len(selected) == 2: break
    return selected


class Retriever:
    def __init__(self, directory=INDEX):
        self.directory = Path(directory)
        done = index.read(self.directory/'complete.json')
        record = index.read(self.directory/'registration.json')
        if (done['schema'] != index.SCHEMA or done['benchmark_records_read']
                or record['source_sha256'] != index.sha(index.__file__)
                or done['registration_sha256'] != index.sha(self.directory/'registration.json')
                or record['revision'] != index.REVISION or record['model'] != index.MODEL):
            raise ValueError('Require the exact completed question-independent semantic index')
        for name, digest in done['digests'].items():
            if index.sha(self.directory/name) != digest: raise ValueError('Changed semantic-index artifact')
        corpus = Path(record['corpus'])
        if index.sha(corpus/'passages.jsonl') != record['corpus_sha256']:
            raise ValueError('Changed underlying factual passages')
        if index.sha(corpus/'complete.json') != record['corpus_complete_sha256']:
            raise ValueError('Changed frozen corpus completion receipt')
        corpus_done = index.read(corpus/'complete.json')
        if index.sha(corpus/'tfidf.pkl') != corpus_done['digests']['tfidf.pkl']:
            raise ValueError('Changed frozen lexical index bytes')
        self.passages = [json.loads(line) for line in (corpus/'passages.jsonl').open()]
        identities = index.read(self.directory/'passage-identities.json')
        if identities != [{'id': r['id'], 'subject': r['subject']} for r in self.passages]:
            raise ValueError('Passage-vector identity/order mismatch')
        self.vectors = np.load(self.directory/'vectors.npy', allow_pickle=False)
        if (self.vectors.shape != (len(self.passages),384) or not np.isfinite(self.vectors).all()
                or not np.allclose(np.linalg.norm(self.vectors,axis=1),1.,atol=1e-5)):
            raise ValueError('Malformed frozen passage vectors')
        cached = pickle.loads((corpus/'tfidf.pkl').read_bytes())
        if cached['sha256'] != record['corpus_sha256']:
            raise ValueError('Stale lexical index; rebuilding is forbidden')
        self.vectorizer, self.matrix = cached['vectorizer'], cached['matrix']
        if self.matrix.shape[0] != len(self.passages): raise ValueError('Lexical row coverage mismatch')
        index.torch.set_num_threads(2)
        self.encoder = index.Encoder(self.directory/'assets')
        if self.encoder.parameters != index.read(self.directory/'summary.json')['model_parameters']:
            raise ValueError('Changed retrieval-encoder parameter count')

    @lru_cache(maxsize=128)
    def encoded(self, query):
        vectors, receipt = self.encoder.encode([query], query=True)
        return vectors[0], {'model': index.MODEL, 'revision': index.REVISION,
            'input_sha256': hashlib.sha256((index.INSTRUCTION+query).encode()).hexdigest(),
            **receipt, 'charged_query_forwards': 1,
            'compute': receipt['model_parameters']*receipt['actual_forward_token_slots']/3.09e9}

    def retrieve(self, query, subject):
        if not isinstance(query, str) or not query.strip() or len(query) > 800:
            raise ValueError('Only a bounded public retrieval query is accepted')
        vector, receipt = self.encoded(query)
        dense = self.vectors@vector
        lexical = (self.matrix@self.vectorizer.transform([query]).T).toarray().ravel()
        return rank(self.passages, dense, lexical, subject), dict(receipt)


@lru_cache(maxsize=1)
def retriever():
    return Retriever()


def evaluate_text(text, subject, public_question):
    if subject not in ('chemistry', 'physics') or not isinstance(public_question,str) or not public_question.strip():
        raise ValueError('A public science question is required')
    queries = parsing.queries(text)
    fallback = not queries
    if fallback: queries = [public_question.partition('\n\nOptions:\n')[0][:800]]
    results, charges, seen = [], [], set()
    for query in queries:
        found, receipt = retriever().retrieve(query,subject)
        charges.append(receipt)
        passages = []
        for entry in found:
            key = entry['source'],entry['text']
            if key in seen or len(seen) >= 3: continue
            seen.add(key)
            passages.append(entry)
        results.append({'query': query, 'ok': True, 'public_question_fallback': fallback, 'passages': passages})
    return results, charges
