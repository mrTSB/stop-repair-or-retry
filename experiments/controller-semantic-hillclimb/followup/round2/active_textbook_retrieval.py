"""Bounded, deterministic lookup in the already sealed local textbook corpus."""
from functools import lru_cache
import hashlib
from pathlib import Path
import pickle
import re

from evidence import Retriever

CORPUS = Path(__file__).resolve().parents[4]/'runs/controller-semantic-followup/round2/advanced-textbooks-v1'
MAX_QUERIES, MAX_CHARS, MAX_PASSAGES = 2, 200, 3


def queries(text):
    result = []
    for match in re.finditer(r'<search>\s*(.*?)\s*</search>', text, flags=re.I | re.S):
        value = ' '.join(match.group(1).split())
        if value and value not in result: result.append(value)
        if len(result) == MAX_QUERIES: break
    return result


@lru_cache(maxsize=1)
def retriever():
    # Acquisition verifies every corpus file before constructing this reader.
    if not (CORPUS/'tfidf.pkl').is_file(): raise ValueError('Frozen index required; no implicit rebuilding')
    cached = pickle.loads((CORPUS/'tfidf.pkl').read_bytes())
    if cached['sha256'] != hashlib.sha256((CORPUS/'passages.jsonl').read_bytes()).hexdigest():
        raise ValueError('Stale frozen index; rebuilding is forbidden')
    return Retriever(CORPUS)


def evaluate_text(text, subject):
    if subject not in ('chemistry', 'physics'): raise ValueError('Declared science subjects only')
    results, seen = [], set()
    for query in queries(text):
        if len(query) > MAX_CHARS:
            results.append({'query': query[:MAX_CHARS], 'ok': False, 'error': 'Query exceeds200 characters', 'passages': []})
            continue
        found = retriever().retrieve({'domain': 'science', 'subject': subject, 'question': query})
        passages = []
        for entry in found:
            key = entry['source'], entry['text']
            if key in seen or len(seen) >= MAX_PASSAGES: continue
            seen.add(key)
            passages.append({k: entry[k] for k in ('title', 'text', 'source', 'retrieval_score')})
        results.append({'query': query, 'ok': True, 'passages': passages})
    return results
