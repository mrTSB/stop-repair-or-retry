"""Placeholder-safe active retrieval with an explicit public-question fallback."""
import re

import active_textbook_retrieval as previous

CORPUS = previous.CORPUS
MAX_QUERIES, MAX_CHARS, MAX_PASSAGES = previous.MAX_QUERIES, previous.MAX_CHARS, previous.MAX_PASSAGES
MAX_FALLBACK_CHARS = 800
retriever = previous.retriever
PLACEHOLDERS = {'query', 'search query', 'your query', 'concept', 'scientific concept',
                'scientific concept or governing equation', 'keywords', 'search terms'}


def queries(text):
    result = []
    for match in re.finditer(r'<search>\s*(.*?)\s*</search>', text, flags=re.I | re.S):
        value = ' '.join(re.sub(r'<[^>]*>', ' ', match.group(1)).split())
        words = re.findall(r'[A-Za-z]{3,}', value.casefold())
        if not words or value.casefold() in PLACEHOLDERS or len(value) > MAX_CHARS: continue
        if value not in result: result.append(value)
        if len(result) == MAX_QUERIES: break
    return result


def evaluate_text(text, subject, public_question):
    if subject not in ('chemistry', 'physics') or not isinstance(public_question, str) or not public_question.strip():
        raise ValueError('A public science question is required for deterministic fallback')
    selected = queries(text); fallback = not selected
    if fallback:
        selected = [public_question.partition('\n\nOptions:\n')[0][:MAX_FALLBACK_CHARS]]
    results, seen = [], set()
    for query in selected:
        found = retriever().retrieve({'domain': 'science', 'subject': subject, 'question': query})
        passages = []
        for entry in found:
            key = entry['source'], entry['text']
            if key in seen or len(seen) >= MAX_PASSAGES: continue
            seen.add(key); passages.append({k: entry[k] for k in ('title', 'text', 'source', 'retrieval_score')})
        results.append({'query': query, 'ok': True, 'public_question_fallback': fallback, 'passages': passages})
    return results
