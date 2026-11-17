"""Keep gate-time observations separate from paired labels and retry-time evidence."""

from dataclasses import dataclass
import json

import numpy as np

from data import MODEL_DIGEST, PREVIOUS, read, sha
from evaluate import check_disjoint, fresh, jsonl, published, validate_response
from scoring import canonical_answer, correct
from signals import compact_features, question_features, recovery, short_features


CONFIDENCE_WIDTH = 22
RETRY_WIDTH = 12
SHORT_INPUT_OFFSETS = {'math': -8, 'science': -26}


@dataclass
class Observed:
    ids: tuple
    domains: np.ndarray
    valid: np.ndarray
    compact_tokens: np.ndarray
    prompt_tokens: np.ndarray
    raw: np.ndarray

    def take(self, mask):
        return Observed(tuple(key for key, include in zip(self.ids, mask) if include), self.domains[mask],
                        self.valid[mask], self.compact_tokens[mask], self.prompt_tokens[mask], self.raw[mask])


@dataclass
class Paired:
    first: Observed
    compact_correct: np.ndarray
    short_correct: np.ndarray
    short_valid: np.ndarray
    disagree: np.ndarray
    comparison: np.ndarray
    short_tokens: np.ndarray

    def take(self, mask):
        return Paired(self.first.take(mask), self.compact_correct[mask], self.short_correct[mask], self.short_valid[mask],
                      self.disagree[mask], self.comparison[mask], self.short_tokens[mask])


def question_rows(source, split):
    file = source / 'splits' / f'{split}-questions.jsonl'
    selection = json.loads((source / 'splits' / 'selection.json').read_bytes())['splits'][split]
    rows = read(file)
    if sha(file) != selection['sha256'] or len(rows) != selection['n']:
        raise ValueError(f'Changed {split} split')
    return rows


def records(source, name):
    target = source / f'{name}.jsonl'
    return jsonl(target if target.exists() else target.with_suffix('.jsonl.gz'))


def index_responses(questions, replies):
    by_id = {entry['id']: entry for entry in replies}
    expected = {row['id'] for row in questions}
    if set(by_id) != expected or len(by_id) != len(replies) or len(expected) != len(questions):
        raise ValueError('Response and question coverage differ')
    return by_id


def new_pairs(source, split):
    questions = question_rows(source, split)
    by_id = index_responses(questions, records(source, f'{split}-pairs'))
    merged = []
    for row in questions:
        rec = by_id[row['id']]
        validate_response(row, rec['compact'], rec['short'])
        merged.append({**row, 'compact': rec['compact'], 'short': rec['short']})
    return merged


def archived():
    rows = [row for split in ('train', 'dev', 'test') for row in published(split)]
    rows.extend(row for split in ('train', 'test') for row in fresh(PREVIOUS, split))
    measurements = {}
    for split in ('train', 'dev'):
        path = PREVIOUS / 'pilots' / 'confidence' / f'{split}.jsonl.gz'
        measurements.update({entry['id']: entry for entry in jsonl(path)})
    for row in rows:
        entry = measurements.get(row['id'])
        if entry is None:
            continue
        compact, captured = row['compact'], entry['compact']
        if not entry['matches_archive'] or any(captured[field] != compact[field]
                                               for field in ('text', 'finish', 'input_tokens', 'output_tokens', 'model_digest')):
            raise ValueError(f'Confidence archive differs from the paired compact response for {row["id"]}')
        compact['confidence'] = entry['confidence']
    return rows


def load_training(source):
    history = archived()
    extra, dev = (new_pairs(source, split) for split in ('train', 'dev'))
    check_disjoint(history, extra, dev, question_rows(source, 'test'))
    return history, extra, dev


def probability_features(values, parsed):
    if values is None:
        return [0.0] * CONFIDENCE_WIDTH
    evidence = np.array(values, dtype=float)
    if evidence.shape != (13,) or not np.isfinite(evidence).all() or (evidence < 0).any() or (evidence > 1.000001).any():
        raise ValueError('Invalid compact confidence features')
    if not parsed:
        evidence[:4] = 0
    positions = [1, 2, 4, 5, 7, 8, 10, 11]
    prob = np.clip(evidence[positions], 1e-7, 1 - 1e-7)
    log_odds = np.log(prob / (1 - prob)) / 4
    if not evidence[0]:
        log_odds[:2] = 0
    return [1.0, *evidence, *log_odds]


def visible_features(row):
    domain, question, first = row['domain'], row['question'], row['compact']
    candidate, support = recovery(domain, question, first)
    raw = question_features(domain, row['subject'], question)
    raw += compact_features(domain, question, first, candidate, support)
    raw += probability_features(first.get('confidence'), bool(first['answer']))
    return raw, candidate, support


def observe(rows):
    first = [row['compact'] for row in rows]
    if any(record['model_digest'] != MODEL_DIGEST for record in first):
        raise ValueError('Changed solver weights in a first response')
    costs = np.array([[record['input_tokens'], record['output_tokens']] for record in first])
    return Observed(tuple(row['id'] for row in rows), np.array([row['domain'] for row in rows]),
                    np.array([bool(r['answer']) for r in first]), costs.sum(axis=1), costs[:, 0],
                    np.array([visible_features(row)[0] for row in rows], dtype=float))


def comparison_features(row):
    domain, initial, retry = row['domain'], row['compact'], row['short']
    raw, candidate, support = visible_features(row)
    a, b = (canonical_answer(record['answer']).upper() if record['answer'] else '' for record in (initial, retry))
    evidence = retry['evidence']
    pa, pb = evidence.get(a, 0.0), evidence.get(b, 0.0)
    extra = [pa, pb, pb - pa, support.get(b, 0.0) - support.get(a, 0.0),
             bool(candidate.answer and canonical_answer(candidate.answer).upper() == b)]
    return raw + short_features(domain, retry) + extra


def verify_short_prompt_cost(row, retry):
    expected = row['compact']['input_tokens'] + SHORT_INPUT_OFFSETS[row['domain']]
    if retry['input_tokens'] != expected:
        raise ValueError(f'Unexpected short-prompt token count for {row["id"]} under the pinned tokenizer')


def paired(rows):
    first = observe(rows)
    disagreements, short_valid, comparison, short_tokens = [], [], [], []
    compact_correct, short_correct = [], []
    for row in rows:
        cur, other = row['compact'], row['short']
        verify_short_prompt_cost(row, other)
        short_valid.append(bool(other['answer']))
        disagreements.append(canonical_answer(cur['answer']) != canonical_answer(other['answer']))
        comparison.append(comparison_features(row))
        short_tokens.append(other['input_tokens'] + other['output_tokens'])
        compact_correct.append(correct(cur, row['answer']))
        short_correct.append(correct(other, row['answer']))
    return Paired(first, np.array(compact_correct), np.array(short_correct), np.array(short_valid),
                  np.array(disagreements), np.array(comparison, dtype=float), np.array(short_tokens))
