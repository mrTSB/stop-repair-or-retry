"""Disjoint TRAIN examples for an explicitly supervised in-context review pilot.

The entire64-question pilot cohort is excluded before choosing demonstrations.
Only observed question text/subject choose neighbors; no current answer, solver
output or outcome affects retrieval. Correct example contents are TRAIN labels.
"""
import argparse
from collections import Counter
from pathlib import Path
import pickle
import re
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import fit_expanded1280_completion as source
import option_proposal_pairs as options

io, fixed = source.io, source.fixed
SCHEMA = 'disjoint1216-science-TRAIN-example-pool-v1'


def stem(row):
    return row['question'].removesuffix(options.SCIENCE_SUFFIX).partition('\n\nOptions:\n')[0].strip()


def grams(text):
    words = re.findall(r'\w+', text.casefold())
    return {tuple(words[i:i+5]) for i in range(max(0, len(words)-4))}


def similarity(left, right):
    a, b = grams(left), grams(right)
    return len(a & b)/max(1, len(a | b))


def examples(rows, excluded):
    result = []
    for row in rows:
        if row['id'] in excluded: continue
        if row['subject'] not in ('chemistry', 'physics') or row['answer'] not in options.LETTERS:
            raise ValueError('Only labeled science TRAIN examples')
        choices = options.options({**row, 'answer': ''})
        answer = options.LETTERS.index(row['answer'])
        if answer >= len(choices): raise ValueError('Reference is not an offered option')
        result.append({'id': row['id'], 'subject': row['subject'], 'stem': stem(row),
            'correct_answer_content': choices[answer]})
    if len({r['id'] for r in result}) != len(result) or {r['id'] for r in result} & set(excluded):
        raise ValueError('Duplicate or excluded example')
    return result


class Retriever:
    def __init__(self, directory):
        directory = Path(directory); done = io.read(directory/'complete.json')
        if done['schema'] != SCHEMA or done['development_or_test_read'] or done['model_calls']:
            raise ValueError('Require completed TRAIN-only pool')
        io.verify({str(directory/n): h for n, h in done['digests'].items()})
        self.manifest = io.read(directory/'manifest.json')
        self.rows = [io.json.loads(v) for v in (directory/'pool.jsonl').read_bytes().splitlines()]
        self.vectorizer, self.matrix = pickle.loads((directory/'index.pkl').read_bytes())
        self.ids = {r['id'] for r in self.rows}
        self._grams = [grams(r['stem']) for r in self.rows]

    def context(self, row):
        if row.get('answer') != '' or row['id'] in self.ids:
            raise ValueError('Current reference or current question cannot enter demonstration retrieval')
        text = stem(row); values = (self.matrix@self.vectorizer.transform([text]).T).toarray().ravel()
        query_grams = grams(text)
        overlap = [len(query_grams & value)/max(1, len(query_grams | value)) for value in self._grams]
        eligible = [i for i, candidate in enumerate(self.rows) if candidate['subject'] == row['subject']
            and overlap[i] < .3 and text.casefold() != candidate['stem'].casefold()]
        if not eligible: return '', None
        index = max(eligible, key=lambda i: (values[i], self.rows[i]['id']))
        if values[index] <= 0: return '', None
        example = self.rows[index]
        packet = ('SOLVED TRAINING EXAMPLE (different question; use relevant principles, never copy its option letter):\n'
            +'Question: '+' '.join(example['stem'].split()[:120])
            +'\nCorrect answer content: '+' '.join(example['correct_answer_content'].split()[:60]))
        return packet, {'example_id': example['id'], 'similarity': float(values[index]),
            'fivegram_jaccard': overlap[index], 'packet_sha256': fixed.fingerprint(packet)}


def build(args):
    spec = io.read(args.expanded_fit/'registration.json'); done = io.read(args.expanded_fit/'complete.json')
    rows, _, _, _, _, expected = source.inputs(SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()}))
    if expected != spec or done['registration_sha256'] != io.sha(args.expanded_fit/'registration.json'):
        raise ValueError('Changed sealed1280 TRAIN inputs')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.expanded_fit/n): h for n, h in done['digests'].items()})
    pilot = io.read(args.late/'registration.json'); pilot_done = io.read(args.late/'complete.json')
    if (pilot['schema'] != 'strong-late-review-timing-TRAIN-v1' or pilot['questions'] != 64
            or pilot_done['registration_sha256'] != io.sha(args.late/'registration.json') or not pilot_done['training_only']):
        raise ValueError('Require entire registered64 TRAIN cohort exclusion')
    io.verify(pilot['source_files']); io.verify(pilot['input_evidence'])
    io.verify({str(args.late/n): h for n, h in pilot_done['digests'].items()})
    excluded = {r['id'] for r in pilot['selection']}
    if len(excluded) != 64 or not excluded <= {r['id'] for r in rows}:
        raise ValueError('Exact pilot exclusion changed')
    pool = examples(rows, excluded)
    if len(pool) != 1216 or Counter(r['subject'] for r in pool) != {'chemistry': 608, 'physics': 608}:
        raise ValueError('Require all1216 disjoint examples without outcome filtering')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
    matrix = vectorizer.fit_transform([r['stem'] for r in pool])
    manifest = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': {**spec['input_evidence'], str((args.expanded_fit/'complete.json').resolve()): io.sha(args.expanded_fit/'complete.json'),
            str((args.late/'registration.json').resolve()): io.sha(args.late/'registration.json'),
            str((args.late/'complete.json').resolve()): io.sha(args.late/'complete.json')},
        'examples': 1216, 'subject_counts': {'chemistry': 608, 'physics': 608},
        'excluded_questions': 64, 'excluded_ids': sorted(excluded), 'all_pilot_questions_excluded': True,
        'source': 'Sealed1280 science TRAIN cohort; no DEV/TEST records',
        'labels': 'Correct offered option content only; no unverified generated rationale',
        'selection': 'One highest TFIDF similarity within observed subject, excluding equal stems and fivegram Jaccard>=0.3',
        'features': 'Question stem only, word unigrams/bigrams,50000 features,sublinear TF',
        'packet_limits': {'stem_words': 120, 'correct_answer_words': 60},
        'reference_answers_of_other_TRAIN_questions_in_prompt': True,
        'current_question_reference_in_prompt': False, 'development_or_test_read': False, 'model_calls': 0}
    args.output.mkdir(parents=True, exist_ok=False)
    io.jsonl(args.output/'pool.jsonl', pool); io.atomic_bytes(args.output/'index.pkl', pickle.dumps((vectorizer, matrix), protocol=5))
    io.write(args.output/'manifest.json', manifest)
    io.verify(manifest['source_files']); io.verify(manifest['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'examples': 1216,
        'model_calls': 0, 'development_or_test_read': False,
        'digests': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file()}})
    print({'examples': 1216, 'excluded_pilot_questions': 64, 'complete_sha256': io.sha(args.output/'complete.json')}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('expanded-fit', 'late', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    with fixed.training_only_reads(), fixed.guarded.no_model_calls(): build(args)


if __name__ == '__main__': main()
