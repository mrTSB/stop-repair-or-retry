"""Reserve512 additional math TRAIN questions using public fields only.

Question IDs/stems are projected from historical and protected records without
decoding answer values. No inference, correctness filtering or split changes.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

import expanded_mentor_io as io
import reserve
from auxiliary_second_paired_contract import project, fingerprint

SCHEMA = 'disjoint-math512-public-only-TRAIN-reservation-v1'
SEED = 2026112251
COUNT = 512


def projected(path, fields):
    with path.open('rb') as stream:
        return [project(raw, fields) for raw in stream if raw.strip()]


def public_source(path):
    import pyarrow as pa
    # Project the question column before conversion to Python objects.
    with pa.memory_map(str(path), 'r') as source:
        questions = pa.ipc.open_stream(source).read_all().select(['question']).column('question').to_pylist()
    return [{'id': f'gsm8k-train-{i:05d}', 'domain': 'math', 'subject': 'math',
             'question': text, 'answer': '', 'source': 'openai/gsm8k@e53f048856ff4f594e959d75785d2c2d37b678ee'}
            for i, text in enumerate(questions) if isinstance(text, str) and text.strip()]


def choose(candidates, ids, stems, index):
    chosen, exclusions = [], Counter()
    for row in sorted(candidates, key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).digest()):
        text = reserve.normalized(row['question'])
        stem = hashlib.sha256(reserve.previous.stem(row['question']).encode()).hexdigest()
        if row['id'] in ids or stem in stems or index.duplicate(text):
            exclusions['historical_reserved_or_near_duplicate'] += 1
            continue
        index.add(text); chosen.append(row)
    if len(chosen) < COUNT: raise ValueError('Insufficient disjoint math TRAIN pool')
    return chosen[:COUNT], dict(exclusions), len(chosen)-COUNT


def run(args):
    if args.output.exists(): raise ValueError('Preserve every previous reservation')
    source = reserve.previous.MATH
    selection = io.read(args.data/'splits/selection.json')
    if io.sha(source) != selection['math_sha256']: raise ValueError('Pinned math source changed')
    base = io.read(args.base_fit/'registration.json')
    legacy = Path(base['paths']['legacy_examples']); observer = Path(base['paths']['base'])
    # Bind the actual old encoder training set, not a later auxiliary-trained observer.
    import fit_completion_pomdp as fitted
    cached = fitted.cached.observer_from(observer)
    if cached._recipe['data_sha256'] != io.sha(legacy): raise ValueError('Changed old encoder training ancestry')
    ids, stems, index, files = set(), set(), reserve.DuplicateIndex(), {}
    protected_counts = {}
    for root in (io.PROJECT/'runs', io.PROJECT/'results'):
        for path in sorted(root.rglob('*-questions.jsonl')):
            rows = projected(path, {'id', 'question'}); files[str(path.resolve())] = io.sha(path)
            for row in rows:
                ids.add(row['id']); text = reserve.normalized(row['question'])
                stems.add(hashlib.sha256(reserve.previous.stem(row['question']).encode()).hexdigest())
                if text not in index.exact: index.add(text)
            if path.parent.resolve() == (args.data/'splits').resolve(): protected_counts[path.name] = len(rows)
        for path in sorted(root.rglob('historical-exclusions.json')):
            old = io.read(path); files[str(path.resolve())] = io.sha(path)
            ids.update(old.get('ids', [])); stems.update(old.get('normalized_stem_sha256', []))
    legacy_ids = {r['id'] for r in projected(legacy, {'id'})}; ids.update(legacy_ids)
    if protected_counts != {'train-questions.jsonl': 512, 'dev-questions.jsonl': 256, 'test-questions.jsonl': 1024}:
        raise ValueError('All three protected split memberships must be excluded')
    chosen, exclusions, unused = choose(public_source(source), ids, stems, index)
    if {r['id'] for r in chosen}&legacy_ids: raise ValueError('Selected question seen by frozen observer')
    files.update({str(p.resolve()): io.sha(p) for p in (source, legacy, args.base_fit/'registration.json',
        args.data/'splits/selection.json', observer/'summary.json')})
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'input_evidence': files, 'seed': SEED, 'questions': COUNT,
        'selection_rule': 'Fixed hash order; recorded IDs, normalized stems and85% comparable-length word-fivegram exclusion',
        'public_rows_sha256': fingerprint(chosen), 'math_source_sha256': io.sha(source),
        'legacy_encoder_overlap': 0, 'historical_or_reserved_overlap': 0,
        'protected_public_projection_counts': protected_counts, 'protected_reference_values_decoded': False,
        'source_question_column_only': True, 'correctness_filtering': False, 'model_calls': 0,
        'training_only': True, 'reserved_test_inferred_or_graded': False,
        'excluded_ids_sha256': fingerprint(sorted(ids)), 'excluded_stems_sha256': fingerprint(sorted(stems)),
        'exclusions': exclusions, 'unused_disjoint_math_after_reservation': unused,
        'future_collection': 'All seven unchanged plain solver paths; no review; all512 questions before label joining',
        'paths': {k: str(getattr(args, k).resolve()) for k in ('data', 'base_fit', 'output')}}
    io.verify(files); args.output.mkdir(parents=True)
    io.jsonl(args.output/'train-questions.jsonl', chosen)
    io.write(args.output/'registration.json', record)
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': io.sha(args.output/'registration.json'),
        'questions': COUNT, 'reference_values_decoded': False, 'model_calls': 0,
        'digests': {p.name: io.sha(p) for p in args.output.iterdir()}})
    print({'questions': COUNT, 'registration_sha256': io.sha(args.output/'registration.json'),
        'unused_disjoint_math': unused, 'legacy_encoder_overlap': 0, 'protected_reference_values_decoded': False}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'base-fit', 'output'): p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
