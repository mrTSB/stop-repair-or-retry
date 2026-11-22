"""Explicit supervised TRAIN-example retrieval with frozen bounded review prompts.

Reference answer contents belong only to disjoint TRAIN demonstrations. No
current/pilot reference is in the index. Original ground-review source is left
intact; process-local injection changes only the declared evidence packet and
protocol identity for this separate registration.
"""
import argparse
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import sys
import time

import grounded_review_runner_v2 as predecessor
import training_example_pool as pool

original = predecessor.original
spec, io, protocol = original.spec, original.io, original.protocol
SCHEMA = 'supervised-disjoint-example1024-review-TRAIN-v1'
NAMES = ('Example1024 first guided', 'Example1024 retained second repair')
PATHS = ('example1024-guided', 'example1024-guided/reuse')
POOL = Path(__file__).resolve().parents[4]/'runs/controller-semantic-followup/round2/training-example-pool-v1'
EXTRAS = ('grounded1024-root-review-pilot-v2', 'advanced-grounded1024-root-review-pilot-v1', 'arithmetic1024-root-review-pilot-v1')
base_registration = original.registration


@lru_cache(maxsize=1)
def retriever(): return pool.Retriever(POOL)


@lru_cache(maxsize=128)
def retrieved(qid, question, subject):
    return retriever().context({'id': qid, 'question': question, 'subject': subject, 'answer': ''})


def context(qid, question, subject): return retrieved(qid, question, subject)[0]


def example_row(row):
    question = spec.public(row)
    packet = context(row['id'], row['question'], row['subject'])
    question += ('\n\nOPTIONAL SOLVED TRAINING EXAMPLE. It may be irrelevant to this question. '
        'Use only applicable facts or principles; answer the ORIGINAL QUESTION above, '
        'and do not copy a training example answer.\n'+(packet or 'No eligible training example.'))
    return {**row, 'question': question+protocol.old.SCIENCE_SUFFIX}


def require_priority(args):
    predecessor.require_priority(args)
    folder = args.data.parent/'arithmetic1024-root-review-pilot-v1'
    done = io.read(folder/'complete.json')
    io.verify({str(folder/n): h for n, h in done['digests'].items()})
    capacity = protocol.strict_records(folder/'capacity.jsonl')
    if not capacity or capacity[-1] != {'min_containers': 0, 'max_containers': 2}:
        raise ValueError('Arithmetic collector must finish and release cloud capacity')


def registration(args):
    require_priority(args)
    record = base_registration(args); done = io.read(POOL/'complete.json')
    manifest = io.read(POOL/'manifest.json')
    rows = spec.public_inputs(args)[0]
    if set(manifest['excluded_ids']) != {r['id'] for r in rows}:
        raise ValueError('Every pilot question must be excluded from the entire demonstration index')
    record['source_files'] = io.sources(sys.modules[__name__])
    record['input_evidence'].update({str((POOL/n).resolve()): h for n, h in done['digests'].items()})
    record['input_evidence'][str((POOL/'complete.json').resolve())] = io.sha(POOL/'complete.json')
    record['input_evidence'].update(manifest['input_evidence'])
    record.update(runner_schema=SCHEMA, evidence_kind='One disjoint supervised TRAIN example',
        example_pool_directory=str(POOL), example_pool_questions=1216, excluded_pilot_questions=64,
        references_supplied=True, current_question_reference_supplied=False,
        reference_scope='Only correct option CONTENT from disjoint TRAIN examples; no current/pilot labels',
        retrieved_examples=[{'id': r['id'], 'example': retrieved(r['id'], r['question'], r['subject'])[1]} for r in rows],
        full_trajectory=['original compact root', 'unchanged1024/512 bounded reviewer with disjoint TRAIN example',
            'unchanged128-token guided repair', 'unchanged128-token retained repair'],
        registered_known_union_questions=50, same_tool_and_examples_required_for_matched_B=True,
        original_unused_textbook_integrity_checks_retained=True)
    record['teacher'].update(phase1_visibility='Original question/current solution plus one disjoint labeled TRAIN example',
        phase2_visibility='Unchanged bounded continuation of paid thinking, only if no complete decision')
    record['questions_with_training_example'] = record.pop('questions_with_textbook_context')
    record['training_example_tokens_charged_in_teacher_input'] = record.pop('textbook_tokens_charged_in_teacher_input')
    return record


@contextmanager
def configured_runner():
    old = (spec.SCHEMA, spec.PATHS, spec.EXTRA_BANKS, spec.grounded_row, spec.context, spec.SCREEN,
        original.NAMES, original.registration, original.require_priority)
    spec.SCHEMA, spec.PATHS = SCHEMA, PATHS
    spec.EXTRA_BANKS = (*spec.EXTRA_BANKS, *EXTRAS)
    spec.grounded_row, spec.context = example_row, context
    spec.SCREEN = {**spec.SCREEN, 'scope': 'TRAIN headroom of disjoint labeled examples; matched B, learned policy and fresh validation required'}
    original.NAMES, original.registration, original.require_priority = NAMES, registration, require_priority
    try: yield
    finally:
        (spec.SCHEMA, spec.PATHS, spec.EXTRA_BANKS, spec.grounded_row, spec.context, spec.SCREEN,
            original.NAMES, original.registration, original.require_priority) = old
        retrieved.cache_clear(); retriever.cache_clear()


def grade(args):
    bundle, known, done = original.validated_complete(args)
    rows, trees, reviews, ordinary, ordinary_reviews, late, *_ = bundle
    target = args.output/'graded'
    if target.exists(): raise ValueError('Preserve the first completed diagnostic')
    labels = {r['id']: r for r in original.old.fixed.lines(args.data/'splits/train-questions.jsonl')}
    if any(protocol.public_row(labels[r['id']]) != r for r in rows): raise ValueError('Reference question changed')
    reports, private, screen = original.aggregate(rows, trees, reviews, ordinary, ordinary_reviews, late, known,
        {r['id']: labels[r['id']]['answer'] for r in rows}, spec.extra_solver_answers(args, rows), expected_known=50)
    timing = []
    for row in rows:
        started = time.perf_counter()
        for _ in range(10): packet, receipt = retriever().context(row)
        timing.append((time.perf_counter()-started)*100.)
        if (packet, receipt) != retrieved(row['id'], row['question'], row['subject']):
            raise ValueError('Retrieval replay changed')
    original.validated_complete(args); target.mkdir(); io.write(target/'private-outcomes.json', private)
    result = {'schema': SCHEMA, 'reports': reports, 'screen': screen, 'questions': 64, 'coverage': done['coverage'],
        'complete_sha256': io.sha(args.output/'complete.json'), 'private_outcomes_sha256': io.sha(target/'private-outcomes.json'),
        'training_only': True, 'test_used': False, 'new_neural_calls': 0, 'controller_accuracy_claim': False,
        'current_question_reference_supplied': False, 'disjoint_TRAIN_reference_contents_supplied': True,
        'matched_B_required_if_promoted': True, 'transport_error_records': done['transport_error_records'],
        'acquisition_compute_is_lower_bound': done['acquisition_compute_is_lower_bound'],
        'retrieval_CPU': {'mean_ms': sum(timing)/64, 'max_question_ms': max(timing),
            'repetitions_per_question': 10, 'index_setup_excluded': True, 'included_in_parameter_token_proxy': False}}
    io.write(target/'summary.json', result)
    print({'screen': screen, 'retrieval_CPU': result['retrieval_CPU'],
        'methods': {name: reports['all_science']['methods'][name] for name in NAMES}}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'collect', 'grade'))
    for name in ('data', 'mentor', 'complete', 'late', 'output', 'priority'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with predecessor.guarded.training_only_reads(), configured_runner():
        {'register': original.register, 'collect': original.collect, 'grade': grade}[args.phase](args)


if __name__ == '__main__': main()
