"""Prespecified active-retrieval pilot with the same frozen reviewer and solver."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time

import grounded_review_runner_v2 as predecessor
import active_textbook_review_protocol_v2 as spec

original = predecessor.original
io, protocol = original.io, original.protocol
SCHEMA = 'active-textbook-review-runner-v2'
NAMES = ('Active textbook V2 first guided', 'Active textbook V2 retained second repair')
base_registration = original.registration


def require_priority(args):
    predecessor.require_priority(args)
    for name in ('arithmetic1024-root-review-pilot-v1', 'example1024-root-review-pilot-v1', 'active-textbook1024-root-review-pilot-v1'):
        folder = args.data.parent/name; done = io.read(folder/'complete.json')
        io.verify({str(folder/n): h for n, h in done['digests'].items()})
        capacity = protocol.strict_records(folder/'capacity.jsonl')
        if not capacity or capacity[-1] != {'min_containers': 0, 'max_containers': 2}:
            raise ValueError('Every prior cloud collector must finish and release capacity')


def registration(args):
    require_priority(args)
    corpus = spec.retrieval.CORPUS; completed = io.read(corpus/'complete.json')
    if completed['benchmark_records_read'] or completed['model_calls']:
        raise ValueError('Require independently built textbook exposition')
    io.verify({str(corpus/n): h for n, h in completed['digests'].items()})
    record = base_registration(args)
    record['source_files'] = io.sources(sys.modules[__name__])
    record['input_evidence'].update({str((corpus/n).resolve()): h for n, h in completed['digests'].items()})
    record['input_evidence'][str((corpus/'complete.json').resolve())] = io.sha(corpus/'complete.json')
    record.update(runner_schema=SCHEMA, minimum_physical_teacher_calls=128,
        full_trajectory=['original compact root', '1024-token reasoning starting with up to2 search queries',
            'deterministic local textbook retrieval', '512-token review decision with paid reasoning and retrieved passages',
            '128-token guided compact repair', '128-token retained compact repair'],
        registered_known_union_questions=50, corpus_directory=str(corpus),
        active_retrieval={'query_source': 'Already-paid current reasoning, not labels or future responses',
            'maximum_queries': 2, 'prefilled_opening_search_tag': True, 'literal_placeholders_rejected': True,
            'fallback': 'Public original question stem, capped800 characters, if no valid completed query', 'maximum_generated_query_characters': 200, 'maximum_unique_passages': 3,
            'maximum_words_per_passage': 85, 'minimum_similarity': .06,
            'subject_filter': True, 'model_or_network_calls': 0,
            'question_or_outcome_conditioned_corpus_build': False, 'benchmark_reference_access': False,
            'all_tool_output_tokens_charged': True, 'CPU_time_reported_separately': True,
            'compact_receives_unchanged_answer_and_check_packet': True,
            'matched_B_must_receive_identical_tool_availability': True},
        promotion_requires='Novel correct paths, matched retrieval-capable B, learned POMDP integration and fresh validation')
    record['teacher'].update(always_two_phases=True,
        phase1_visibility='Original question and paid current solution; request concept queries before reasoning',
        phase2_visibility='Original question, paid current solution, all paid reasoning, bounded retrieved textbook excerpts')
    for key in ('retrieval_neural_calls', 'retrieval_CPU_not_in_parameter_token_proxy',
                'questions_with_textbook_context', 'textbook_tokens_charged_in_teacher_input'):
        record.pop(key)
    return record


@contextmanager
def configured_runner():
    before = original.spec, original.NAMES, original.registration, original.require_priority
    original.spec, original.NAMES = spec, NAMES
    original.registration, original.require_priority = registration, require_priority
    try: yield
    finally:
        original.spec, original.NAMES, original.registration, original.require_priority = before


def grade(args):
    bundle, known, done = original.validated_complete(args)
    rows, trees, reviews, ordinary, ordinary_reviews, late, *_ = bundle
    target = args.output/'graded'
    if target.exists(): raise ValueError('Preserve the first completed diagnostic')
    labels = {r['id']: r for r in original.old.fixed.lines(args.data/'splits/train-questions.jsonl')}
    if any(protocol.public_row(labels[r['id']]) != r for r in rows):
        raise ValueError('Reference question differs from paid prompt')
    reports, private, screen = original.aggregate(rows, trees, reviews, ordinary, ordinary_reviews, late, known,
        {r['id']: labels[r['id']]['answer'] for r in rows}, spec.extra_solver_answers(args, rows), expected_known=50)
    tool = [known[r['id'], 'judge', '']['response'] for r in rows]; timing = []
    for row, review in zip(rows, tool):
        text = '<search>'+spec.paid_reasoning(review['calls'][0]); started = time.perf_counter()
        for _ in range(10): result = spec.retrieval.evaluate_text(text, row['subject'], spec.public(row))
        timing.append((time.perf_counter()-started)*100.)
        if result != review['retrieval_results']: raise ValueError('Retrieval replay differs from acquired evidence')
    original.validated_complete(args); target.mkdir(); io.write(target/'private-outcomes.json', private)
    result = {'schema': spec.SCHEMA, 'runner_schema': SCHEMA, 'reports': reports, 'screen': screen,
        'questions': 64, 'coverage': done['coverage'], 'complete_sha256': io.sha(args.output/'complete.json'),
        'private_outcomes_sha256': io.sha(target/'private-outcomes.json'),
        'training_only': True, 'test_used': False, 'new_neural_calls': 0, 'controller_accuracy_claim': False,
        'matched_B_required_if_promoted': True, 'transport_error_records': done['transport_error_records'],
        'acquisition_compute_is_lower_bound': done['acquisition_compute_is_lower_bound'],
        'active_retrieval': {'questions_with_queries': sum(bool(v['retrieval_results']) for v in tool),
            'queries': sum(v['retrieval_query_count'] for v in tool),
            'public_question_fallbacks': sum(v['public_question_fallback'] for v in tool),
            'questions_with_passages': sum(any(r['passages'] for r in v['retrieval_results']) for v in tool),
            'passages': sum(len(r['passages']) for v in tool for r in v['retrieval_results']),
            'mean_replay_cpu_ms': sum(timing)/64, 'max_question_replay_cpu_ms': max(timing),
            'repetitions_per_question': 10, 'included_in_parameter_token_proxy': False}}
    io.write(target/'summary.json', result)
    print({'screen': screen, 'active_retrieval': result['active_retrieval'],
        'methods': {name: reports['all_science']['methods'][name] for name in NAMES}}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=('register', 'collect', 'grade'))
    for name in ('data', 'mentor', 'complete', 'late', 'output', 'priority'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with predecessor.guarded.training_only_reads(), configured_runner():
        {'register': original.register, 'collect': original.collect, 'grade': grade}[args.phase](args)


if __name__ == '__main__': main()
