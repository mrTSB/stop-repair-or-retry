"""Register, collect and grade a separately declared numerical-tool review pilot."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time

import grounded_review_runner_v2 as predecessor
import arithmetic_review_protocol as spec

original = predecessor.original
io, protocol = original.io, original.protocol
SCHEMA = 'arithmetic-assisted-review-runner-v1'
NAMES = ('Arithmetic1024 first guided', 'Arithmetic1024 retained second repair')
base_registration = original.registration


def require_priority(args):
    predecessor.require_priority(args)
    folder = args.data.parent/'advanced-grounded1024-root-review-pilot-v1'
    done = io.read(folder/'complete.json')
    io.verify({str(folder/n): h for n, h in done['digests'].items()})
    capacity = protocol.strict_records(folder/'capacity.jsonl')
    if not capacity or capacity[-1] != {'min_containers': 0, 'max_containers': 2}:
        raise ValueError('The preceding cloud collector must finish and release capacity')


def registration(args):
    require_priority(args)
    record = base_registration(args)
    record['source_files'] = io.sources(sys.modules[__name__])
    record.update(runner_schema=SCHEMA, minimum_physical_teacher_calls=128,
        full_trajectory=['original compact root', '1024-token reasoning with optional numeric expressions',
            'bounded deterministic arithmetic', '512-token review decision with paid tool output',
            '128-token guided compact repair', '128-token retained compact repair'],
        registered_known_union_questions=50,
        calculator={'new_review_capability': True, 'reference_access': False, 'network_access': False,
            'arbitrary_code_execution': False, 'maximum_expressions': 8, 'maximum_expression_characters': 256,
            'maximum_AST_nodes': 96, 'maximum_absolute_exponent': 100, 'maximum_absolute_value': 1e100,
            'functions': sorted(spec.calculator.FUNCTIONS), 'constants': sorted(spec.calculator.CONSTANTS),
            'real_floating_point_arithmetic_only': True, 'results_and_errors_preserved': True,
            'tool_output_tokens_charged_in_teacher_input': True,
            'compact_receives_unchanged_answer_and_check_packet': True,
            'CPU_runtime_reported_separately_from_parameter_token_proxy': True,
            'matched_B_must_receive_identical_tool_availability': True},
        promotion_requires='Novel correct paths, matched arithmetic-capable B, learned POMDP integration and fresh validation')
    record['teacher'].update(always_two_phases=True,
        phase1_visibility='Original question and current solution; request bounded numerical expressions when relevant',
        phase2_visibility='Original question, current solution, entire paid reasoning and deterministic numerical results')
    for key in ('retrieval_neural_calls', 'retrieval_CPU_not_in_parameter_token_proxy',
                'questions_with_textbook_context', 'textbook_tokens_charged_in_teacher_input'):
        record.pop(key)
    return record


@contextmanager
def configured_runner():
    old = original.spec, original.NAMES, original.registration, original.require_priority
    original.spec, original.NAMES = spec, NAMES
    original.registration, original.require_priority = registration, require_priority
    try: yield
    finally:
        original.spec, original.NAMES, original.registration, original.require_priority = old


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
    tool = [known[r['id'], 'judge', '']['response'] for r in rows]
    benchmark = []
    for row in tool:
        text = spec.paid_reasoning(row['calls'][0]); started = time.perf_counter()
        for _ in range(100): result = spec.calculator.evaluate_text(text)
        benchmark.append((time.perf_counter()-started)*10.)
        if result != row['calculator_results']: raise ValueError('Calculator replay differs from acquired results')
    original.validated_complete(args); target.mkdir()
    io.write(target/'private-outcomes.json', private)
    result = {'schema': spec.SCHEMA, 'runner_schema': SCHEMA, 'reports': reports, 'screen': screen,
        'questions': 64, 'coverage': done['coverage'], 'complete_sha256': io.sha(args.output/'complete.json'),
        'private_outcomes_sha256': io.sha(target/'private-outcomes.json'),
        'training_only': True, 'test_used': False, 'new_neural_calls': 0, 'controller_accuracy_claim': False,
        'matched_B_required_if_promoted': True, 'new_tool_capability_explicit': True,
        'transport_error_records': done['transport_error_records'],
        'acquisition_compute_is_lower_bound': done['acquisition_compute_is_lower_bound'],
        'calculator': {'questions_requesting_expressions': sum(bool(v['calculator_results']) for v in tool),
            'expressions': sum(v['calculator_expression_count'] for v in tool),
            'valid_expressions': sum(v['ok'] for row in tool for v in row['calculator_results']),
            'mean_replay_cpu_ms': sum(benchmark)/64, 'max_question_replay_cpu_ms': max(benchmark),
            'repetitions_per_question': 100, 'included_in_parameter_token_proxy': False}}
    io.write(target/'summary.json', result)
    print({'screen': screen, 'calculator': result['calculator'],
        'methods': {name: reports['all_science']['methods'][name] for name in NAMES}}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'collect', 'grade'))
    for name in ('data', 'mentor', 'complete', 'late', 'output', 'priority'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256'); args = parser.parse_args()
    with predecessor.guarded.training_only_reads(), configured_runner():
        {'register': original.register, 'collect': original.collect, 'grade': grade}[args.phase](args)


if __name__ == '__main__': main()
