"""Register, collect, and grade a reference-blind contrastive review pilot."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

import evidence_first_review_runner as infrastructure
import contrastive_review_protocol as spec

original = infrastructure.original
io, protocol = original.io, original.protocol
SCHEMA = 'contrastive-three-phase-review-runner-v1'
NAMES = ('Contrastive first guided', 'Contrastive retained second repair')


def require_priority(args):
    infrastructure.predecessor.require_priority(args)
    folder = args.data.parent/'broad-evidence-first-textbook-root-review-pilot-v4'
    done = io.read(folder/'complete.json')
    io.verify({str(folder/n): h for n, h in done['digests'].items()})
    capacity = protocol.strict_records(folder/'capacity.jsonl')
    if not capacity or capacity[-1] != {'min_containers': 0, 'max_containers': 2}:
        raise ValueError('Prior cloud collector must finish and release capacity')


def registration(args):
    require_priority(args)
    record = infrastructure.base_registration(args)
    record['source_files'] = io.sources(sys.modules[__name__])
    record.update(runner_schema=SCHEMA, minimum_physical_teacher_calls=192, physical_teacher_calls=192,
        maximum_total_physical_calls=320, registered_known_union_questions=50,
        full_trajectory=['original compact root', '640-token question-only option audit',
            '384-token counterexample check with paid audit and root', '512-token decision with both paid analyses',
            '128-token compact guided repair', '128-token compact retained repair'],
        matched_B_must_receive_identical_review_capability=True,
        promotion_requires='At least6 novel TRAIN paths and cost screen; then learned POMDP integration, strongest matched B and fresh validation',
        final_C_target={'accuracy_strictly_greater_than': .80, 'gap_above_best_A_B_at_least': .10, 'compute_below_best_B': True},
        option_handling='All original choices, order and letters preserved; no selection using labels or outcomes')
    record['teacher'].update(always_two_phases=False, always_three_phases=True,
        thinking_cap=1024, audit_cap=640, counterexample_cap=384, total_output_cap=1536,
        phase1_visibility='Public original question only; no current answer',
        phase2_visibility='Original question, paid option audit, paid original root solution',
        phase3_visibility='Original question, root solution, both paid analyses')
    for key in ('retrieval_neural_calls', 'retrieval_CPU_not_in_parameter_token_proxy',
                'questions_with_textbook_context', 'textbook_tokens_charged_in_teacher_input'):
        record.pop(key)
    return record


@contextmanager
def configured_runner():
    before = (original.spec, original.NAMES, original.registration, original.require_priority,
              original.Journal, original.attempts_checked, infrastructure.spec)
    original.spec, original.NAMES = spec, NAMES
    original.registration, original.require_priority = registration, require_priority
    original.Journal, original.attempts_checked = infrastructure.Journal, infrastructure.attempts_checked
    infrastructure.spec = spec
    try:
        yield
    finally:
        (original.spec, original.NAMES, original.registration, original.require_priority,
         original.Journal, original.attempts_checked, infrastructure.spec) = before


def register(args):
    record = registration(args)
    args.output.mkdir(parents=True, exist_ok=False)
    io.write(args.output/'registration.json', record)
    print({'registration_sha256': io.sha(args.output/'registration.json'), 'maximum_calls': 320, 'calls_started': 0}, flush=True)


def grade(args):
    bundle, known, done = original.validated_complete(args)
    rows, trees, reviews, ordinary, ordinary_reviews, late, *_ = bundle
    target = args.output/'graded'
    if target.exists(): raise ValueError('Preserve the first completed grading result')
    labels = {r['id']: r for r in original.old.fixed.lines(args.data/'splits/train-questions.jsonl')}
    if any(protocol.public_row(labels[r['id']]) != r for r in rows):
        raise ValueError('Reference question differs from the completed paid question')
    reports, private, screen = original.aggregate(rows, trees, reviews, ordinary, ordinary_reviews, late, known,
        {r['id']: labels[r['id']]['answer'] for r in rows}, spec.extra_solver_answers(args, rows), expected_known=50)
    original.validated_complete(args)
    target.mkdir()
    io.write(target/'private-outcomes.json', private)
    result = {'schema': spec.SCHEMA, 'runner_schema': SCHEMA, 'reports': reports, 'screen': screen,
        'questions': 64, 'coverage': done['coverage'], 'complete_sha256': io.sha(args.output/'complete.json'),
        'private_outcomes_sha256': io.sha(target/'private-outcomes.json'),
        'training_only': True, 'test_used': False, 'development_used': False,
        'new_neural_calls': 0, 'controller_accuracy_claim': False, 'matched_B_required_if_promoted': True,
        'transport_error_records': done['transport_error_records'],
        'acquisition_compute_is_lower_bound': done['acquisition_compute_is_lower_bound']}
    io.write(target/'summary.json', result)
    print({'screen': screen, 'methods': {name: reports['all_science']['methods'][name] for name in NAMES}}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'collect', 'grade'))
    for name in ('data', 'mentor', 'complete', 'late', 'output', 'priority'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    with infrastructure.predecessor.guarded.training_only_reads(), configured_runner():
        {'register': register, 'collect': original.collect, 'grade': grade}[args.phase](args)


if __name__ == '__main__':
    main()
