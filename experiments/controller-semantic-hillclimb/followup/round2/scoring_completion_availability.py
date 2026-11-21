"""Secondary preregistered TRAIN availability analysis of the same192 paid tokens.

This never changes the primary observation screen or submits a generated token.
Any future use as an action must consume a continuation and get matched controls.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import scoring_observation_pilot as pilot

SCHEMA = 'scoring-token-secondary-availability-v1'
LETTERS = 'ABCDEFGHIJ'


def offered_letters(row):
    if row.get('answer') != '' or row['domain'] != 'science':
        raise ValueError('Read offered options from the public science question only')
    body = row['question'].partition('\n\nOptions:\n')[2]
    labels = re.findall(r'(?m)^([A-J])\. ', body)
    if not 4 <= len(labels) <= 10 or labels != list(LETTERS[:len(labels)]):
        raise ValueError('Require the entire canonical option sequence; never drop a question')
    return labels


def hypothetical_answer(text, letters):
    answer = text.strip()
    return answer if len(answer) == 1 and answer in letters else ''


def specification(args):
    primary = pilot.read(args.output / 'registration.json')
    if primary != pilot.registration(args):
        raise ValueError('Primary scoring registration changed')
    rows, _, _, _, _ = pilot.public_inputs(args)
    return {'schema': SCHEMA, 'primary_registration_sha256': pilot.sha(args.output / 'registration.json'),
        'source_files': {**pilot.source_files(), str(Path(__file__).resolve()): pilot.sha(__file__)},
        'selection': [{'id': row['id'], 'subject': row['subject'], 'offered_letters': offered_letters(row)} for row in rows],
        'questions': 64, 'observation_paths': list(pilot.PATHS), 'hypothetical_completions': 192,
        'acceptance_rule': 'response.strip() must be exactly one uppercase offered option letter; everything else invalid',
        'case_conversion': False, 'extra_calls': 0, 'extra_tokens': 0, 'extra_retries': 0,
        'original_answers_changed': False, 'primary_feature_screen_changed': False,
        'comparison': 'All64 original seven-path answer union versus the same union plus all three hypothetical completions',
        'panels': ['all_science', 'chemistry', 'physics'], 'path_reports': list(pilot.PATHS),
        'validity_or_correctness_subset_selection': False, 'score_only_after_primary_full192_seal': True,
        'promotion': 'No controller promotion. Future completion is a NEW action charged one compact continuation; total1 after root or total2 after repair/retry, with matched controls and revised POMDP actions required.',
        'training_only': True, 'test_used': False, 'development_outcomes_used': False}


def register(args):
    if any((args.output / name).exists() for name in ('train.jsonl', 'raw-generations.jsonl', 'complete.json')):
        raise ValueError('Secondary hypothesis must be registered before the first primary scoring call')
    record = specification(args)
    target = args.output / 'secondary-availability'
    target.mkdir(exist_ok=False)
    pilot.write(target / 'registration.json', record)
    print({'secondary_registered': True, 'registration_sha256': pilot.sha(target/'registration.json'),
           'extra_calls': 0}, flush=True)


def aggregate(rows, trees, references, known, raw_records):
    expected = {(row['id'], path) for row in rows for path in pilot.PATHS}
    if len(rows) != 64 or Counter(row['subject'] for row in rows) != pilot.COUNTS or set(known) != expected:
        raise ValueError('Always analyze every fixed64 science question and all192 completions')
    raw = {item['request_sha256']: item for item in raw_records}
    if len(raw) != 192 or len(raw_records) != 192:
        raise ValueError('Require all192 unique physical scoring receipts')
    per_question, outcomes = [], []
    for row in rows:
        qid = row['id']; tree = trees[qid]
        offered = offered_letters(row); reference = references[qid]
        if reference not in offered:
            raise ValueError('TRAIN reference is not an offered option')
        old = {path: bool(tree[path]['answer'] == reference) for path in pilot.protocol.PLAIN}
        current = {}
        for path in pilot.PATHS:
            item = known[qid, path]; receipt = raw[item['request_sha256']]
            if receipt['response_sha256'] != item['raw_response_sha256']:
                raise ValueError('Secondary interpretation changed paid raw provenance')
            answer = hypothetical_answer(receipt['response']['response'], offered)
            correct = bool(answer and answer == reference)
            nodes = [tree['']] + ([tree[path]] if path else [])
            input_tokens = sum(node['input_tokens'] for node in nodes) + item['observer_input_tokens']
            output_tokens = sum(node['output_tokens'] for node in nodes) + item['observer_output_tokens']
            value = {'id': qid, 'subject': row['subject'], 'path': path, 'answer': answer,
                'valid': bool(answer), 'correct': correct, 'original_current_correct': old[path],
                'original_current_valid': tree[path]['answer'] in offered,
                'original_finish': tree[path].get('finish'),
                'prefix_marker_present': item['prefix']['marker_present'],
                'hypothetical_total_continuations': 1 if not path else 2,
                'full_hypothetical_trajectory_input_tokens': input_tokens,
                'full_hypothetical_trajectory_output_tokens': output_tokens,
                'full_hypothetical_trajectory_compute': 4*(input_tokens+output_tokens)/3.09,
                'request_sha256': item['request_sha256'], 'raw_response_sha256': item['raw_response_sha256']}
            outcomes.append(value); current[path] = value
        per_question.append({'id': qid, 'subject': row['subject'], 'plain': old, 'completion': current,
            'plain_oracle_correct': any(old.values()), 'completion_oracle_correct': any(v['correct'] for v in current.values()),
            'combined_oracle_correct': any(old.values()) or any(v['correct'] for v in current.values())})
    panels = {}
    for name in ('all_science', 'chemistry', 'physics'):
        selected = [q for q in per_question if name == 'all_science' or q['subject'] == name]
        panels[name] = {'questions': len(selected),
            'original_seven_path_correct': {p: sum(q['plain'][p] for q in selected) for p in pilot.protocol.PLAIN},
            'original_seven_path_oracle_correct': sum(q['plain_oracle_correct'] for q in selected),
            'hypothetical_three_completion_oracle_correct': sum(q['completion_oracle_correct'] for q in selected),
            'combined_ten_path_oracle_correct': sum(q['combined_oracle_correct'] for q in selected),
            'novel_correct_beyond_original_seven': sum(q['combined_oracle_correct'] and not q['plain_oracle_correct'] for q in selected),
            'paths': {}}
        for path in pilot.PATHS:
            values = [q['completion'][path] for q in selected]
            panels[name]['paths'][path or 'root'] = {'n': len(values),
                'valid': sum(v['valid'] for v in values), 'correct': sum(v['correct'] for v in values),
                'original_current_valid': sum(v['original_current_valid'] for v in values),
                'original_current_correct': sum(v['original_current_correct'] for v in values),
                'wins_vs_current': sum(v['correct'] and not v['original_current_correct'] for v in values),
                'losses_vs_current': sum(not v['correct'] and v['original_current_correct'] for v in values),
                'mean_full_hypothetical_trajectory_tokens': sum(v['full_hypothetical_trajectory_input_tokens']+v['full_hypothetical_trajectory_output_tokens'] for v in values)/len(values),
                'mean_full_hypothetical_trajectory_compute': sum(v['full_hypothetical_trajectory_compute'] for v in values)/len(values)}
    return panels, outcomes


def grade(args):
    # No reference labels or future branch outcomes are read until this full seal succeeds.
    rows, trees, known, complete = pilot.validated_complete(args)
    target = args.output / 'secondary-availability'
    if (target/'summary.json').exists():
        raise ValueError('Preserve the completed secondary analysis')
    if pilot.read(target/'registration.json') != specification(args):
        raise ValueError('Secondary preregistration changed')
    reference_rows = pilot.fixed.lines(args.data / 'splits/train-questions.jsonl')
    by_id = {row['id']: row for row in reference_rows}
    for row in rows:
        if pilot.protocol.public_row(by_id[row['id']]) != row:
            raise ValueError('TRAIN label row and sealed public prompt differ')
    references = {row['id']: by_id[row['id']]['answer'] for row in rows}
    panels, outcomes = aggregate(rows, trees, references, known,
        pilot.protocol.strict_records(args.output/'raw-generations.jsonl'))
    pilot.write(target/'private-outcomes.json', outcomes)
    pilot.write(target/'summary.json', {'schema': SCHEMA, 'panels': panels,
        'registration_sha256': pilot.sha(target/'registration.json'),
        'primary_complete_sha256': pilot.sha(args.output/'complete.json'),
        'private_outcomes_sha256': pilot.sha(target/'private-outcomes.json'),
        'extra_calls': 0, 'reused_paid_observation_calls': 192,
        'paid_observation_compute': sum(v['observer_compute'] for v in known.values()),
        'transport_error_records': complete['transport_error_records'],
        'acquisition_compute_is_lower_bound': complete['acquisition_compute_is_lower_bound'],
        'original_answers_changed': False, 'primary_screen_changed': False,
        'controller_result': False, 'training_only': True, 'test_used': False,
        'scope': 'Preregistered secondary hindsight answer-availability diagnostic only. No outcome-selected subset, policy promotion, or fresh-data claim.'})
    print(json.dumps({'secondary_complete': True, 'summary_sha256': pilot.sha(target/'summary.json'),
                     'panels': panels}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'grade'))
    for name in ('data', 'mentor', 'base', 'legacy-examples', 'baseline-diagnostic', 'refined-completion', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    {'register': register, 'grade': grade}[args.phase](args)
