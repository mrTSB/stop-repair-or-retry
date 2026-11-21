"""Aggregate audited held-out failure causes without exporting example text.

This is a post-grading diagnostic, never a controller or a model-selection
input for the same panel. Hindsight counts do not claim achievable gains.
"""

import argparse
import json
from pathlib import Path

from confirmation_common import evidence, load_banks, read, sha, verify_fixed_prefix
from scoring import correct


def run(args):
    frozen = json.loads((args.study / 'frozen.json').read_bytes())
    audit = json.loads((args.study / 'test-audit.json').read_bytes())
    if not audit['passed'] or audit['freeze_sha256'] != sha(args.study / 'frozen.json'):
        raise ValueError('Audit all held-out outcomes first')
    if audit['scored_outcomes_sha256'] != sha(args.study / 'test/per-question.jsonl'):
        raise ValueError('Changed scored outcomes')
    if frozen['evidence'] != evidence(args.data):
        raise ValueError('Changed frozen inputs')
    verify_fixed_prefix(frozen, args.data)
    protocol = frozen['selected']['C']['protocol']
    reuse = Path(frozen['reuse_bank']['directory']) if 'reuse_bank' in frozen else None
    rows, trees, original, late, deeper, deep_reviews = load_banks(args.data, 'test', args.medium, protocol, reuse)
    by_id = {r['id']: r for r in rows}
    scored = read(args.study / 'test/per-question.jsonl')
    grouped = {family: {r['id']: r for r in scored if r['policy'] == family} for family in ('A', 'B', 'C')}
    panels = {'joint': set(by_id)}
    for row in scored:
        panels.setdefault(row['panel'], set()).add(row['id'])
    report = {}
    for panel, ids in panels.items():
        current = {'questions': len(ids), 'families': {}}
        for family in ('A', 'B', 'C'):
            config = frozen['selected'][family]
            special = config['protocol'] in ('medium', 'root-medium', 'mentor-reuse', 'full-packet')
            bank = deeper if special else trees
            reviews = deep_reviews if special else late if config['protocol'] == 'late' else original
            counts = {'errors': 0, 'errors_without_a_paid_review': 0,
                      'errors_with_a_correct_paid_review': 0,
                      'errors_with_paid_reviews_all_wrong': 0,
                      'errors_with_a_correct_earlier_visited_answer': 0,
                      'errors_where_initial_answer_was_correct': 0}
            for qid in ids:
                item = grouped[family][qid]
                if item['correct']:
                    continue
                row = by_id[qid]
                parts = item['path'].split('/') if item['path'] else []
                visited = [''] + ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
                paid = [reviews[qid, d['path']] for d in item['decisions'] if d['queried']]
                any_good_review = any(correct(n, row['answer']) for n in paid)
                counts['errors'] += 1
                counts['errors_without_a_paid_review'] += int(not paid)
                counts['errors_with_a_correct_paid_review'] += int(any_good_review)
                counts['errors_with_paid_reviews_all_wrong'] += int(bool(paid) and not any_good_review)
                counts['errors_with_a_correct_earlier_visited_answer'] += int(any(correct(bank[qid][p], row['answer']) for p in visited[:-1]))
                counts['errors_where_initial_answer_was_correct'] += int(correct(bank[qid][''], row['answer']))
            assert counts['errors'] == sum(counts[k] for k in ('errors_without_a_paid_review',
                                                               'errors_with_a_correct_paid_review',
                                                               'errors_with_paid_reviews_all_wrong'))
            current['families'][family] = counts
        current['C_vs_B'] = {'C_only_correct': sum(grouped['C'][q]['correct'] and not grouped['B'][q]['correct'] for q in ids),
                             'B_only_correct': sum(grouped['B'][q]['correct'] and not grouped['C'][q]['correct'] for q in ids),
                             'both_correct': sum(grouped['B'][q]['correct'] and grouped['C'][q]['correct'] for q in ids),
                             'both_wrong': sum(not grouped['B'][q]['correct'] and not grouped['C'][q]['correct'] for q in ids)}
        report[panel] = current
    payload = {'freeze_sha256': sha(args.study / 'frozen.json'),
               'scored_outcomes_sha256': sha(args.study / 'test/per-question.jsonl'),
               'source_sha256': sha(__file__), 'raw_examples_present': False,
               'interpretation': 'Post-grading hindsight diagnosis only. A correct earlier answer or review is not an achievable policy gain estimate. The three review-error categories partition errors; the other flags overlap.',
               'panels': report}
    destination = args.study / 'error-analysis.json'
    if destination.exists():
        raise ValueError('Do not overwrite a diagnostic receipt')
    destination.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'data', 'medium'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
