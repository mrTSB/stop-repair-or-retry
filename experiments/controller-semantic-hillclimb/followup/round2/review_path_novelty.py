"""Training-only attainable-path novelty of the completed independent pilot.

The union uses references for an oracle diagnostic, never for an online policy.
All64 registered science questions remain in the denominator.
"""

import argparse
import json
from pathlib import Path

import independent_review_pilot as pilot
from banks import committed
from difficulty_pomdp import PLAIN
from scoring import correct
from prepare_mixed_mentor import sha, fingerprint


def calculate(rows, trees, strong, earlier, new):
    paths = ('guided', 'guided/reuse', 'retry/reuse')
    results = []
    for row in rows:
        qid = row['id']
        plain = [bool(correct(trees[qid][path], row['answer'])) for path in PLAIN]
        retained = [bool(correct(strong[qid, 'solver', path], row['answer'])) for path in paths]
        critical = bool(correct(earlier[qid, 'solver'], row['answer']))
        independent = bool(correct(new[qid, 'solver'], row['answer']))
        before = any(plain + retained + [critical])
        results.append({'id': qid, 'subject': row['subject'], 'plain_union': any(plain),
            'plain_and_retained_union': any(plain + retained), 'before_union': before,
            'independent_correct': independent, 'after_union': before or independent,
            'novel_success': independent and not before,
            'independent_replaces_prior_success_with_error': before and not independent})
    def aggregate(selected):
        names = ('plain_union', 'plain_and_retained_union', 'before_union', 'independent_correct',
                 'after_union', 'novel_success', 'independent_replaces_prior_success_with_error')
        return {'questions': len(selected), **{name: sum(row[name] for row in selected) for name in names}}
    return {'all': aggregate(results), 'subjects': {subject: aggregate([r for r in results if r['subject'] == subject])
             for subject in ('chemistry', 'physics')}, 'private_rows': results}


def run(args):
    if args.output.exists():
        raise ValueError('Use a new novelty receipt; never overwrite results')
    _, rows, trees, strong, earlier, hashes = pilot.controls(args)
    summary = json.loads((args.independent / 'summary.json').read_bytes())
    if (summary['questions'], summary['test_used'], summary['bank_sha256'], summary['contract_sha256']) != (
            64, False, sha(args.independent / 'train.jsonl'), sha(args.independent / 'contract.json')):
        raise ValueError('Complete and preserve every registered independent pair')
    raw = committed(args.independent / 'train.jsonl')
    new = {(r['id'], r['kind']): r['response'] for r in raw}
    if len(raw) != 128 or set(new) != {(r['id'], kind) for r in rows for kind in ('judge', 'solver')}:
        raise ValueError('Incomplete or duplicated independent grid')
    for row in rows:
        safe = {**row, 'answer': ''}; current = strong[row['id'], 'solver', 'guided']
        judge = new[row['id'], 'judge']
        pilot.verify_review(safe, current, judge)
        pilot.verify_solver(safe, current, judge, new[row['id'], 'solver'])
    result = calculate(rows, trees, strong, earlier, new)
    result.update(source_sha256=sha(__file__), independent_summary_sha256=sha(args.independent / 'summary.json'),
        independent_bank_sha256=summary['bank_sha256'], paired_controls_sha256=fingerprint(hashes),
        prior_plain_paths=list(PLAIN), prior_retained_paths=['guided', 'guided/reuse', 'retry/reuse'],
        pure_guided_retry='Already represented by its identical plain repair/retry prompt and seed',
        interpretation='Reference-based training oracle over all allowed paths; not attainable controller accuracy.',
        development_or_test_used=False, neural_calls=0, cloud_calls=0)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'private_rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'mentor', 'critical', 'independent', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
