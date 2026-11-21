"""Summarize the complete registered mentor cohort, without selecting responses."""

import argparse
import json
from pathlib import Path
import time

from prepare_mixed_mentor import training_view, sha
from scoring import correct


def run(args):
    while not (args.mentor / 'complete.json').exists():
        time.sleep(30)
    rows, trees, judges, old_ids, contract = training_view(args)
    selected = {r['id'] for r in contract['selection']}
    rows = [r for r in rows if r['id'] in selected]
    groups = {'all': rows, 'new': [r for r in rows if r['id'] not in old_ids],
              'old': [r for r in rows if r['id'] in old_ids]}
    groups.update({'new_' + s: [r for r in groups['new'] if r['subject'] == s]
                   for s in ('math', 'chemistry', 'physics')})
    result = {}
    for label, group in groups.items():
        methods = {}
        for name, path in (('root', ''), ('repair_once', 'repair'), ('repair_twice', 'repair/repair'),
                           ('retry_once', 'retry'), ('retry_twice', 'retry/retry'),
                           ('strong_guided', 'guided'), ('retained_second_repair', 'guided/reuse'),
                           ('retained_after_retry', 'retry/reuse'), ('mentor_answer', None)):
            nodes = [judges[r['id'], ''] if path is None else trees[r['id']][path] for r in group]
            methods[name] = {'correct': sum(correct(n, r['answer']) for n, r in zip(nodes, group)),
                             'n': len(group),
                             'mean_node_tokens': sum(n['input_tokens'] + n['output_tokens'] for n in nodes) / len(group)}
        strong_wrong = sum(not correct(trees[r['id']]['guided'], r['answer']) for r in group)
        wrong_mentor_and_student = sum(not correct(judges[r['id'], ''], r['answer'])
                                      and not correct(trees[r['id']]['guided'], r['answer']) for r in group)
        result[label] = {'methods': methods, 'strong_guided_errors': strong_wrong,
                         'errors_with_wrong_mentor': wrong_mentor_and_student}
    output = {'scope': 'Training only; no policy result or fresh-data claim',
              'mentor_contract_sha256': sha(args.mentor / 'contract.json'),
              'mentor_bank_sha256': sha(args.mentor / 'train.jsonl'), 'groups': result,
              'tokens': 'Node-only diagnostic, not total trajectory or controller compute',
              'test_used': False}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old-data', 'new-data', 'mentor', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
