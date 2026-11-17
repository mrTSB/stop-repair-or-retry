"""Skip a tight-gate retry when its compact response yields an exact correction."""

import argparse
import json
from pathlib import Path

import numpy as np

from study import locked as tight_lock
from arithmetic import correction
from experiment import WORK, first_rows, record_sha, summary
from features import observe
from hillclimb import new_run, paired, save
from scoring import correct
from study_data import records, sha


def evidence(data, output):
    _, tight_digest = tight_lock(data, output.parent)
    return {'tight_gate_sha256': tight_digest, 'code_sha256': sha(Path(__file__))}


def tune(data, output):
    with new_run(output) as stage:
        save(stage / 'frozen.json', {'evidence': evidence(data, output),
                                     'rule': 'Submit an exact correction and skip any scheduled retry for that question.'})


def compile_plan(data, output, rows):
    locked = output / 'frozen.json'
    if json.loads(locked.read_bytes())['evidence'] != evidence(data, output):
        raise ValueError('The frozen arithmetic-check inputs changed')
    parent = output.parent / 'test-plan.json'
    tight = json.loads(parent.read_bytes())
    if tight['first_sha256'] != record_sha(data, 'test-first'):
        raise ValueError('The arithmetic check and tight gate have different first responses')
    checks = {}
    for row in rows:
        visible = {key: row[key] for key in ('domain', 'question', 'compact')}
        answer = correction(visible)
        if answer:
            checks[row['id']] = answer
    return {'frozen_sha256': sha(locked), 'tight_plan_sha256': sha(parent), 'corrections': checks,
            'requests': [key for key in tight['requests'] if key not in checks]}


def plan(data, output):
    payload = compile_plan(data, output, first_rows(data))
    path = output / 'test-plan.json'
    if path.exists() and json.loads(path.read_bytes()) != payload:
        raise ValueError('Arithmetic choices differ from the existing test plan')
    save(path, payload)
    print(f'{len(payload["requests"])} retries, {len(payload["corrections"])} arithmetic submissions, zero additional solver calls')


def score(data, output):
    rows = first_rows(data)
    state = observe(rows)
    payload = json.loads((output / 'test-plan.json').read_bytes())
    if payload != compile_plan(data, output, rows):
        raise ValueError('Arithmetic-check test plan changed')
    tight_path = output.parent / 'test/per_question.jsonl'
    main_path = output.parents[1] / 'test/per_question.jsonl'
    tight, main = ([json.loads(line) for line in source.read_text().splitlines()] for source in (tight_path, main_path))
    if any([item['id'] for item in sample] != list(state.ids) for sample in (tight, main)):
        raise ValueError('The paired controls cover different test questions')
    initial = np.array([correct(row['compact'], row['answer']) for row in rows])
    corrections = payload['corrections']
    checked = np.array([row['id'] in corrections for row in rows])
    exact = np.array([correct({'answer': corrections.get(row['id'], '')}, row['answer']) for row in rows])
    requested = set(payload['requests'])
    query = np.array([row['id'] in requested for row in rows])
    if requested != {entry['id'] for entry in tight if entry['retry'] and entry['id'] not in corrections}:
        raise ValueError('The arithmetic check changed a non-verified retry decision')
    shortlist = {row['id']: row['short'] for key in ('test-retries', 'tight-budget-retries') for row in records(data, key)}
    accept = np.array([item['submitted_retry'] for item in tight]) & query
    final = np.where(checked, exact, np.array([item['correct'] for item in tight])).astype(int)
    tokens = np.where(checked, state.compact_tokens, np.array([item['tokens'] for item in tight]))
    possible = np.array([key in shortlist and correct(shortlist[key], row['answer']) for key, row in zip(state.ids, rows)])
    report = summary(state, {'correct': final, 'tokens': tokens, 'query': query, 'accepted': accept,
                             'possible_fixes': query & ~initial & possible}, initial)
    report['all'].update(arithmetic_submissions=int(checked.sum()),
                         arithmetic_fixes=int((checked & exact & ~initial).sum()),
                         arithmetic_regressions=int((checked & ~exact & initial).sum()),
                         retries_skipped=int((checked & np.array([item['retry'] for item in tight])).sum()))
    outcomes = {'tight_budget_gate': tight}
    for name in ('retry_on_invalid', 'previous_gate', 'arithmetic_and_invalid', 'new_gate'):
        outcomes[name] = [item['outcomes'][name] for item in main]
    rng = np.random.default_rng(20261120)
    comparisons = {}
    for name, reference in outcomes.items():
        observations = {'correct': np.array([item['correct'] for item in reference]),
                        'work': np.array([item['tokens'] for item in reference]) * WORK}
        comparisons[name] = paired({'correct': final, 'work': tokens * WORK}, observations, rng)
    result = {'methods': {'tight_gate_and_arithmetic': report}, 'paired_vs_controls': comparisons,
              'test_plan_sha256': sha(output / 'test-plan.json'), 'tight_scores_sha256': sha(tight_path),
              'primary_scores_sha256': sha(main_path)}
    with new_run(output / 'test') as stage:
        save(stage / 'summary.json', result)
        with (stage / 'per_question.jsonl').open('w', encoding='utf-8') as stream:
            for i, row in enumerate(rows):
                stream.write(json.dumps({'id': row['id'], 'domain': row['domain'], 'correct': int(final[i]),
                                         'tokens': int(tokens[i]), 'retry': bool(query[i]), 'submitted_retry': bool(accept[i]),
                                         'submitted_arithmetic': bool(checked[i])}) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('tune', 'plan', 'score'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'tune': tune, 'plan': plan, 'score': score}[args.phase](args.data, args.output)


if __name__ == '__main__':
    main()
