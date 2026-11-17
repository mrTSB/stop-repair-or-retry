"""Freeze the next retry gate and evaluate it on disjoint questions."""

import argparse
import json
from pathlib import Path

import numpy as np

from arithmetic import correction
from study_data import LAST, index_responses, records, sha, validate_response
from experiment import WORK, evidence as prior_evidence, first_rows, record_sha, summary
from features import comparison_features, observe, verify_short_prompt_cost
from gate import Gate, accept_retry, dispatch
from hillclimb import new_run, paired, save
from margin_model import Learner, allocate, confidence_view, load_fit, pair_view, select
from scoring import canonical_answer, correct


PREVIOUS_GATE = LAST.parent / 'frozen.json'


def evidence(source):
    paths = [*Path(__file__).resolve().parent.glob('*.py'), *(source / 'splits').glob('*.json*')]
    result = {'code_or_split/' + path.name: sha(path) for path in sorted(paths)}
    result.update({'base/' + key: value for key, value in prior_evidence(LAST).items()})
    for name in ('dev-pairs', 'train-probes', 'prior-retries'):
        result[f'observations/{name}'] = record_sha(source, name)
    for name in ('test-first', 'test-retries'):
        result[f'earlier_test/{name}'] = record_sha(LAST, name)
    result['earlier_gate/frozen.json'] = sha(PREVIOUS_GATE)
    return result


def frozen(source, output):
    path = output / 'frozen.json'
    value = json.loads(path.read_bytes())
    if value['evidence'] != evidence(source):
        raise ValueError('Inputs differ from the freeze')
    return value, sha(path)


def tune(source, output):
    train, probes, weights, dev, earlier_dev = load_fit(source)
    previous = json.loads(PREVIOUS_GATE.read_bytes())['selected']
    result = select(train, probes, weights, dev, earlier_dev, previous)
    candidates = result.pop('candidates')
    with new_run(output) as stage:
        save(stage / 'frozen.json', {'evidence': evidence(source), 'train_n': len(train), 'probe_n': len(probes),
                                     'dev_n': len(dev), 'earlier_dev_n': len(earlier_dev), 'selected': result})
        save(stage / 'development-search.json', candidates)
    print(json.dumps({key: value for key, value in result.items() if key != 'model'}, indent=2))


def compile_plan(source, rows, locked, digest):
    rows = [{key: row[key] for key in ('id', 'domain', 'subject', 'question', 'compact')} for row in rows]
    state = observe(rows)
    previous = json.loads(PREVIOUS_GATE.read_bytes())['selected']
    prior = Gate.restore(previous['model'])
    current = locked['selected']
    gate = Learner.restore(current['model'])
    settings = current['settings']
    observations = confidence_view(rows)
    estimates = gate.predict(state, observations, settings['distilled_share'])
    verified = np.array([bool(correction(row)) for row in rows])
    selected = allocate(state, estimates, settings, verified)
    masks = {'retry_on_invalid': ~state.valid, 'previous_gate': dispatch(state, prior.predict(state), previous['settings']),
             'arithmetic_and_invalid': ~state.valid,
             'new_gate_without_arithmetic': allocate(state, estimates, settings), 'new_gate': selected}
    union = np.any(list(masks.values()), axis=0)
    return {'question_sha256': sha(source / 'splits/test-questions.jsonl'), 'first_sha256': record_sha(source, 'test-first'),
            'frozen_sha256': digest, 'requests': {name: [key for key, take in zip(state.ids, mask) if take]
                                                 for name, mask in masks.items()},
            'retry_union': [key for key, take in zip(state.ids, union) if take],
            'arithmetic_eligible': [key for key, take in zip(state.ids, verified) if take]}


def plan(source, output):
    locked, digest = frozen(source, output)
    payload = compile_plan(source, first_rows(source), locked, digest)
    path = output / 'test-plan.json'
    if path.exists() and json.loads(path.read_bytes()) != payload:
        raise ValueError('Existing plan differs from the frozen decision')
    save(path, payload)
    print(f'{len(payload["retry_union"])} distinct short calls needed')
    for name, ids in payload['requests'].items():
        print(name, len(ids))


def score(source, output):
    locked, digest = frozen(source, output)
    rows = first_rows(source)
    state = observe(rows)
    compiled = json.loads((output / 'test-plan.json').read_bytes())
    if compiled != compile_plan(source, rows, locked, digest):
        raise ValueError('Test plan differs from frozen policy')
    requested = set(compiled['retry_union'])
    replies = index_responses([row for row in rows if row['id'] in requested], records(source, 'test-retries'))
    initial = np.array([correct(row['compact'], row['answer']) for row in rows])
    candidates = [correction(row) for row in rows]
    verified = np.array([bool(answer) for answer in candidates])
    arithmetic_ok = np.array([correct({'answer': answer}, row['answer']) for answer, row in zip(candidates, rows)])
    short_ok, short_valid, disagree = (np.zeros(len(rows), dtype=bool) for _ in range(3))
    short_cost = np.zeros(len(rows), dtype=int)
    visited = {}
    for index, row in enumerate(rows):
        if row['id'] not in replies:
            continue
        retry = replies[row['id']]['short']
        validate_response(row, row['compact'], retry)
        verify_short_prompt_cost(row, retry)
        short_ok[index] = correct(retry, row['answer'])
        short_valid[index] = bool(retry['answer'])
        disagree[index] = canonical_answer(retry['answer']) != canonical_answer(row['compact']['answer'])
        short_cost[index] = retry['input_tokens'] + retry['output_tokens']
        visited[index] = {**row, 'short': retry}
    saved = locked['selected']
    new = Learner.restore(saved['model'])
    previous = Gate.restore(json.loads(PREVIOUS_GATE.read_bytes())['selected']['model'])
    results = {}
    for name, ids in ({'compact_only': []} | compiled['requests']).items():
        ids = set(ids)
        requests = np.array([key in ids for key in state.ids])
        probability, margin = np.zeros(len(rows)), 1
        if requests.any() and name in ('previous_gate', 'new_gate', 'new_gate_without_arithmetic'):
            seen = [visited[i] for i in np.flatnonzero(requests)]
            comparative = np.array([comparison_features(row) for row in seen])
            if name == 'previous_gate':
                probability[requests] = previous.comparison(comparative)
                margin = previous.margin
            else:
                probability[requests] = new.comparison(comparative, pair_view(seen), saved['settings']['distilled_share'])
                margin = new.base.margin
        accepted = requests & accept_retry(state.valid, short_valid, disagree, probability, margin)
        used_arithmetic = verified if name in ('new_gate', 'arithmetic_and_invalid') else np.zeros_like(verified)
        if (used_arithmetic & requests).any():
            raise ValueError('A verified first answer was scheduled for retry')
        grade = np.where(used_arithmetic, arithmetic_ok, np.where(accepted, short_ok, initial)).astype(int)
        results[name] = {'correct': grade, 'tokens': state.compact_tokens + requests * short_cost,
                         'query': requests, 'accepted': accepted, 'possible_fixes': requests & ~initial & short_ok}
        results[name]['arithmetic'] = used_arithmetic
    measurements = {name: {'correct': r['correct'], 'work': r['tokens'] * WORK} for name, r in results.items()}
    rng = np.random.default_rng(20261118)
    details = {name: summary(state, row, initial) for name, row in results.items()}
    for name, result in results.items():
        details[name]['all']['arithmetic_submissions'] = int(result['arithmetic'].sum())
        details[name]['all']['arithmetic_fixes'] = int((result['arithmetic'] & arithmetic_ok & ~initial).sum())
        details[name]['all']['arithmetic_regressions'] = int((result['arithmetic'] & ~arithmetic_ok & initial).sum())
    result = {'methods': details, 'paired_new_vs_controls': {name: paired(measurements['new_gate'], measurements[name], rng)
                                                             for name in ('retry_on_invalid', 'previous_gate', 'arithmetic_and_invalid',
                                                                          'new_gate_without_arithmetic')},
              'paired_gate_only_vs_invalid': paired(measurements['new_gate_without_arithmetic'], measurements['retry_on_invalid'], rng),
              'test_plan_sha256': sha(output / 'test-plan.json'), 'retry_responses_sha256': record_sha(source, 'test-retries')}
    with new_run(output / 'test') as stage:
        save(stage / 'summary.json', result)
        with (stage / 'per_question.jsonl').open('w', encoding='utf-8') as stream:
            for i, row in enumerate(rows):
                scores = {name: {'correct': int(r['correct'][i]), 'tokens': int(r['tokens'][i]),
                                 'retry': bool(r['query'][i]), 'submitted_retry': bool(r['accepted'][i]),
                                 'submitted_arithmetic': bool(r['arithmetic'][i])}
                          for name, r in results.items()}
                stream.write(json.dumps({'id': row['id'], 'domain': row['domain'], 'compact_correct': bool(initial[i]),
                                         'outcomes': scores}) + '\n')
    for name, results in details.items():
        s = results['all']
        print(f'{name:<24} {s["accuracy"]:>6.2%} {s["mean_solver_equivalent_tokens"]:>8.2f} equivalent tokens '
              f'{s["queries"]} retries, {s["fixes"]} fixes, {s["regressions"]} regressions')
    print(json.dumps(result['paired_new_vs_controls'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('tune', 'plan', 'score'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'tune': tune, 'plan': plan, 'score': score}[args.phase](args.data, args.output)


if __name__ == '__main__':
    main()
