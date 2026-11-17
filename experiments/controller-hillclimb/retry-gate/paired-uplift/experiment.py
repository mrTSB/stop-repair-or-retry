"""Freeze a gate, plan unseen retries from first responses, and score paired decisions."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from data import DATA, MODEL, MODEL_DIGEST, PARENT, PREVIOUS, sha
from compact import parse_answer, prompt
from evaluate import PUBLISHED, validate_response
from features import comparison_features, index_responses, load_training, observe, paired as supervised, question_rows, records, verify_short_prompt_cost
from gate import Gate, accept_retry, dispatch, fit_and_select
from hillclimb import new_run, paired, save
from planner import SOLVER_PARAMETERS_B
from scoring import canonical_answer, correct


WORK = 4 / SOLVER_PARAMETERS_B


def record_sha(source, name):
    path = source / f'{name}.jsonl'
    payload = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix('.jsonl.gz').read_bytes())
    return hashlib.sha256(payload).hexdigest()


def evidence(source):
    here = Path(__file__).resolve().parent
    files = {
        'code': [*here.glob('*.py'), *(PARENT / name for name in ('collect_pairs.py', 'evaluate.py', 'signals.py')),
                 PARENT / 'token-confidence/probe.py', *(PARENT.parent / name for name in ('compact.py', 'collect.py', 'planner.py', 'hillclimb.py')),
                 DATA.parent / 'scoring.py'],
        'fresh/questions': list((source / 'splits').glob('*.json*')),
        'published/questions': [DATA / f'{split}-questions.jsonl' for split in ('train', 'dev', 'test')],
        'published/responses': list(PUBLISHED.glob('*-4b-*.jsonl.gz')),
        'prior/questions': list((PREVIOUS / 'splits').glob('*.json*')),
        'prior/responses': list(PREVIOUS.glob('*-pairs.jsonl.gz')),
        'prior/confidence': list((PREVIOUS / 'pilots/confidence').glob('*.jsonl.gz')),
    }
    sums = {f'{group}/{path.name}': sha(path) for group, paths in files.items() for path in sorted(paths)}
    return sums | {f'fresh/{split}-pairs.jsonl': record_sha(source, f'{split}-pairs') for split in ('train', 'dev')}


def load_frozen(source, output):
    path = output / 'frozen.json'
    frozen = json.loads(path.read_bytes())
    if frozen['evidence'] != evidence(source):
        raise ValueError('Frozen source, training inputs, or test question split changed')
    return frozen, sha(path)


def tune(source, output):
    history, extra, validation = load_training(source)
    training = supervised(history + extra)
    result = fit_and_select(training, supervised(validation))
    payload = {'evidence': evidence(source), 'archived_train_pairs': len(history), 'new_train_pairs': len(extra),
               'fresh_development_pairs': len(validation), 'selected': result['selected'],
               'without_token_confidence': result['without_token_confidence'],
               'training_confidence_count': sum('confidence' in r['compact'] for r in history + extra)}
    with new_run(output) as pending:
        save(pending / 'frozen.json', payload)
    for name in ('selected', 'without_token_confidence'):
        selected = result[name]
        print(name, selected['model']['kind'], json.dumps({k: v for k, v in selected.items() if k != 'model'}))


def first_rows(source):
    questions = question_rows(source, 'test')
    records_by_id = index_responses(questions, records(source, 'test-first'))
    result = []
    for question in questions:
        first = records_by_id[question['id']]['compact']
        formatted = prompt(question['domain'], question['question'])
        if (first['model'] != MODEL or first['model_digest'] != MODEL_DIGEST
                or first['prompt_sha256'] != hashlib.sha256(formatted.encode()).hexdigest()
                or first['answer'] != parse_answer(question['domain'], first['text'])
                or first['input_tokens'] <= 0 or not 0 <= first['output_tokens'] <= 128):
            raise ValueError(f'First response does not match the fixed protocol for {question["id"]}')
        result.append({**question, 'compact': first})
    return result


def compile_plan(source, frozen, frozen_sha, states):
    requests = {'retry_on_invalid': ~states.valid}
    for name, key in [('learned_gate', 'selected'), ('without_token_confidence', 'without_token_confidence')]:
        picked = frozen[key]
        fitted = Gate.restore(picked['model'])
        requests[name] = dispatch(states, fitted.predict(states), picked['settings'])
    union = np.any(list(requests.values()), axis=0)
    return {'question_sha256': sha(source / 'splits/test-questions.jsonl'), 'first_sha256': record_sha(source, 'test-first'),
            'frozen_sha256': frozen_sha, 'requests': {name: [key for key, use in zip(states.ids, mask) if use]
                                                      for name, mask in requests.items()},
            'retry_union': [key for key, use in zip(states.ids, union) if use]}


def plan(source, output):
    frozen, frozen_sha = load_frozen(source, output)
    states = observe(first_rows(source))
    payload = compile_plan(source, frozen, frozen_sha, states)
    path = output / 'test-plan.json'
    if path.exists() and json.loads(path.read_bytes()) != payload:
        raise ValueError('Existing test plan differs from the frozen policy and observations')
    save(path, payload)
    print(f'{len(payload["retry_union"])} distinct retries required across all test policies')
    for name, requests in payload['requests'].items():
        print(f'{name}: {len(requests)} retries')


def summary(states, result, initial):
    groups = {}
    for domain in ('all', 'math', 'science'):
        mask = states.domains == domain if domain != 'all' else np.ones(len(states.ids), dtype=bool)
        graded = result['correct'][mask].astype(bool)
        correct_count = int(graded.sum())
        fixes = int(np.count_nonzero(graded & ~initial[mask]))
        regressions = int(np.count_nonzero(~graded & initial[mask]))
        query, accepted = result['query'][mask], result['accepted'][mask]
        groups[domain] = {'n': int(mask.sum()), 'correct': correct_count, 'accuracy': float(result['correct'][mask].mean()),
                          'mean_total_tokens': float(result['tokens'][mask].mean()),
                          'mean_solver_equivalent_tokens': float(WORK * result['tokens'][mask].mean()),
                          'queries': int(query.sum()), 'invalid_queries': int((query & ~states.valid[mask]).sum()),
                          'valid_queries': int((query & states.valid[mask]).sum()), 'submissions_from_retry': int(accepted.sum()),
                          'fixes': fixes, 'regressions': regressions,
                          'possible_fixes_queried': int(result['possible_fixes'][mask].sum())}
    return groups


def score(source, output):
    frozen, frozen_sha = load_frozen(source, output)
    rows = first_rows(source)
    states = observe(rows)
    compiled = json.loads((output / 'test-plan.json').read_bytes())
    if compiled != compile_plan(source, frozen, frozen_sha, states):
        raise ValueError('Test plan does not match the frozen gate')
    retry_ids = set(compiled['retry_union'])
    by_id = index_responses([row for row in rows if row['id'] in retry_ids], records(source, 'test-retries'))
    initial = np.array([correct(row['compact'], row['answer']) for row in rows], dtype=bool)
    short_correct = np.zeros(len(rows), dtype=bool)
    short_valid = np.zeros(len(rows), dtype=bool)
    disagree = np.zeros(len(rows), dtype=bool)
    short_cost = np.zeros(len(rows), dtype=int)
    observed_pairs = {}
    for i, row in enumerate(rows):
        if row['id'] not in by_id:
            continue
        retry = by_id[row['id']]['short']
        validate_response(row, row['compact'], retry)
        verify_short_prompt_cost(row, retry)
        short_correct[i] = correct(retry, row['answer'])
        short_valid[i] = bool(retry['answer'])
        disagree[i] = canonical_answer(row['compact']['answer']) != canonical_answer(retry['answer'])
        short_cost[i] = retry['input_tokens'] + retry['output_tokens']
        observed_pairs[i] = comparison_features({**row, 'short': retry})

    behavior = {'compact_only': []} | compiled['requests']
    results = {}
    for name, requests in behavior.items():
        selected_ids = set(requests)
        query = np.array([key in selected_ids for key in states.ids])
        probability, margin = np.zeros(len(rows)), 1
        if name in ('learned_gate', 'without_token_confidence') and query.any():
            selected = frozen['selected'] if name == 'learned_gate' else frozen[name]
            model = Gate.restore(selected['model'])
            probability[query] = model.comparison(np.array([observed_pairs[i] for i in np.flatnonzero(query)]))
            margin = model.margin
        accepted = query & accept_retry(states.valid, short_valid, disagree, probability, margin)
        results[name] = {'correct': np.where(accepted, short_correct, initial).astype(int),
                         'tokens': states.compact_tokens + query * short_cost, 'query': query, 'accepted': accepted,
                         'possible_fixes': query & short_correct & ~initial}
    measurements = {name: {'correct': r['correct'], 'work': r['tokens'] * WORK} for name, r in results.items()}
    rng = np.random.default_rng(20261117)
    payload = {'test_plan_sha256': sha(output / 'test-plan.json'), 'retry_responses_sha256': record_sha(source, 'test-retries'),
               'methods': {name: summary(states, result, initial) for name, result in results.items()},
               'paired_vs_retry_on_invalid': {name: paired(result, measurements['retry_on_invalid'], rng)
                                              for name, result in measurements.items() if name != 'retry_on_invalid'}}
    with new_run(output / 'test') as pending:
        save(pending / 'summary.json', payload)
        with (pending / 'per_question.jsonl').open('w', encoding='utf-8') as stream:
            for i, key in enumerate(states.ids):
                outcome = {name: {'correct': int(result['correct'][i]), 'tokens': int(result['tokens'][i]),
                                  'retry': bool(result['query'][i]), 'submitted_retry': bool(result['accepted'][i])}
                           for name, result in results.items()}
                stream.write(json.dumps({'id': key, 'domain': states.domains[i], 'compact_correct': bool(initial[i]),
                                         'outcomes': outcome}) + '\n')
    for name, metrics in payload['methods'].items():
        s = metrics['all']
        print(f'{name:<26} {s["accuracy"]:>7.2%} {s["mean_solver_equivalent_tokens"]:>8.2f} equivalent tokens, '
              f'{s["queries"]} retries, {s["fixes"]} fixes, {s["regressions"]} regressions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('tune', 'plan', 'score'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    actions = {'tune': tune, 'plan': plan, 'score': score}
    actions[args.phase](args.data, args.output)


if __name__ == '__main__':
    main()
