"""Evaluate an independently frozen retry gate with a tighter solver-token budget."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study_data import index_responses, records, sha, validate_response
from experiment import WORK, first_rows, record_sha, summary
from features import comparison_features, observe, paired as supervised, verify_short_prompt_cost
from gate import Gate, accept_retry, baseline, fold_ids, outcome, profile
from hillclimb import new_run, paired, save
from margin_model import allocate, load_fit
from margin_study import frozen as main_freeze
from scoring import canonical_answer, correct


def probabilities(original, direct):
    if direct:
        return {**original, 'fix': original['direct_fix'], 'harm': original['direct_harm']}
    return original


def evaluate(data, estimates, comparison, settings, direct, margin, control):
    requests = allocate(data.first, probabilities(estimates, direct), settings)
    return profile(data, outcome(data, requests, comparison, margin), control)


def cross_fitted(train, folds, design):
    estimates, comparison = {}, np.empty(len(folds))
    for fold in range(5):
        held = folds == fold
        model = Gate.fit(train.take(~held), *design)
        for name, values in model.predict(train.first.take(held)).items():
            estimates.setdefault(name, np.empty(len(folds)))[held] = values
        comparison[held] = model.comparison(train.comparison[held])
    return estimates, comparison


def evidence(data, output):
    _, main_digest = main_freeze(data, output.parent)
    return {'primary_freeze_sha256': main_digest, 'tight_budget_code_sha256': sha(Path(__file__))}


def tune(data, output):
    history, _, _, fresh, completed = load_fit(data)
    original, earlier, dev = (supervised(rows) for rows in (history, completed, fresh))
    train = supervised(history + completed)
    primary = baseline(dev)
    folds = np.r_[fold_ids(original.first), fold_ids(earlier.first)]
    original_mask = np.arange(len(folds)) < len(original.first.ids)
    groups = ((original_mask, original, baseline(original)), (~original_mask, earlier, baseline(earlier)))
    fits, candidates = {}, []
    for design in product(('linear', 'context'), (20, 64), (.35, .44, .57, .66)):
        model = fits[design] = Gate.fit(train, *design)
        estimates = model.predict(dev.first)
        comparison = model.comparison(dev.comparison)
        for credit, invalid_weight, harm_weight, direct in product((.55, .65, .75, .82, .9), (.9, 1.5, 2.1), (1, 1.6), (False, True)):
            settings = {'credit': credit, 'invalid_weight': invalid_weight, 'harm_weight': harm_weight}
            dev_metrics = evaluate(dev, estimates, comparison, settings, direct, model.margin, primary)
            if dev_metrics['accuracy_gain'] >= .01 and dev_metrics['tokens_saved'] >= 2.5:
                candidates.append({'design': design, 'settings': settings, 'direct': direct, 'dev': dev_metrics})

    ranked = sorted(candidates, key=lambda r: (r['dev']['accuracy_gain'], r['dev']['tokens_saved']), reverse=True)[:18]
    cached = {}
    best, best_rank = None, None
    for candidate in ranked:
        design = candidate['design']
        if design not in cached:
            cached[design] = cross_fitted(train, folds, design)
        predictions, comparison = cached[design]
        checks = [evaluate(block, {name: values[mask] for name, values in predictions.items()}, comparison[mask],
                           candidate['settings'], candidate['direct'], design[-1], control) for mask, block, control in groups]
        candidate['archive_out_of_fold'], candidate['completed_out_of_fold'] = checks
        if candidate['dev']['tokens_saved'] < 6 or any(check['accuracy_gain'] <= 0 for check in checks):
            continue
        net_correct = sum(round(check['n'] * check['accuracy_gain']) for check in checks)
        rank = (net_correct, candidate['dev']['accuracy_gain'], candidate['dev']['tokens_saved'])
        if best_rank is None or rank > best_rank:
            best, best_rank = candidate, rank
    if best is None:
        raise ValueError('No tighter gate improved accuracy in both out-of-fold groups')
    with new_run(output) as stage:
        save(stage / 'frozen.json', {'evidence': evidence(data, output), 'training_pairs': len(folds),
                                     'selected': best, 'model': fits[best['design']].as_dict()})
        save(stage / 'development-finalists.json', ranked)
    print(json.dumps(best, indent=2))


def locked(data, output):
    path = output / 'frozen.json'
    freeze = json.loads(path.read_bytes())
    if freeze['evidence'] != evidence(data, output):
        raise ValueError('Tight-budget freeze inputs changed')
    return freeze, sha(path)


def compile_plan(data, output, rows, freeze, digest):
    primary = output.parent / 'test-plan.json'
    parent = json.loads(primary.read_bytes())
    first_sha = record_sha(data, 'test-first')
    questions_sha = sha(data / 'splits/test-questions.jsonl')
    if parent['first_sha256'] != first_sha or parent['question_sha256'] != questions_sha:
        raise ValueError('Tight-budget gate and primary plan have different first responses')
    visible = [{key: row[key] for key in ('id', 'domain', 'subject', 'question', 'compact')} for row in rows]
    state = observe(visible)
    model = Gate.restore(freeze['model'])
    selected = freeze['selected']
    p = probabilities(model.predict(state), selected['direct'])
    take = allocate(state, p, selected['settings'])
    queries = [key for key, use in zip(state.ids, take) if use]
    existing = set(parent['retry_union'])
    return {'first_sha256': first_sha, 'question_sha256': questions_sha, 'frozen_sha256': digest,
            'primary_plan_sha256': sha(primary), 'requests': queries,
            'retry_union': [key for key in queries if key not in existing]}


def plan(data, output):
    freeze, digest = locked(data, output)
    payload = compile_plan(data, output, first_rows(data), freeze, digest)
    path = output / 'test-plan.json'
    if path.exists() and json.loads(path.read_bytes()) != payload:
        raise ValueError('Existing plan differs from the frozen tight-budget decisions')
    save(path, payload)
    print(f'{len(payload["requests"])} tight-budget retries, including {len(payload["retry_union"])} distinct from the primary run')


def score(data, output):
    freeze, digest = locked(data, output)
    rows = first_rows(data)
    payload = json.loads((output / 'test-plan.json').read_bytes())
    if payload != compile_plan(data, output, rows, freeze, digest):
        raise ValueError('Tight-budget plan changed after freeze')
    primary = json.loads((output.parent / 'test-plan.json').read_bytes())
    queried = set(payload['requests'])
    all_ids = set(primary['retry_union']) | set(payload['retry_union'])
    collected = records(data, 'test-retries') + records(data, 'tight-budget-retries')
    replies = index_responses([row for row in rows if row['id'] in all_ids], collected)
    state = observe(rows)
    initial = np.array([correct(row['compact'], row['answer']) for row in rows])
    query = np.array([row['id'] in queried for row in rows])
    visited, correct_after, valid_after, disagree, cost = [], [], [], [], []
    for row in (row for row in rows if row['id'] in queried):
        short = replies[row['id']]['short']
        validate_response(row, row['compact'], short)
        verify_short_prompt_cost(row, short)
        visited.append(comparison_features({**row, 'short': short}))
        correct_after.append(correct(short, row['answer']))
        valid_after.append(bool(short['answer']))
        disagree.append(canonical_answer(short['answer']) != canonical_answer(row['compact']['answer']))
        cost.append(short['input_tokens'] + short['output_tokens'])
    model = Gate.restore(freeze['model'])
    accepted = np.zeros(len(rows), dtype=bool)
    grades = initial.copy()
    tokens = state.compact_tokens.copy()
    fixes = np.zeros(len(rows), dtype=bool)
    if visited:
        chosen = accept_retry(state.valid[query], np.array(valid_after), np.array(disagree),
                              model.comparison(np.array(visited)), model.margin)
        accepted[query] = chosen
        grades[query] = np.where(chosen, correct_after, initial[query])
        tokens[query] += cost
        fixes[query] = ~initial[query] & correct_after
    policy = {'correct': grades.astype(int), 'tokens': tokens, 'query': query, 'accepted': accepted, 'possible_fixes': fixes}
    comparisons = {}
    reference_path = output.parent / 'test/per_question.jsonl'
    reference = [json.loads(line) for line in reference_path.read_text().splitlines()]
    if [entry['id'] for entry in reference] != list(state.ids):
        raise ValueError('The tight-budget and primary scores cover different test questions')
    rng = np.random.default_rng(20261119)
    for name in ('retry_on_invalid', 'previous_gate', 'new_gate_without_arithmetic'):
        previous = [entry['outcomes'][name] for entry in reference]
        comparisons[name] = paired({'correct': policy['correct'], 'work': tokens * WORK},
                                   {'correct': np.array([item['correct'] for item in previous]),
                                    'work': np.array([item['tokens'] for item in previous]) * WORK}, rng)
    report = {'methods': {'tight_budget_gate': summary(state, policy, initial)}, 'paired_vs_controls': comparisons,
              'test_plan_sha256': sha(output / 'test-plan.json'), 'primary_scores_sha256': sha(reference_path),
              'extra_retry_responses_sha256': record_sha(data, 'tight-budget-retries')}
    with new_run(output / 'test') as stage:
        save(stage / 'summary.json', report)
        with (stage / 'per_question.jsonl').open('w', encoding='utf-8') as stream:
            for idx, row in enumerate(rows):
                item = {'id': row['id'], 'domain': row['domain'], 'correct': int(grades[idx]), 'tokens': int(tokens[idx]),
                        'retry': bool(query[idx]), 'submitted_retry': bool(accepted[idx])}
                stream.write(json.dumps(item) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('tune', 'plan', 'score'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    {'tune': tune, 'plan': plan, 'score': score}[options.phase](options.data, options.output)


if __name__ == '__main__':
    main()
