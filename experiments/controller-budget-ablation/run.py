"""Exploratory, explicitly restricted controller variants on cached development data."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / 'results/controller-hillclimb/pomdp-abc/hybrid-margin'
sys.path.insert(0, str(ROOT / 'experiments/controller-hillclimb/pomdp-abc/hybrid-margin'))
import margin_data  # Initialize the frozen experiment's shared module paths.
from context_pomdp import outcome
from extended_study import records
from hillclimb import new_run, save
from late_pairs import merged_records
from study import fixed
from study_common import BASELINES, evidence, models, run_policy, sha, statistics


def action_only_llm(row, tree, judges, max_extra):
    """Buy the LLM's action, but do not pass its proposed solution to the solver."""
    if max_extra not in (1, 2):
        raise ValueError('Only the declared one/two-continuation variants are supported')
    path = ''
    visited, acquired, decisions = [''], [], []
    for _ in range(max_extra):
        judge = judges[row['id'], path]
        action = judge['action']
        if action not in ('stop', 'repair', 'retry'):
            raise ValueError('Unexpected parsed LLM action')
        acquired.append(path)
        decisions.append({'path': path, 'action': action, 'queried': True})
        if action == 'stop':
            break
        path = f'{path}/{action}' if path else action
        visited.append(path)
    return outcome(row, tree, judges, path, visited, acquired, decisions)


def validate_restricted(row, tree, judges, result, max_extra):
    path = result['path']
    parts = path.split('/') if path else []
    visited = [''] + ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
    acquired = [d['path'] for d in result['decisions']]
    assert result['solver_calls'] == len(visited) <= max_extra + 1
    assert result['judge_calls'] == len(acquired) <= max_extra
    assert all(p in ('repair', 'retry') for p in parts)
    solver = sum(tree[p]['input_tokens'] + tree[p]['output_tokens'] for p in visited)
    review = sum(judges[row['id'], p]['input_tokens'] + judges[row['id'], p]['output_tokens'] for p in acquired)
    assert result['compute'] == (4 * solver + 3 * review) / 3.09
    paid_tree = {p: tree[p] for p in visited}
    paid_judges = {(row['id'], p): judges[row['id'], p] for p in acquired}
    expected = {k: v for k, v in result.items() if k != 'correct'}
    for reference in ('', 'unavailable-reference'):
        blinded = action_only_llm({**row, 'answer': reference}, paid_tree, paid_judges, max_extra)
        assert {k: v for k, v in blinded.items() if k != 'correct'} == expected


def run(args):
    frozen = json.loads((PRIMARY / 'frozen.json').read_bytes())
    data = PRIMARY / 'data'
    if evidence(data) != frozen['evidence']:
        raise ValueError('The primary experiment source or data has changed')
    rows, _, base_judges = records(data, 'dev')
    _, trees, late_judges = merged_records(data, 'dev')
    planners = models(frozen, PRIMARY)
    results = {name: [fixed(r, trees[r['id']], *configuration) for r in rows]
               for name, configuration in BASELINES.items()}
    for family in ('A', 'B', 'C'):
        results[family] = [run_policy(family, r, trees[r['id']], base_judges, late_judges,
                                      planners.get(family), frozen['selected'][family]) for r in rows]
    checks = 0
    for max_extra in (1, 2):
        name = f'B action-only, {max_extra} continuation(s)'
        results[name] = []
        for row in rows:
            tree = trees[row['id']]
            result = action_only_llm(row, tree, base_judges, max_extra)
            validate_restricted(row, tree, base_judges, result, max_extra)
            results[name].append(result)
            checks += 1
    stats = {k: statistics(v) for k, v in results.items()}
    limited = ('Stop', 'Repair once', 'Retry once')
    highest = max(limited, key=lambda k: (stats[k]['accuracy'], -stats[k]['mean_compute']))
    floor = stats[highest]['accuracy'] + .1
    scenarios = {}
    for max_extra in (1, 2):
        name = f'B action-only, {max_extra} continuation(s)'
        b = stats[name]
        threshold = max(stats['A']['accuracy'], b['accuracy']) + .1
        criteria = {'A_plus_10_points': stats['A']['accuracy'] >= floor - 1e-12,
                    'B_plus_10_points': b['accuracy'] >= floor - 1e-12,
                    'C_plus_10_points': stats['C']['accuracy'] >= threshold - 1e-12,
                    'C_compute_below_B': stats['C']['mean_compute'] < b['mean_compute']}
        scenarios[name] = {'highest_restricted_baseline': highest, 'A_B_required': floor,
                           'C_required': threshold, 'criteria': criteria,
                           'all_required_passed': all(criteria.values())}
    exposed = json.loads((PRIMARY / 'test/summary.json').read_bytes())['methods']
    a, b, c = (exposed[k]['accuracy'] for k in ('A', 'B', 'C'))
    strongest = max(exposed[k]['accuracy'] for k in BASELINES)
    feasibility = {'source': 'Already exposed 800-question result; algebra only, not new validation.',
                   'maximum_highest_baseline_with_fixed_A': a - .1,
                   'maximum_B_with_fixed_C': c - .1,
                   'minimum_baseline_accuracy_reduction': max(0., strongest - (a - .1)),
                   'minimum_B_accuracy_reduction': max(0., b - (c - .1))}
    report = {'split': 'previously used development', 'questions': len(rows),
              'fresh_validation': False, 'exploratory_after_primary_test': True,
              'interpretation': 'percentage points', 'methods': stats,
              'restricted_baselines': limited, 'scenarios': scenarios,
              'exposed_result_feasibility': feasibility,
              'primary_freeze_sha256': sha(PRIMARY / 'frozen.json'),
              'primary_evidence_unchanged': True,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'restricted_outcome_charge_checks': checks, 'restricted_blind_paid_replays': checks * 2,
              'new_inference_calls': 0,
              'limitations': ['Restricted baselines have one additional solver call; A/C retain two.',
                             'Restricted B ignores reviewer solutions and proposed answers; full cached review tokens are charged.',
                             'These are unequal-budget/information ablations, not replacements for strong references.',
                             'No candidate is selected to manufacture an accuracy ordering.']}
    with new_run(args.output) as stage:
        save(stage / 'summary.json', report)
        raw = ''.join(json.dumps({'method': name, **r}) + '\n' for name, items in results.items() for r in items)
        (stage / 'per_question.jsonl.gz').write_bytes(gzip.compress(raw.encode(), mtime=0))
        lines = ['Exploratory development-only budget/information ablation. Not a fresh result.', '',
                 '| Method | Accuracy (%) | Compute proxy |', '|---|---:|---:|']
        for name, value in stats.items():
            lines.append(f'| {name} | {100 * value["accuracy"]:.3f} | {value["mean_compute"]:.1f} |')
        lines += ['', 'One-continuation fixed baseline panel: Stop, Repair once, Retry once.', '',
                  'A/C and the full-strength references retain their frozen policies. Restricted B follows only the LLM action and uses plain compact-solver continuations. All original review tokens are charged.', '',
                  'The restricted conditions are explicit resource/information ablations. They do not replace the primary 800-question table or establish equal-budget superiority.', '',
                  'All requested criteria passed in any restricted scenario: **' + str(any(s['all_required_passed'] for s in scenarios.values())) + '**.', '',
                  'Full criteria and audit receipts: `summary.json`.', '']
        (stage / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'methods': stats, 'scenarios': scenarios, 'feasibility': feasibility}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
