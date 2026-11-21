"""Reference-blind saved-response replay and independent inference-cost accounting.

These utilities do not access a dataset, collect responses, or grade answers.
They are intended for the later prospective collector and audit stage.
"""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from difficulty_pomdp import MissingObservation
from prospective_freeze import observer_slots, sha


def prefix(tree, path):
    parts = path.split('/') if path else []
    paths = [''] + ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
    return {p: tree[p] for p in paths}


@contextmanager
def encoder_ledger(model):
    """Record one charged observation per unique model/input within a policy run.

Cached observations are still charged. Identical observer assets shared between
the two hybrid modes are charged once for the same paid response.
"""
    ledger, restore = {}, []
    if model is None:
        yield ledger
        return
    try:
        for _, observer in observer_slots(model):
            original = observer.values
            had_own = 'values' in vars(observer)
            previous = vars(observer).get('values')
            observer.load()
            recipe_sha = sha(Path(observer.directory) / 'recipe.json')
            parameters = observer._recipe['parameters'] / 1e9
            def wrapped(row, path, tree, judge=None, observer=observer, original=original,
                        recipe_sha=recipe_sha, parameters=parameters):
                if row.get('answer'):
                    raise ValueError('A reference answer reached the inference observer')
                paid = prefix(tree, path)
                entry = observer.entry(row, path, paid, judge)
                digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
                kind = 'solver' if judge is None else 'review'
                key = recipe_sha, row['id'], kind, path, digest
                value = original(row, path, paid, judge)
                charge = {'recipe_sha256': recipe_sha, 'kind': kind, 'path': path,
                          'slots': int(value[3]), 'parameters_billions': parameters,
                          'input_sha256': digest}
                if key in ledger and ledger[key] != charge:
                    raise ValueError('Changed repeated neural observation')
                ledger[key] = charge
                return value
            observer.values = wrapped
            restore.append((observer, had_own, previous))
        yield ledger
    finally:
        for observer, had_own, previous in restore:
            if had_own:
                observer.values = previous
            else:
                del observer.values


def replay_paid_inputs(function, row, available_tree, available_reviews, family, model=None):
    """Expose only the sequentially requested paid prefix; erase references first."""
    if family not in ('A', 'B', 'C', 'baseline'):
        raise ValueError('Unknown policy role')
    safe = {**row, 'answer': ''}
    tree, reviews, current = {'': available_tree['']}, {}, ''
    with encoder_ledger(model) as ledger:
        for _ in range(8):
            ledger.clear()
            try:
                result = function(safe, dict(tree), dict(reviews))
                if result['path'] != current:
                    raise ValueError('Policy discarded a paid solver response or returned an unpaid answer')
                result = {k: v for k, v in result.items() if k != 'correct'}
                audit_costs(result, tree, reviews, list(ledger.values()), family)
                return result, list(ledger.values())
            except (MissingObservation, KeyError) as error:
                missing = error
                if isinstance(error, KeyError):
                    key = error.args[0]
                    if isinstance(key, str) and key in available_tree:
                        missing = MissingObservation('solver', key)
                    elif isinstance(key, tuple) and key in available_reviews:
                        missing = MissingObservation('judge', key[1])
                    else:
                        raise
                kind, path = missing.kind, missing.path
                if kind == 'solver':
                    parent = path.rsplit('/', 1)[0] if '/' in path else ''
                    if path in tree or parent != current or len(tree) >= 3:
                        raise ValueError('Policy requested a nonsequential or over-budget solver path') from missing
                    if path not in available_tree:
                        raise
                    tree[path], current = available_tree[path], path
                elif kind == 'judge':
                    key = safe['id'], path
                    maximum = 0 if family in ('A', 'baseline') else 1 if family == 'C' else 2
                    if key in reviews or path != current or len(reviews) >= maximum or (family == 'C' and path):
                        raise ValueError('Policy requested a forbidden review') from missing
                    if key not in available_reviews:
                        raise
                    reviews[key] = available_reviews[key]
                else:
                    raise ValueError('Unexpected observation type') from missing
    raise ValueError('Policy exceeded the finite inference horizon')


def audit_costs(item, paid_tree, paid_reviews, encoders, family):
    expected_tree = prefix(paid_tree, item['path'])
    if set(expected_tree) != set(paid_tree) or len(paid_tree) > 3:
        raise ValueError('Paid solver receipts are not a single bounded trajectory')
    decisions = item.get('decisions', [])
    acquired = [d['path'] for d in decisions if d['queried']]
    if len(acquired) != len(set(acquired)) or set(acquired) != {p for _, p in paid_reviews}:
        raise ValueError('Review decisions disagree with the paid ledger')
    if family in ('A', 'baseline') and acquired:
        raise ValueError('A and fixed baselines cannot acquire LLM reviews')
    if family == 'C' and (len(acquired) > 1 or any(acquired)):
        raise ValueError('This C permits one optional root review')
    if len(acquired) > 2:
        raise ValueError('Exceeded review decision horizon')
    solver = sum(r['input_tokens'] + r['output_tokens'] for r in paid_tree.values())
    judge, physical = 0, 0
    for node in paid_reviews.values():
        judge += node['input_tokens'] + node['output_tokens']
        calls = node.get('calls')
        count = node.get('physical_api_calls', len(calls) if calls is not None else 1)
        physical += count
        if calls is not None:
            if len(calls) != count:
                raise ValueError('Incomplete physical LLM review receipts')
            if any(sum(c[field] for c in calls) != node[field] for field in ('input_tokens', 'output_tokens')):
                raise ValueError('Review phase token totals disagree')
    if (solver, judge, len(paid_tree), len(paid_reviews)) != (
            item['solver_tokens'], item['judge_tokens'], item['solver_calls'], item['judge_calls']):
        raise ValueError('Logical solver/review charges disagree with receipts')
    if 'judge_physical_calls' in item and item['judge_physical_calls'] != physical:
        raise ValueError('Physical review calls disagree')
    slots = sum(r['slots'] for r in encoders)
    if any(type(r['slots']) is not int or r['slots'] <= 0 or not math.isfinite(r['parameters_billions'])
           or r['parameters_billions'] <= 0 for r in encoders):
        raise ValueError('Invalid measured neural slot or parameter count')
    if slots != item.get('verifier_token_slots', 0) or slots != item.get('verifier_tokens', 0):
        raise ValueError('Neural token slots disagree with actual observation inputs')
    if len(encoders) != item.get('verifier_calls', 0):
        raise ValueError('Neural calls disagree with the paid observation ledger')
    encoder_work = sum(r['slots'] * r['parameters_billions'] for r in encoders)
    expected = (4 * solver + 3 * judge + encoder_work) / 3.09
    sensitivity = (4 * solver + 35 * judge + encoder_work) / 3.09
    if not math.isclose(item['compute'], expected, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError('Inference compute proxy omitted or double-counted work')
    if not math.isclose(item.get('total_parameter_compute', expected), sensitivity, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError('Total-review-parameter sensitivity disagrees')
    return {'passed': True, 'solver_calls': len(paid_tree), 'logical_review_calls': len(paid_reviews),
            'physical_review_calls': physical, 'encoder_calls': len(encoders), 'encoder_slots': slots}
