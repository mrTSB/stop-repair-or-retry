"""Exact-count checks for the user's >80% C target and retained gap targets.

This module does not select policies or load references. A development pass
can never be reported as fresh confirmation. The same rules apply to all
future matched comparisons; improving B raises C's required accuracy.
"""
import math

BASELINES = ('Stop', 'Repair once', 'Repair twice', 'Retry once', 'Retry twice')


def assess(rows, *, fresh_validation=False):
    by_name = {r['method']: r for r in rows}
    if len(rows) != 8 or set(by_name) != {*BASELINES, 'A', 'B', 'C'}:
        raise ValueError('Require exactly the five fixed baselines and best A/B/C')
    n = by_name['Stop']['n']
    if type(n) is not int or n <= 0:
        raise ValueError('Positive common question count required')
    for row in rows:
        if (row['n'] != n or type(row['correct']) is not int or not 0 <= row['correct'] <= n
                or not math.isfinite(row['mean_compute']) or row['mean_compute'] <= 0):
            raise ValueError('Require matched counts and finite positive inference compute')
        if not math.isclose(row['accuracy'], row['correct']/n, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Reported accuracy disagrees with integer outcomes')
    a, b, c = (by_name[k]['correct'] for k in ('A', 'B', 'C'))
    best_baseline = max(by_name[k]['correct'] for k in BASELINES)
    # Integer inequalities avoid rounding 79.999... into an80% pass.
    checks = {'A_gap_at_least10pp': 10*(a-best_baseline) >= n,
              'B_gap_at_least10pp': 10*(b-best_baseline) >= n,
              'C_gap_at_least10pp': 10*(c-max(a,b)) >= n,
              'C_accuracy_strictly_over80percent': 5*c > 4*n,
              'C_compute_strictly_below_B': by_name['C']['mean_compute'] < by_name['B']['mean_compute']}
    target_ab = best_baseline+(n+9)//10
    target_c = max((4*n)//5+1, max(a,b)+(n+9)//10)
    return {'n': n, 'checks': checks, 'all_metrics_passed': all(checks.values()),
            'fresh_validation': bool(fresh_validation),
            'fresh_targets_confirmed': bool(fresh_validation) and all(checks.values()),
            'minimum_correct': {'A': target_ab, 'B': target_ab, 'C': target_c},
            'additional_correct_needed': {'A': max(0,target_ab-a), 'B': max(0,target_ab-b), 'C': max(0,target_c-c)},
            'accuracy_targets_feasible_on_this_panel': target_ab <= n and target_c <= n}
