"""Exact observed-subject accuracy/cost composition; no inference or label IO."""
from itertools import product
import math

SUBJECTS = ('math','chemistry','physics')
COUNTS = {'math':128,'chemistry':64,'physics':64}


def metrics(items):
    if not items or len({r['id'] for r in items}) != len(items):
        raise ValueError('Nonempty unique question records required')
    if any(r['correct'] not in (0,1,False,True) or not math.isfinite(r['compute']) or r['compute'] <= 0 for r in items):
        raise ValueError('Invalid correctness or compute')
    return {'n':len(items),'correct':sum(int(r['correct']) for r in items),
            'accuracy':sum(int(r['correct']) for r in items)/len(items),
            'mean_compute':math.fsum(r['compute'] for r in items)/len(items)}


def frontier(candidates,subject):
    if subject not in COUNTS: raise ValueError('Unknown observed subject')
    cheapest = {}
    for candidate in candidates:
        value = candidate['subjects'][subject]
        if value['n'] != COUNTS[subject]: raise ValueError('Changed common subject panel')
        key = value['correct']
        if key not in cheapest or (value['mean_compute'],candidate['key']) < (
                cheapest[key]['subjects'][subject]['mean_compute'],cheapest[key]['key']):
            cheapest[key] = candidate
    kept,best_cost = [],float('inf')
    for correct in sorted(cheapest,reverse=True):
        candidate = cheapest[correct]
        cost = candidate['subjects'][subject]['mean_compute']
        if cost < best_cost:
            kept.append(candidate)
            best_cost = cost
    return kept


def combine(choices):
    correct = sum(choices[s]['subjects'][s]['correct'] for s in SUBJECTS)
    return {'n':256,'correct':correct,'accuracy':correct/256,
            'mean_compute':math.fsum(COUNTS[s]*choices[s]['subjects'][s]['mean_compute'] for s in SUBJECTS)/256}


def best_B(candidates):
    choices = {s:min(candidates,key=lambda c:(-c['subjects'][s]['correct'],c['subjects'][s]['mean_compute'],c['key'])) for s in SUBJECTS}
    return choices,combine(choices)


def best_C(candidates,ceiling):
    if not math.isfinite(ceiling) or ceiling <= 0: raise ValueError('Finite matched-B compute ceiling required')
    fronts = {s:frontier(candidates,s) for s in SUBJECTS}
    best,metrics_best = None,None
    for values in product(*(fronts[s] for s in SUBJECTS)):
        choices = dict(zip(SUBJECTS,values))
        value = combine(choices)
        if value['mean_compute'] >= ceiling: continue
        identity = tuple(choices[s]['key'] for s in SUBJECTS)
        key = (-value['correct'],value['mean_compute'],identity)
        if best is None or key < best:
            best,metrics_best = key,(choices,value)
    return metrics_best, {s:len(fronts[s]) for s in SUBJECTS}
