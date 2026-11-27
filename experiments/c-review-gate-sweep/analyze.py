"""Summarize sweep.json: per-subject frontier, uniform-setting curve, combined Lagrangian frontier."""
import json, sys
from collections import defaultdict

d = json.load(open(sys.argv[1]))
res = d['results']; counts = d['subject_counts']; N = sum(counts.values())
A = (153, 1087.27); B = (179, 3858.28); C0 = (177, 3728.58)
by = defaultdict(dict)
for r in res:
    by[r['subject']][(r['penalty'], r['threshold'])] = r

def pareto(points):  # points: (correct, compute, key) ; keep max correct for less compute
    out, best = [], -1
    for c, k, key in sorted(points, key=lambda p: (p[1], -p[0])):
        if c > best: out.append((c, k, key)); best = c
    return out

print('selected settings:', d['selected_settings'])
for s, combos in by.items():
    pts = [(r['correct'], r['compute_sum'] / r['n'], key) for key, r in combos.items()]
    print(f'\n== {s} (n={counts[s]}) Pareto: correct, mean compute, (penalty, threshold), review rate')
    for c, k, key in pareto(pts):
        print(f'  {c:4d}  {k:8.1f}  {key}  rr={combos[key]["review_rate"]:.2f}')

print('\n== Uniform settings (same penalty/threshold for all subjects), Pareto over all 256')
uni = []
for key in by[next(iter(by))]:
    c = sum(by[s][key]['correct'] for s in by); k = sum(by[s][key]['compute_sum'] for s in by) / N
    rr = sum(by[s][key]['review_rate'] * counts[s] for s in by) / N
    uni.append((c, k, key, rr))
for c, k, key in pareto([(u[0], u[1], u[2]) for u in uni]):
    rr = next(u[3] for u in uni if u[2] == key)
    print(f'  {c:4d}  {k:8.1f}  {key}  rr={rr:.2f}')

print('\n== Per-subject settings, combined frontier via Lagrangian (max correct - mu*compute_sum/1000)')
seen = set()
for mu in [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2]:
    pick = {s: max(combos, key=lambda key: combos[key]['correct'] - mu * combos[key]['compute_sum'] / 1000)
            for s, combos in by.items()}
    c = sum(by[s][pick[s]]['correct'] for s in by); k = sum(by[s][pick[s]]['compute_sum'] for s in by) / N
    if (c, round(k, 1)) in seen: continue
    seen.add((c, round(k, 1)))
    print(f'  mu={mu:<6} {c:4d} ({c/N:.1%})  {k:8.1f}  {pick}')
print(f'\nRefs: A {A}, B {B}, selected C {C0}')

print('\n== Exact combined frontier with per-subject settings (all Pareto combinations, compute > 1000)')
import itertools
P = {}
for s, combos in by.items():
    pts = [(r['correct'], r['compute_sum'], key, r['review_rate'] * r['n']) for key, r in combos.items()]
    pts.sort(key=lambda p: (p[1], -p[0])); keep, best = [], -1
    for p in pts:
        if p[0] > best: keep.append(p); best = p[0]
    P[s] = keep
rows = []
for x in itertools.product(*P.values()):
    rows.append((sum(p[0] for p in x), sum(p[1] for p in x) / N, sum(p[3] for p in x) / N, {s: p[2] for s, p in zip(P, x)}))
rows.sort(key=lambda r: (r[1], -r[0])); best = -1
for c, k, rr, pick in rows:
    if c > best and k > 1000:
        print(f'  {c:4d} ({c/N:.1%})  {k:8.1f}  vsC {k/C0[1]-1:+.1%}  vsB {k/B[1]-1:+.1%}  rr={rr:.2f}  {pick}'); best = c
