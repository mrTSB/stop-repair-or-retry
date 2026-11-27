"""Sweep C's review gate (penalty, threshold) per subject on cached dev.

Copied into <bundle>/scripts/reproduction/ and invoked via run.py, exactly like
replay_latest.py. Loads C the same way, first checks the selected settings
reproduce Tanvir's per-question items, then re-evaluates every subject's rows
under each (penalty, threshold). No model calls, no test reads, no fitting.
"""
import argparse
from contextlib import ExitStack
import copy
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / 'experiments/controller-semantic-hillclimb/followup/round2'
sys.path.insert(0, str(AREA))

from threadpoolctl import threadpool_limits
import development_bayesian_hybrid_v3 as c_runtime

io = c_runtime.io
RUNS = ROOT / 'runs/controller-semantic-followup/round2'
PENALTIES = [0.0, 0.01, 0.03, 0.1, 0.248, 0.5, 1.0, 2.0]
THRESHOLDS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
STATE = {}


def run_combo(job):
    subject, penalty, threshold = job
    model, rows, trees, judges = STATE['model'], STATE['rows'], STATE['trees'], STATE['judges']
    expert = model.experts[subject]
    saved = expert['settings']
    expert['settings'] = {'penalty': penalty, 'threshold': threshold}
    out = []
    try:
        for row in rows:
            if row['subject'] != subject:
                continue
            item = c_runtime.original.policy.routed(row, trees[row['id']], judges, model, {}, 'C')
            out.append({'id': row['id'], 'correct': int(item['correct']), 'compute': float(item['compute']),
                        'queried': bool(item['decisions'][0].get('queried', False)), 'path': item['path']})
    finally:
        expert['settings'] = saved
    return {'subject': subject, 'penalty': penalty, 'threshold': threshold, 'n': len(out),
            'correct': sum(o['correct'] for o in out), 'compute_sum': math.fsum(o['compute'] for o in out),
            'review_rate': sum(o['queried'] for o in out) / max(len(out), 1), 'items': out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    guards = c_runtime.original.prior
    with guards.guard.no_test_reads(), guards.guarded.no_model_calls(), threadpool_limits(limits=1):
        c_folder = RUNS / 'development-bayesian-model-hybrid-v3'
        spec = io.read(c_folder / 'registration.json')
        with c_runtime.configured():
            bundle, _, _, record = c_runtime.original.inputs(
                SimpleNamespace(previous=Path(spec['previous']), output=c_folder))
        if record != spec:
            raise ValueError('C input registration differs')
        c_rows, c_trees, c_judges, _, caches, *_ = bundle
        summary = io.read(c_folder / 'summary.json')
        chosen = summary['C_candidate']
        if io.sha(chosen['model_path']) != chosen['model_sha256']:
            raise ValueError('C selected model changed')
        model = pickle.loads(Path(chosen['model_path']).read_bytes())
        c_expected = io.read(c_folder / 'private-selected-items.json')
        with ExitStack() as stack:
            for child in model.models.values():
                parents = child.models if isinstance(child, c_runtime.original.policy.Planner) else [child]
                for parent in parents:
                    stack.enter_context(c_runtime.original.canonical.cached_model(parent, caches))
            # Sanity: selected settings must reproduce the archived per-question C items.
            t0 = time.time()
            base = [c_runtime.original.policy.routed(r, c_trees[r['id']], c_judges, model, {}, 'C') for r in c_rows]
            for item, expected in zip(base, c_expected, strict=True):
                for key in ('path', 'correct', 'compute'):
                    if item.get(key) != expected.get(key):
                        raise ValueError(f'Baseline C differs for {item["id"]}: {key}')
            print(f'Baseline C reproduced: {sum(i["correct"] for i in base)}/{len(base)} '
                  f'mean compute {math.fsum(i["compute"] for i in base)/len(base):.2f} '
                  f'({time.time()-t0:.1f}s)', flush=True)
            selected = {s: e['settings'] for s, e in model.experts.items()}
            STATE.update(model=model, rows=c_rows, trees=c_trees, judges=c_judges)
            subjects = sorted({r['subject'] for r in c_rows})
            jobs = [(s, p, t) for s in subjects for p in PENALTIES for t in THRESHOLDS]
            t0 = time.time()
            ctx = mp.get_context('fork')
            with ctx.Pool(args.workers) as pool:
                results = []
                for i, res in enumerate(pool.imap_unordered(run_combo, jobs), 1):
                    results.append(res)
                    if i % 20 == 0:
                        print(f'{i}/{len(jobs)} combos ({time.time()-t0:.0f}s)', flush=True)
    counts = {s: sum(r['subject'] == s for r in c_rows) for s in subjects}
    io.write(args.output / 'sweep.json', {'selected_settings': selected, 'subject_counts': counts,
                                          'penalties': PENALTIES, 'thresholds': THRESHOLDS,
                                          'results': results, 'test_used': False, 'new_model_calls': 0})
    print('done', len(results), 'combos', flush=True)


if __name__ == '__main__':
    main()
