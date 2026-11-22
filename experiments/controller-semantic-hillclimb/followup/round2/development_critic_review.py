"""Full256 DEV comparison of two separately fitted review-critic hybrids.

All128 science root-review observations are sealed before any new policy score.
The exact canonical compact responses, strongest A/B/C, and all five original
baselines are retained. No cloud acquisition or fresh-test access is permitted.
"""
import argparse
from contextlib import contextmanager
import copy
import gc
from itertools import product
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits
import torch

import expanded_c_canonical_cost as canonical
import fit_critic_review as fitting
import critic_review_hybrid as runtime
import development_completion_blend as guard

io, diagnostic = fitting.io, fitting.diagnostic
SCHEMA = 'science-review-critic-full256-development-v1'


def inputs(args):
    source = io.read(args.source/'registration.json')
    if io.read(args.source/'complete.json') != canonical.seal_payload(args.source): raise ValueError('Canonical C seal changed')
    original_args = SimpleNamespace(**{k: Path(v) for k, v in source['paths'].items()})
    bundle = canonical.inputs(original_args)
    if canonical.contract(original_args, bundle) != source: raise ValueError('Canonical study inputs changed')
    io.verify(source['source_files']); summary = io.read(args.source/'summary.json')
    spec = io.read(args.models/'registration.json'); done = io.read(args.models/'complete.json')
    if (done['schema'] != fitting.SCHEMA or spec['schema'] != fitting.SCHEMA
            or done['registration_sha256'] != io.sha(args.models/'registration.json')
            or done['new_encoder_calls'] or done['new_cloud_calls'] or done['development_or_test_read']):
        raise ValueError('Require completed TRAIN-only hybrid fits')
    io.verify(spec['source_files']); io.verify(spec['input_evidence'])
    io.verify({str(args.models/n): h for n, h in done['digests'].items()})
    if spec['paths']['base'] != summary['original_selected_C']['model_path']: raise ValueError('Different fixed hybrid base')
    models = {}
    for name in ('option', 'combined'):
        path = args.models/(name+'.pkl')
        if io.sha(path) != done['models'][name]['model_sha256']: raise ValueError('Changed fitted critic')
        models[name] = pickle.loads(path.read_bytes())
    a = io.read(args.a_reference); a_done = io.read(args.a_reference.parent/'complete.json')
    io.verify({str(args.a_reference.parent/n): h for n, h in a_done['artifacts'].items()})
    source_summary = io.read(original_args.source/'summary.json')
    if a['baselines'] != source_summary['baselines']: raise ValueError('Different A/B/C baseline panel')
    evidence = {**bundle[-1], **spec['input_evidence'], **{str((args.models/n).resolve()): h for n, h in done['digests'].items()}}
    for p in (args.source/'registration.json', args.source/'complete.json', args.source/'summary.json',
              args.models/'complete.json', args.a_reference, args.a_reference.parent/'complete.json'):
        evidence[str(p.resolve())] = io.sha(p)
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {n: str(getattr(args, n).resolve()) for n in ('models', 'source', 'a_reference', 'output')},
        'dev_questions_sha256': source['evidence'][str(original_args.data/'splits/dev-questions.jsonl')],
        'development_questions': 256, 'new_encoder_passes': 128, 'new_cloud_calls': 0,
        'families': ['option', 'combined'], 'penalties': list(canonical.previous.PENALTIES),
        'thresholds': list(canonical.previous.THRESHOLDS), 'utilities': list(canonical.previous.UTILITIES),
        'candidate_count': 52, 'math_settings': summary['original_selected_C']['settings']['by_subject']['math'],
        'math_rule': 'Exact original selected C decisions for every candidate',
        'previous_C': summary['original_selected_under_canonical_receipts'], 'A': a['selected_A'], 'B': summary['B'],
        'baselines': source_summary['baselines'], 'matched_complete_controls': a['matched_complete_controls'],
        'selection': 'Retain previous C; maximize accuracy among candidates strictly cheaper than strongest B, then minimize compute',
        'test_used': False, 'fresh_validation': False, 'device': 'cpu', 'threads': 2,
        'resume': 'Per-question forward attempt journal; unresolved attempts block automatic repeats'}
    return bundle, models, original_args, record


def features(args, bundle, models, record):
    rows, _, trees, judges = bundle[:4]; science = [r for r in rows if r['subject'] != 'math']
    directory = args.output/'features'; directory.mkdir(exist_ok=True)
    local = SimpleNamespace(output=directory, registration_sha256=args.registration_sha256)
    if (directory/'complete.json').exists(): raise ValueError('Preserve completed observations')
    known = diagnostic.restore(local, science); observer = None
    try:
        for row in science:
            qid = row['id']
            if qid in known: continue
            diagnostic.extractor.memory_guard(directory, observer is not None)
            if observer is None:
                observer = copy.deepcopy(models['option'].critic.observer); observer.load(); torch.set_num_threads(2)
            io.append(directory/'attempts.jsonl', {'id': qid, 'public_sha256': diagnostic.inputs.fingerprint(diagnostic.public(row)),
                                                 'registration_sha256': args.registration_sha256})
            start = time.perf_counter(); safe = diagnostic.public(row); paid = {'': trees[qid]['']}; judge = judges[qid, '']
            state, _, _, slots = observer.values(safe, '', paid, judge); entry = observer.entry(safe, '', paid, judge)
            item = {'id': qid, 'logit': float(diagnostic.margin(state)/observer.temperature), 'state': state.tolist(),
                'token_slots': int(slots), 'input_sha256': diagnostic.inputs.fingerprint(entry),
                'elapsed_seconds': time.perf_counter()-start, 'registration_sha256': args.registration_sha256}
            io.append(directory/'observations.jsonl', item); known[qid] = item; observer._cache.clear()
            if len(known) % 32 == 0: print({'review_observations': len(known), 'total': 128}, flush=True)
    finally: observer = None; gc.collect()
    diagnostic.restore(local, science, True)
    io.write(directory/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'questions': 128, 'encoder_passes': 128, 'actual_token_slots': sum(v['token_slots'] for v in known.values()),
        'digests': {n: io.sha(directory/n) for n in ('attempts.jsonl', 'observations.jsonl', 'memory.jsonl')},
        'new_cloud_calls': 0, 'test_used': False})


@contextmanager
def cached(model, caches, saved):
    observer = model.critic.observer; observer._recipe = io.read(Path(observer.directory)/'recipe.json')
    def load():
        if observer._model is not None: raise ValueError('No live neural model in sealed replay')
    def values(row, path, tree, judge=None):
        if row.get('answer') or path != '' or judge is None or set(tree) != {''}: raise ValueError('Review sensor scope changed')
        item = saved[row['id']]; entry = observer.entry(row, path, tree, judge)
        if diagnostic.inputs.fingerprint(entry) != item['input_sha256']: raise ValueError('Changed cached critic input')
        return np.asarray(item['state']), np.zeros(8), np.zeros(8), item['token_slots']
    with canonical.cached_model(model, caches), patch.object(observer, 'load', load), patch.object(observer, 'values', values):
        yield


def replay(args, bundle, models, original_args, record):
    rows, _, trees, judges, _, _, caches = bundle[:7]; directory = args.output/'features'
    done = io.read(directory/'complete.json')
    if done['registration_sha256'] != args.registration_sha256 or done['questions'] != 128: raise ValueError('Wrong feature cohort')
    io.verify({str(directory/n): h for n, h in done['digests'].items()})
    saved = diagnostic.restore(SimpleNamespace(output=directory, registration_sha256=args.registration_sha256),
                               [r for r in rows if r['subject'] != 'math'], True)
    anchor = pickle.loads(Path(record['previous_C']['model_path']).read_bytes()) if 'model_path' in record['previous_C'] else pickle.loads(
        (original_args.source/'models'/(record['previous_C']['model']+'.pkl')).read_bytes())
    with canonical.cached_model(anchor, caches):
        old_items = [runtime.original.evaluate(r, trees[r['id']], judges, anchor, record['previous_C']['settings'], 'C') for r in rows]
    old_stats = canonical.previous.measured(old_items)
    if old_stats != record['previous_C']['dev']: raise ValueError('Previous canonical C metrics changed')
    math = {r['id']: item for r, item in zip(rows, old_items) if r['subject'] == 'math'}
    candidates, all_items = [], {}
    for name, model in models.items():
        grid = []
        with cached(model, caches, saved):
            for penalty, threshold in product(record['penalties'], record['thresholds']):
                setting = {'penalty': penalty, 'threshold': threshold}
                full = {'by_subject': {'math': record['math_settings'], 'chemistry': setting, 'physics': setting}}
                items = [runtime.evaluate(r, trees[r['id']], judges, model, full, 'C') for r in rows]
                if any(item != math[r['id']] for r, item in zip(rows, items) if r['subject'] == 'math'):
                    raise ValueError('Math decisions or charges changed')
                grid.append((full, items))
            for utility in record['utilities']:
                selected = {s: max(range(21), key=lambda i: (
                    (v := canonical.previous.measured([x for r, x in zip(rows, grid[i][1]) if r['subject'] == s]))['accuracy']
                    - utility*v['mean_compute']/1000, v['accuracy'], -v['mean_compute'])) for s in ('chemistry', 'physics')}
                full = {'by_subject': {'math': record['math_settings'], **{s: grid[i][0]['by_subject'][s] for s, i in selected.items()}}}
                items = [math[r['id']] if r['subject'] == 'math' else grid[selected[r['subject']]][1][i] for i, r in enumerate(rows)]
                grid.append((full, items))
        for index, (settings, items) in enumerate(grid):
            key = name+'-'+str(index); all_items[key] = items
            candidates.append({'key': key, 'model': name, 'model_path': str((args.models/(name+'.pkl')).resolve()),
                'model_sha256': io.sha(args.models/(name+'.pkl')), 'policy_module': 'critic_review_hybrid',
                'settings': settings, 'dev': canonical.previous.measured(items)})
        print({'completed_model': name, 'candidates': len(candidates), 'best_correct': max(c['dev']['correct'] for c in candidates)}, flush=True)
    if len(candidates) != 52: raise ValueError('Incomplete registered grid')
    eligible = [c for c in [record['previous_C'], *candidates] if c['dev']['mean_compute'] < record['B']['dev']['mean_compute']]
    selected = max(eligible, key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
    summary = {'schema': SCHEMA, 'selected_C': selected, 'candidates': candidates, 'previous_C': record['previous_C'],
        'A': record['A'], 'B': record['B'], 'baselines': record['baselines'], 'matched_complete_controls': record['matched_complete_controls'],
        'C_above_higher_A_B': selected['dev']['accuracy']-max(record[n]['dev']['accuracy'] for n in ('A', 'B')),
        'C_compute_saved_vs_B': 1-selected['dev']['mean_compute']/record['B']['dev']['mean_compute'],
        'previous_C_exactly_reproduced': True, 'math_exactly_preserved': True, 'development_questions': 256,
        'new_encoder_passes': 128, 'new_cloud_calls': 0, 'test_used': False, 'fresh_validation': False}
    io.write(args.output/'private-candidate-items.json', all_items); io.write(args.output/'summary.json', summary)
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'artifacts': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
        'feature_complete_sha256': io.sha(directory/'complete.json'), 'test_used': False})
    print({'selected_C': selected['dev'], 'gap': summary['C_above_higher_A_B']}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'extract', 'replay'))
    for n in ('models', 'source', 'a-reference', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with guard.no_test_reads():
        with diagnostic.guarded.guarded.no_model_calls(): bundle, models, original_args, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed development registration')
        with io.lock(args.output/'.run.lock'):
            if args.phase == 'extract': features(args, bundle, models, record)
            else:
                with diagnostic.guarded.guarded.no_model_calls(), threadpool_limits(limits=1): replay(args, bundle, models, original_args, record)


if __name__ == '__main__': main()
