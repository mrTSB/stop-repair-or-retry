"""Complete the critic replay's missing cached features without new LLM calls.

The first registered replay stopped on a missing old observer feature. This
separate variant preserves it, imports its128 completed new-critic forwards,
and supplies every missing original-observer science prefix from a fixed grid.
Previously sealed canonical solver responses supply missing branches uniformly.
"""
import argparse
import copy
import gc
from itertools import product
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits
import torch

import development_critic_review as parent

io, diagnostic, canonical, runtime, guard, cached = parent.io, parent.diagnostic, parent.canonical, parent.runtime, parent.guard, parent.cached
SCHEMA = 'science-review-critic-complete-cache-development-v2'


def inputs(args):
    old_spec = io.read(args.previous/'registration.json')
    old_args = SimpleNamespace(**{k: Path(v) for k, v in old_spec['paths'].items()}, registration_sha256=io.sha(args.previous/'registration.json'))
    bundle, models, original_args, expected = parent.inputs(old_args)
    if expected != old_spec: raise ValueError('Original registered replay changed')
    io.verify(old_spec['source_files']); io.verify(old_spec['input_evidence'])
    completed = io.read(args.previous/'features/complete.json')
    if completed['registration_sha256'] != old_args.registration_sha256 or completed['questions'] != 128:
        raise ValueError('Require the complete128-critic archive')
    io.verify({str(args.previous/'features'/n): h for n, h in completed['digests'].items()})
    rows, original_trees, trees, judges, candidates, model_specs, caches, substitutions, evidence = bundle
    records = canonical.previous.common.strict_records(original_args.canonical/'dev-observations.jsonl')
    # This validates every imported response against the same frozen prompts,
    # seeds, model digest, paid prefix, and canonical root review.
    _, extra_trees, extra_judges = canonical.previous.study.development_view(original_args.data,
                                                    original_args.canonical/'dev-observations.jsonl')
    imported = []
    for row in rows:
        if row['subject'] == 'math': continue
        qid = row['id']; tree = trees[qid]
        if extra_judges[qid, ''] != judges[qid, '']: raise ValueError('Canonical root review changed')
        for path in ('guided', 'guided/reuse', 'retry/reuse'):
            if path not in tree:
                tree[path] = extra_trees[qid][path]; imported.append({'id': qid, 'path': path, 'source': 'sealed canonical retained-path collection'})
        if 'guided/retry' not in tree:
            node = tree['repair/retry']; paired = canonical.previous.study.mentor_pairs
            if node['seed'] != paired.seed_for(qid, 'guided/retry'): raise ValueError('Retry alias seed differs')
            if node['prompt_sha256'] != diagnostic.hashlib.sha256(canonical.previous.study.old.retry_prompt({**row, 'answer': ''}).encode()).hexdigest():
                raise ValueError('Retry alias prompt differs')
            tree['guided/retry'] = node; imported.append({'id': qid, 'path': 'guided/retry', 'source': 'exact original repair/retry alias'})
    observer = models['combined'].mentor.classifier
    observer._recipe = io.read(Path(observer.directory)/'recipe.json')
    cache = caches[canonical.previous.previous.share_key(observer)]; plan = []
    for row in rows:
        if row['subject'] == 'math': continue
        qid = row['id']; safe = {**row, 'answer': ''}
        for kind, path in [('solver', ''), ('solver', 'guided'), ('solver', 'retry'), ('review', '')]:
            judge = judges[qid, ''] if kind == 'review' else None
            node = trees[qid][path] if judge is None else judge
            key = observer.key(safe, path, node, kind)
            if key not in cache:
                entry = observer.entry(safe, path, runtime.original.prefix(trees[qid], path), judge)
                plan.append({'id': qid, 'kind': kind, 'path': path, 'input_sha256': diagnostic.inputs.fingerprint(entry)})
    evidence = {**old_spec['input_evidence'], **{str((args.previous/'features'/n).resolve()): h for n, h in completed['digests'].items()}}
    for p in (args.previous/'registration.json', args.previous/'features/complete.json', original_args.canonical/'dev-observations.jsonl'):
        evidence[str(p.resolve())] = io.sha(p)
    record = {**old_spec, 'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {k: str(getattr(args, k).resolve()) for k in ('previous', 'output')}, 'inherited_paths': old_spec['paths'],
        'imported_solver_branches': imported, 'new_encoder_passes': len(plan), 'feature_plan': plan,
        'reused_critic_forwards': 128, 'reused_critic_source_registration': old_args.registration_sha256,
        'new_cloud_calls': 0, 'policy_grid_and_fit_unchanged': True,
        'retry_alias': 'Existing plain repair/retry prompt, seed, frozen model, cap, temperature and context; no new call'}
    return bundle, models, original_args, old_args, record


def restored(args, record, require_complete=False):
    folder = args.output/'features'; attempts = diagnostic.protocol.protocol.strict_records(folder/'attempts.jsonl')
    values = diagnostic.protocol.protocol.strict_records(folder/'observations.jsonl'); known = {}
    expected = {diagnostic.inputs.fingerprint(r): r for r in record['feature_plan']}
    attempted = set()
    for row in attempts:
        key = row['key']
        if key not in expected or key in attempted or row['registration_sha256'] != args.registration_sha256: raise ValueError('Changed forward attempt')
        attempted.add(key)
    for row in values:
        key = row['key']
        if (key not in attempted or key in known or row['registration_sha256'] != args.registration_sha256
                or [len(v) for v in row['outputs']] != [4, 8, 8] or not 0 < row['slots'] <= 512
                or not all(np.isfinite(v).all() for v in row['outputs'])): raise ValueError('Changed feature receipt')
        known[key] = row
    if attempted != set(known): raise ValueError('Unresolved forward attempt; no automatic repeat')
    if require_complete and set(known) != set(expected): raise ValueError('Incomplete fixed feature grid')
    return known


def extract(args, bundle, models, record):
    rows, _, trees, judges = bundle[:4]; by_id = {r['id']: r for r in rows}
    folder = args.output/'features'; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists(): raise ValueError('Preserve completed features')
    known = restored(args, record); observer = None
    try:
        for item in record['feature_plan']:
            key = diagnostic.inputs.fingerprint(item)
            if key in known: continue
            diagnostic.extractor.memory_guard(folder, observer is not None)
            if observer is None:
                observer = copy.deepcopy(models['combined'].mentor.classifier); observer.load(); torch.set_num_threads(2)
            io.append(folder/'attempts.jsonl', {'key': key, 'registration_sha256': args.registration_sha256})
            row = {**by_id[item['id']], 'answer': ''}; path = item['path']; tree = runtime.original.prefix(trees[row['id']], path)
            judge = judges[row['id'], ''] if item['kind'] == 'review' else None
            if diagnostic.inputs.fingerprint(observer.entry(row, path, tree, judge)) != item['input_sha256']: raise ValueError('Feature input changed')
            start = time.perf_counter(); values = observer.values(row, path, tree, judge)
            result = {'key': key, 'outputs': [v.tolist() for v in values[:3]], 'slots': int(values[3]),
                'elapsed_seconds': time.perf_counter()-start, 'registration_sha256': args.registration_sha256}
            io.append(folder/'observations.jsonl', result); known[key] = result; observer._cache.clear()
    finally: observer = None; gc.collect()
    restored(args, record, True)
    io.write(folder/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'encoder_passes': len(known), 'token_slots': sum(v['slots'] for v in known.values()),
        'digests': {n: io.sha(folder/n) for n in ('attempts.jsonl', 'observations.jsonl', 'memory.jsonl')}, 'new_cloud_calls': 0})
    print({'completed_old_observer_passes': len(known)}, flush=True)


def prepare_replay(args, bundle, models, old_args, record):
    folder = args.output/'features'; done = io.read(folder/'complete.json')
    if done['registration_sha256'] != args.registration_sha256: raise ValueError('Changed feature completion')
    io.verify({str(folder/n): h for n, h in done['digests'].items()}); values = restored(args, record, True)
    rows, _, trees, judges, _, _, caches = bundle[:7]; by_id = {r['id']: r for r in rows}
    observer = models['combined'].mentor.classifier; cache = caches[canonical.previous.previous.share_key(observer)]
    for item in record['feature_plan']:
        node = trees[item['id']][item['path']] if item['kind'] == 'solver' else judges[item['id'], '']
        key = observer.key(by_id[item['id']], item['path'], node, item['kind'])
        if key in cache: raise ValueError('Never overwrite a previously cached feature')
        value = values[diagnostic.inputs.fingerprint(item)]; cache[key] = (*[np.asarray(v) for v in value['outputs']], value['slots'])
    saved = diagnostic.restore(SimpleNamespace(output=args.previous/'features', registration_sha256=old_args.registration_sha256),
                               [r for r in rows if r['subject'] != 'math'], True)
    return saved


def replay(args, bundle, models, original_args, record, saved):
    rows, _, trees, judges, _, _, caches = bundle[:7]; directory = args.output/'features'
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
        'new_encoder_passes': record['new_encoder_passes'], 'reused_critic_forwards': 128, 'new_cloud_calls': 0, 'test_used': False, 'fresh_validation': False}
    io.write(args.output/'private-candidate-items.json', all_items); io.write(args.output/'summary.json', summary)
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'artifacts': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')},
        'feature_complete_sha256': io.sha(directory/'complete.json'), 'test_used': False})
    print({'selected_C': selected['dev'], 'gap': summary['C_above_higher_A_B']}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'extract', 'replay'))
    for n in ('previous', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with guard.no_test_reads():
        with diagnostic.guarded.guarded.no_model_calls(): bundle, models, original_args, old_args, record = inputs(args)
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json'), 'missing_encoder_passes': len(record['feature_plan']),
                   'imported_solver_branches': len(record['imported_solver_branches']), 'new_cloud_calls': 0}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed extended replay registration')
        with io.lock(args.output/'.run.lock'):
            if args.phase == 'extract': extract(args, bundle, models, record)
            else:
                with diagnostic.guarded.guarded.no_model_calls(), threadpool_limits(limits=1):
                    saved = prepare_replay(args, bundle, models, old_args, record)
                    args.models = old_args.models
                    replay(args, bundle, models, original_args, record, saved)


if __name__ == '__main__': main()
