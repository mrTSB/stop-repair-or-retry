"""Replay the richer Complete16/mentor8 C on all256 DEV with retained A/B/C."""
import argparse
import copy
from itertools import product
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from threadpoolctl import threadpool_limits

import development_nonlinear_completion as a_source
import development_critic_review_extended as c_source
import development_completion_b as b_source
import fit_nonlinear_completion_hybrid_v2 as fitted
import nonlinear_completion_hybrid_v2 as policy

io, canonical = fitted.io, c_source.canonical
SCHEMA = 'nonlinear-Complete16-mentor8-full-development-v1'


def saved_args(directory):
    spec = io.read(directory/'registration.json')
    return SimpleNamespace(**{k: Path(v) for k, v in spec['paths'].items()},
                           registration_sha256=io.sha(directory/'registration.json'))


def inputs(args):
    aa = saved_args(args.a_reference); adata, _, values, _, arecord = a_source.inputs(aa)
    if arecord != io.read(args.a_reference/'registration.json'): raise ValueError('Changed A evidence')
    adone = io.read(args.a_reference/'complete.json')
    io.verify({str(args.a_reference/n): h for n, h in adone['artifacts'].items()})
    ca = saved_args(args.c_cache); bundle, critics, _, old_args, crecord = c_source.inputs(ca)
    if crecord != io.read(args.c_cache/'registration.json'): raise ValueError('Changed canonical C cache')
    cdone = io.read(args.c_cache/'complete.json')
    io.verify({str(args.c_cache/n): h for n, h in cdone['artifacts'].items()})
    c_source.prepare_replay(ca, bundle, critics, old_args, crecord)
    if io.read(args.b_reference/'complete.json') != b_source.seal_payload(args.b_reference): raise ValueError('Changed matched B comparison')
    bspec = io.read(args.b_reference/'registration.json'); io.verify(bspec['source_files'])
    bs = io.read(args.b_reference/'summary.json'); av = io.read(args.a_reference/'summary.json')
    rows, _, trees, judges = bundle[:4]
    if rows != adata[0] or bs['baselines'] != av['baselines'] or bs['B']['dev'] != crecord['B']['dev']:
        raise ValueError('A/B/C panels or strongest matched B changed')
    for row in rows:
        if row['subject'] == 'math': continue
        qid = row['id']
        for path, node in adata[1][qid].items():
            if path in trees[qid] and trees[qid][path] != node: raise ValueError('Changed shared plain response')
            trees[qid][path] = node
    fspec = io.read(args.model/'registration.json'); fdone = io.read(args.model/'complete.json')
    if fspec['schema'] != fitted.SCHEMA or fdone['registration_sha256'] != io.sha(args.model/'registration.json'):
        raise ValueError('Changed fitted transfer')
    io.verify(fspec['source_files']); io.verify(fspec['input_evidence'])
    io.verify({str(args.model/n): h for n, h in fdone['digests'].items()})
    path = args.model/'hybrid.pkl'
    if io.sha(path) != fdone['model_sha256']: raise ValueError('Changed hybrid model')
    model = pickle.loads(path.read_bytes())
    evidence = {**arecord['input_evidence'], **crecord['input_evidence'], **fspec['input_evidence']}
    for directory, marker, key in ((args.a_reference, adone, 'artifacts'), (args.c_cache, cdone, 'artifacts'),
                                    (args.model, fdone, 'digests'), (args.b_reference, b_source.seal_payload(args.b_reference), 'files')):
        evidence.update({str((directory/n).resolve()): h for n, h in marker[key].items()})
        evidence[str((directory/'complete.json').resolve())] = io.sha(directory/'complete.json')
    record = {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]), 'input_evidence': evidence,
        'paths': {k: str(getattr(args, k).resolve()) for k in ('a_reference', 'c_cache', 'b_reference', 'model', 'output')},
        'A': av['selected_A'], 'B': bs['B'], 'previous_C': crecord['previous_C'],
        'baselines': av['baselines'], 'matched_complete_controls': av['matched_complete_controls'],
        'dev_questions_sha256': arecord['dev_questions_sha256'], 'development_questions': 256,
        'penalties': crecord['penalties'], 'thresholds': crecord['thresholds'], 'utilities': crecord['utilities'],
        'candidate_count': 26, 'math_settings': crecord['math_settings'],
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False, 'fresh_validation': False,
        'state_modes': [16, 8], 'maximum_solver_continuations': 2,
        'selection': 'Keep previous C; greatest accuracy among candidates cheaper than strongest matched B, then lowest compute',
        'known_headroom_limit': 'Current cached answer union cannot reach requested C target; incremental study only'}
    return bundle, adata, values, model, record


def dense_cache(data, values):
    old = data[3]; metadata = data[-1]['metadata']; cls = a_source.previous.CachedDense
    observer = cls(old.directory, old.temperature)
    observer._recipe = old._recipe; observer._retriever = old._retriever; observer._backgrounds = old._backgrounds
    observer.expected = {(v['id'], v['path']): v['input_sha256'] for v in metadata}
    observer.saved = {(v['id'], v['path']): (*[values[k][i] for k in ('state', 'plain', 'guided')], int(values['token_slots'][i]))
                      for i, v in enumerate(metadata)}
    observer.joint = {(v['id'], v['path']): values['joint256'][i] for i, v in enumerate(metadata)}
    return observer


def replay(args, bundle, adata, values, model, record):
    rows, _, trees, judges, _, _, caches = bundle[:7]
    deploy = copy.deepcopy(model); model.science.classifier = dense_cache(adata, values)
    candidates, all_items, grid = [], {}, []
    with canonical.cached_model(model.math_C, caches):
        old_items = [canonical.previous.raw_runtime.evaluate(r, trees[r['id']], judges, model.math_C,
                    record['previous_C']['settings'], 'C') for r in rows]
        if canonical.previous.measured(old_items) != record['previous_C']['dev']:
            raise ValueError('Retained C no longer reproduces exactly')
        math = {r['id']: item for r, item in zip(rows, old_items) if r['subject'] == 'math'}
        for penalty, threshold in product(record['penalties'], record['thresholds']):
            setting = {'penalty': penalty, 'threshold': threshold}
            full = {'by_subject': {'math': record['math_settings'], 'chemistry': setting, 'physics': setting}}
            items = [policy.evaluate(r, trees[r['id']], judges, model, full, 'C') for r in rows]
            for row, item in zip(rows, items):
                if row['subject'] == 'math':
                    for k in ('path', 'correct', 'decisions', 'compute', 'total_parameter_compute', 'verifier_token_slots', 'solver_tokens', 'judge_tokens'):
                        if item[k] != math[row['id']][k]: raise ValueError('Changed retained math route or costs: '+k)
                if item['solver_calls'] > 3 or item['judge_calls'] > 1: raise ValueError('Exceeded solver/reviewer horizon')
            grid.append((full, items))
        for utility in record['utilities']:
            selected = {s: max(range(21), key=lambda i: (
                (v := canonical.previous.measured([x for r, x in zip(rows, grid[i][1]) if r['subject'] == s]))['accuracy']
                - utility*v['mean_compute']/1000, v['accuracy'], -v['mean_compute'])) for s in ('chemistry', 'physics')}
            settings = {'by_subject': {'math': record['math_settings'], **{s: grid[i][0]['by_subject'][s] for s, i in selected.items()}}}
            items = [grid[0][1][i] if r['subject'] == 'math' else grid[selected[r['subject']]][1][i] for i, r in enumerate(rows)]
            grid.append((settings, items))
    path = args.output/'hybrid.pkl'; io.atomic_bytes(path, pickle.dumps(deploy, protocol=5))
    for index, (settings, items) in enumerate(grid):
        key = 'complete-hybrid-'+str(index); all_items[key] = items
        candidates.append({'key': key, 'model_path': str(path.resolve()), 'model_sha256': io.sha(path),
            'policy_module': 'nonlinear_completion_hybrid_v2', 'settings': settings, 'dev': canonical.previous.measured(items)})
    if len(candidates) != 26: raise ValueError('Incomplete registered grid')
    eligible = [c for c in [record['previous_C'], *candidates] if c['dev']['mean_compute'] < record['B']['dev']['mean_compute']]
    selected = max(eligible, key=lambda c: (c['dev']['accuracy'], -c['dev']['mean_compute']))
    summary = {'schema': SCHEMA, 'selected_C': selected, 'candidates': candidates,
        **{k: record[k] for k in ('A', 'B', 'previous_C', 'baselines', 'matched_complete_controls')},
        'C_above_higher_A_B': selected['dev']['accuracy']-max(record[k]['dev']['accuracy'] for k in ('A', 'B')),
        'C_compute_saved_vs_B': 1-selected['dev']['mean_compute']/record['B']['dev']['mean_compute'],
        'previous_C_exactly_reproduced': True, 'math_exactly_preserved': True, 'development_questions': 256,
        'new_neural_calls': 0, 'new_cloud_calls': 0, 'test_used': False, 'fresh_validation': False}
    io.write(args.output/'summary.json', summary); io.write(args.output/'private-candidate-items.json', all_items)
    io.verify(record['source_files']); io.verify(record['input_evidence'])
    io.write(args.output/'complete.json', {'schema': SCHEMA, 'registration_sha256': args.registration_sha256,
        'artifacts': {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file() and not p.name.startswith('.')}, 'test_used': False})
    print({'selected_C': selected['dev'], 'new_best_correct': max(c['dev']['correct'] for c in candidates),
           'gap': summary['C_above_higher_A_B']}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register', 'replay'))
    for n in ('a_reference', 'c_cache', 'b_reference', 'model', 'output'): p.add_argument('--'+n.replace('_', '-'), type=Path, required=True)
    p.add_argument('--registration-sha256'); args = p.parse_args()
    with c_source.guard.no_test_reads(), a_source.original.guarded.no_model_calls(), threadpool_limits(limits=1):
        bundle = inputs(args); record = bundle[-1]
        if args.phase == 'register':
            args.output.mkdir(parents=True, exist_ok=False); io.write(args.output/'registration.json', record)
            print({'registration_sha256': io.sha(args.output/'registration.json')}); return
        if (not args.registration_sha256 or io.sha(args.output/'registration.json') != args.registration_sha256
                or io.read(args.output/'registration.json') != record): raise ValueError('Changed registered hybrid replay')
        with io.lock(args.output/'.run.lock'):
            if any(p.name not in ('registration.json', '.run.lock') for p in args.output.iterdir()): raise ValueError('Preserve prior replay')
            replay(args, *bundle)


if __name__ == '__main__': main()
