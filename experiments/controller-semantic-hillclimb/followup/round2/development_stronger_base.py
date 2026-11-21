"""Develop C with the strongest completed requested A and fixed mentor dynamics."""

import argparse
import importlib
from itertools import product
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from banks import canonical_review
from difficulty_pomdp import MissingObservation
from long_review_pilot import review
from reasoning_pilot import short_repair
from transfer_variants import actor
from transport import resilient
from study import fixed
from study_common import BASELINES, statistics
import extended_acquire as old
import mentor_pairs
from prepare_mixed_mentor import sha
from development_mixed_mentor import development_view, select
from prospective_freeze import check_development_identity, check_no_test_used, observer_slots, best
import stronger_base_hybrid as semantic

APP = 'srr-followup-solver-20261120'


def read(path):
    return json.loads(Path(path).read_bytes())


def wait(path, allowed):
    while not path.exists():
        if not allowed:
            raise ValueError('Complete the registered prerequisite or use --wait: ' + str(path))
        time.sleep(30)
    return read(path)


def resolve_a(path, registry, panel, seen=None):
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        raise ValueError('Cyclic retained-A provenance')
    seen.add(path)
    summary = read(path); identity = check_development_identity(path.parent, panel)
    check_no_test_used(summary)
    candidate = summary['selected_A']
    if candidate.get('source_summary'):
        source = Path(candidate['source_summary'])
        if sha(source) != candidate['source_summary_sha256']:
            raise ValueError('Retained A summary changed')
        return resolve_a(source, registry, panel, seen)
    module = candidate.get('policy_module')
    if module is None:
        if candidate.get('pickle') == 'selected-A.pkl':
            module = 'subject_science_pomdp'
        elif identity.get('policy_sha256') == sha(importlib.import_module('continuous_repairability').__file__):
            module = 'continuous_repairability'
        else:
            module = 'repairability_pomdp'
    if module not in ('repairability_pomdp', 'continuous_repairability', 'subject_science_pomdp', 'science_fusion_pomdp'):
        raise ValueError('Unsupported A role')
    if 'model_path' in candidate:
        model_path = Path(candidate['model_path'])
    else:
        filename = candidate.get('pickle') or candidate['model'] + '.pkl'
        model_path = path.parent / filename
        if not model_path.exists():
            digest = identity.get('model_sha256', {}).get(filename)
            matches = [p for p in registry.rglob(filename) if digest is not None and sha(p) == digest]
            if not matches:
                raise ValueError('Cannot resolve recorded A model assets')
            model_path = sorted(matches)[0]
    expected = identity.get('model_sha256', {}).get(model_path.name)
    if expected is not None and sha(model_path) != expected:
        raise ValueError('A model differs from its development identity')
    return {'candidate': candidate, 'module': module, 'summary': str(path), 'summary_sha256': sha(path),
            'identity_sha256': sha(path.parent / 'identity.json'), 'model_path': str(model_path.resolve()),
            'model_sha256': sha(model_path)}


def seed_all(args, panel):
    """Copy every compatible response and old-observer cache without scoring."""
    source = args.seed_development
    summary = wait(source / 'summary.json', args.wait)
    identity = check_development_identity(source, panel)
    if identity['models_sha256'] != sha(args.mentor_models / 'complete.json'):
        raise ValueError('Seed responses/cache must come from these completed parent mentor candidates')
    check_no_test_used(summary)
    bank = source / 'dev-observations.jsonl'
    if summary['bank_sha256'] != sha(bank):
        raise ValueError('Completed source C bank changed')
    development_view(args.data, bank)  # Replay prompt/seed/cap validation.
    target = args.output / 'dev-observations.jsonl'
    if not target.exists():
        shutil.copy2(bank, target)
    elif not target.read_bytes().startswith(bank.read_bytes()):
        raise ValueError('The seed bank prefix changed')
    receipt = {'source_summary_sha256': sha(source / 'summary.json'), 'source_bank_sha256': sha(bank),
               'source_bytes': bank.stat().st_size, 'selection': 'All validated source receipts; no outcome filtering'}
    source_cache = source / 'private-neural-cache.pkl'
    cache = pickle.loads(source_cache.read_bytes()) if source_cache.exists() else {}
    if source_cache.exists():
        receipt['old_observer_cache_sha256'] = sha(source_cache)
    (args.output / 'seed-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return cache


def run(args):
    panel = sha(args.data / 'splits/dev-questions.jsonl')
    for path in args.a_summary:
        wait(path, args.wait)
    parent = wait(args.mentor_models / 'complete.json', args.wait)
    requested = [resolve_a(p, args.data.parent, panel) for p in args.a_summary]
    chosen = max(requested, key=lambda r: (r['candidate']['dev']['accuracy'], -r['candidate']['dev']['mean_compute']))
    args.output.mkdir(parents=True, exist_ok=True)
    model_folder = args.output / 'models'; model_folder.mkdir(exist_ok=True)
    fitting = {'source_sha256': sha(__file__), 'hybrid_sha256': sha(semantic.__file__),
               'parent_mentor_directory': str(args.mentor_models.resolve()),
               'stronger_base_parent_complete_sha256': sha(args.mentor_models / 'complete.json'),
               'parent_identity_sha256': sha(args.mentor_models / 'identity.json'),
               'base_selection': chosen, 'requested_A': requested, 'development_or_test_used_for_mentor_refit': False,
               'mentor_refitted': False, 'transfer_refitted': False,
               'selection_note': 'Base chosen only from completed same-panel development; mentor dynamics and training-bit transfer unchanged.',
               'test_used': False}
    fit_path = model_folder / 'identity.json'
    if fit_path.exists() and read(fit_path) != fitting:
        raise ValueError('Changed stronger-base construction inputs')
    fit_path.write_text(json.dumps(fitting, indent=2) + '\n')
    constructed = []
    for spec in parent['models']:
        path = args.mentor_models / (spec['key'] + '.pkl')
        if sha(path) != spec['sha256']:
            raise ValueError('Parent mentor model changed')
        base = pickle.loads(Path(chosen['model_path']).read_bytes())
        prior = pickle.loads(path.read_bytes())
        model = semantic.build(base, chosen['module'], chosen['candidate'].get('settings', {}), prior,
                               {'construction_identity_sha256': sha(fit_path), 'parent_model_sha256': spec['sha256']})
        key = 'strongbase-' + spec['key']; output = model_folder / (key + '.pkl')
        output.write_bytes(pickle.dumps(model, protocol=5))
        constructed.append({'key': key, 'sha256': sha(output), 'parent_sha256': spec['sha256'],
                            'smoothing': spec['smoothing'], 'semantic_review': spec['semantic_review']})
    completed = {'models': constructed, 'identity_sha256': sha(fit_path), 'base_A': chosen['candidate'],
                 'parent_mentor_complete_sha256': sha(args.mentor_models / 'complete.json'),
                 'mentor_refitted': False, 'test_used': False}
    (model_folder / 'complete.json').write_text(json.dumps(completed, indent=2) + '\n')
    identity = {'source_sha256': sha(__file__), 'hybrid_sha256': sha(semantic.__file__),
                'models_sha256': sha(model_folder / 'complete.json'), 'models_directory': str(model_folder.resolve()),
                'dev_questions_sha256': panel, 'test_used': False, 'base_selection': chosen,
                'seed_development': str(args.seed_development.resolve()),
                'B_references': [str(p.resolve()) for p in args.b_summary],
                'penalties': [0., .02, .04, .08, .12, .18, .24], 'thresholds': [0., .03, .08],
                'maximum_cloud_gpus': 2, 'protocol': 'mixed-strong-root'}
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Changed stronger-base development protocol')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    seeded_cache = seed_all(args, panel)
    bank = args.output / 'dev-observations.jsonl'
    if bank.stat().st_size and not bank.read_bytes().endswith(b'\n'):
        raise ValueError('Preserve and repair the partial last development receipt before resuming')
    rows, trees, judges = development_view(args.data, bank)
    if len(rows) != 256:
        raise ValueError('Use all256 registered development questions')
    request = None
    if args.collect:
        import modal
        request = resilient(modal.Cls.from_name(APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
        tags = {r['name']: r['digest'] for r in request('/api/tags')['models']}
        if tags.get(old.MODEL) != old.MODEL_DIGEST or tags.get(old.JUDGE) != old.JUDGE_DIGEST:
            raise ValueError('Changed frozen inference models')
    pool, candidates, additions = {}, [], 0
    cache_path = args.output / 'private-neural-caches.pkl'
    caches = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    with bank.open('a') as stream:
        import fcntl
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def acquire(row, missing):
            nonlocal additions
            qid, kind, path = row['id'], missing.kind, missing.path
            safe, tree = {**row, 'answer': ''}, trees[qid]
            if kind == 'solver' and path == 'guided/retry':
                node = tree['repair/retry']
            elif request is None:
                raise RuntimeError('Missing paid development response; use --collect') from missing
            elif kind == 'judge' and not path:
                node = canonical_review(review(request, safe, tree[''], 1024, 512))
            elif kind == 'solver' and path in ('guided', 'guided/reuse', 'retry/reuse'):
                ancestor = '' if path == 'guided' else path.rsplit('/', 1)[0]
                node = actor(request, safe, short_repair(safe, tree[ancestor], judges[qid, '']),
                             mentor_pairs.seed_for(qid, path), 128)
            else:
                raise ValueError('Unexpected stronger hybrid response request')
            stream.write(json.dumps({'id': qid, 'kind': kind, 'path': path, 'response': node,
                                     'exact_control_reused': kind == 'solver' and path == 'guided/retry'}) + '\n')
            stream.flush()
            if kind == 'judge':
                judges[qid, path] = node
            else:
                tree[path] = node
            additions += 1

        def evaluate(model, settings):
            values = []
            for row in rows:
                while True:
                    try:
                        values.append(semantic.evaluate(row, trees[row['id']], judges, model, settings, 'C'))
                        break
                    except MissingObservation as missing:
                        acquire(row, missing)
            return values

        for spec in constructed:
            model = pickle.loads((model_folder / (spec['key'] + '.pkl')).read_bytes())
            old_recipe = semantic.asset_id(model.mentor.classifier)
            for _, observer in observer_slots(model):
                observer.device = args.device
                recipe = semantic.asset_id(observer)
                observer._cache = caches.get(recipe, seeded_cache if recipe == old_recipe else {})
            model._asset_pool = pool
            subject = {s: [] for s in ('math', 'chemistry', 'physics')}
            for penalty, threshold in product(identity['penalties'], identity['thresholds']):
                settings = {'penalty': penalty, 'threshold': threshold}
                values = evaluate(model, settings)
                candidate = {'model': spec['key'], 'policy_module': 'stronger_base_hybrid',
                             'settings': settings, 'dev': statistics(values)}
                candidate['dev']['mean_judge_physical_calls'] = sum(v['judge_physical_calls'] for v in values) / len(values)
                candidates.append(candidate)
                for name in subject:
                    subset = [v for r, v in zip(rows, values) if r['subject'] == name]
                    subject[name].append({'settings': settings, 'dev': statistics(subset)})
            for utility in (0., .02, .04, .08, .16):
                settings = {'by_subject': {s: max(v, key=lambda r: (r['dev']['accuracy'] - utility * r['dev']['mean_compute'] / 1000,
                                                                   r['dev']['accuracy'], -r['dev']['mean_compute']))['settings'] for s, v in subject.items()}}
                values = evaluate(model, settings)
                metrics = statistics(values)
                metrics['mean_judge_physical_calls'] = sum(v['judge_physical_calls'] for v in values) / len(values)
                candidates.append({'model': spec['key'], 'policy_module': 'stronger_base_hybrid', 'settings': settings,
                                   'selection_utility_penalty': utility, 'dev': metrics})
            for observer in pool.values():
                caches[semantic.asset_id(observer)] = observer._cache
            cache_path.write_bytes(pickle.dumps(caches, protocol=5))
            (args.output / 'search.json').write_text(json.dumps(candidates, indent=2) + '\n')
            print('Completed stronger-base C', spec['key'], json.dumps(best(candidates)), flush=True)
    # Finish all C acquisition before waiting on independent ordinary+matched B.
    references, saw_ordinary, saw_matched = [], False, False
    for path in args.b_summary:
        value = wait(path, args.wait)
        reference_identity = check_development_identity(path.parent, panel); check_no_test_used(value)
        saw_matched |= 'ordinary_B_source' in reference_identity and 'root_stop_rules' in reference_identity
        saw_ordinary |= reference_identity.get('source_sha256') == sha(Path(__file__).with_name('development_b.py'))
        references.append({**value['B'], 'source_summary': str(path.resolve()), 'source_summary_sha256': sha(path)})
    if not saw_ordinary or not saw_matched:
        raise ValueError('Supply both completed ordinary B and matched full-review B summaries')
    b = best(references)
    baselines = {name: statistics([fixed(r, trees[r['id']], *parameters) for r in rows]) for name, parameters in BASELINES.items()}
    summary = {'development_questions': len(rows), 'baselines': baselines, **select(candidates, b['dev']['mean_compute']),
               'A': chosen['candidate'], 'B': b, 'B_references': references, 'base_selection': chosen,
               'test_used': False, 'bank_sha256': sha(bank), 'new_development_nodes': additions,
               'policy_module': 'stronger_base_hybrid', 'models_directory': str(model_folder.resolve())}
    c = summary['selected_C']
    if c is not None:
        summary['C_above_higher_A_B'] = c['dev']['accuracy'] - max(chosen['candidate']['dev']['accuracy'], b['dev']['accuracy'])
        summary['C_plus_10_points'] = summary['C_above_higher_A_B'] >= .1 - 1e-12
        summary['C_compute_saved_vs_B'] = 1. - c['dev']['mean_compute'] / b['dev']['mean_compute']
    else:
        summary['C_plus_10_points'] = False
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'mentor-models', 'seed-development', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--a-summary', type=Path, nargs='+', required=True)
    p.add_argument('--b-summary', type=Path, nargs='+', required=True)
    p.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    p.add_argument('--wait', action='store_true'); p.add_argument('--collect', action='store_true')
    run(p.parse_args())
