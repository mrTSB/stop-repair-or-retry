"""Four metadata hybrids inheriting the strongest registered completed A.

Imports the entire sealed metadata bank and compatible raw cache. Optional
missing development acquisition is explicit; no test data is read here.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
from itertools import product
from pathlib import Path
import pickle
import threading

import numpy as np

import development_metadata_review as previous
import development_stronger_base as selection
import stronger_base_hybrid as heterogeneous
import stronger_metadata_review as builder
from prospective_freeze import observer_slots

for _module in ('factorized_repairability', 'nonlinear_repairability',
                'subject_science_pomdp', 'science_fusion_pomdp', 'science_option_observer'):
    importlib.import_module(_module)

read, write, sha = previous.read, previous.write, previous.sha
fitting, study, inherited = previous.fitting, previous.study, previous.inherited
SCHEMA = 'stronger-metadata-development-v1'
REQUIRED_A = ('development-repairability-v1', 'development-continuous-v1',
              'development-factorized-v2', 'development-science-option-v1',
              'development-science-fusion-v1', 'development-refined-observer-v1',
              'development-nonlinear-v1')
OPTIONAL_A = ('development-science-listwise-v3',)
PENALTIES, THRESHOLDS, UTILITIES = previous.PENALTIES, previous.THRESHOLDS, previous.UTILITIES


def references(registry, panel, wait):
    paths = [registry / name / 'summary.json' for name in REQUIRED_A]
    for path in paths:
        selection.wait(path, wait)
    optional = [registry / name / 'summary.json' for name in OPTIONAL_A]
    paths += [path for path in optional if path.exists()]
    resolved = [selection.resolve_a(path, registry, panel) for path in paths]
    winner = max(resolved, key=lambda r: (r['candidate']['dev']['accuracy'],
                                         -r['candidate']['dev']['mean_compute']))
    evidence = [{'summary': str(path.resolve()), 'summary_sha256': sha(path),
                 'resolved': item} for path, item in zip(paths, resolved)]
    return winner, evidence, [str(p.resolve()) for p in optional if not p.exists()]


def register(args):
    expected = {'schema': SCHEMA, 'builder_module': 'stronger_metadata_review',
        'policy_module': 'metadata_review_hybrid', 'source_files': fitting.source_files(),
        'paths': {name: str(getattr(args, name).resolve()) for name in (
            'data', 'registry', 'mixed_models', 'newsource_models', 'source',
            'source_models', 'b_reference', 'output')},
        'dev_questions_sha256': sha(args.data / 'splits/dev-questions.jsonl'),
        'development_questions': 256, 'required_A_families': list(REQUIRED_A),
        'optional_A_families': list(OPTIONAL_A), 'device': args.device,
        'allow_missing_development_acquisition': args.collect,
        'penalties': PENALTIES, 'thresholds': THRESHOLDS, 'subject_utility_penalties': UTILITIES,
        'cohorts': ['mixed', 'newsource'], 'smoothing': [8., 32.],
        'selection': 'Highest accuracy across every completed registered A; lower compute breaks a tie',
        'optional_recheck': 'Retain optional A completed by final selection; disclose if stronger than inherited base',
        'maximum_cloud_requests': 2, 'test_used': False, 'final_gap_claim': False}
    path = args.output / 'registration.json'
    if path.exists():
        if read(path) != expected:
            raise ValueError('Changed registered stronger metadata recipe; use a new output')
    else:
        if any(p.name != '.replay.lock' for p in args.output.iterdir()):
            raise ValueError('Unregistered nonempty output must not be overwritten')
        write(path, expected)
    return expected


def share_key(observer):
    return (heterogeneous.asset_id(observer), type(observer).__module__, type(observer).__name__)


def validate_assets(model):
    assets = {}
    for _, observer in observer_slots(model):
        if observer._model is not None:
            raise ValueError('Construct only from unloaded archived observers')
        key = share_key(observer); path = Path(observer.directory) / 'recipe.json'
        recipe = read(path)
        if key[0] not in model.fit_receipt['observer_recipe_sha256']:
            raise ValueError('Unregistered heterogeneous observer recipe')
        fitting.verify_hashes({str(path.parent / name): digest for name, digest in recipe['weights'].items()})
        assets[key] = {'recipe_path': str(path.resolve()), 'recipe_sha256': key[0],
                       'class_module': key[1], 'class_name': key[2],
                       'weights': {str((path.parent / name).resolve()): value for name, value in recipe['weights'].items()}}
    return assets


def construct(args, chosen, evidence, pending, parents):
    fitting.verify_hashes({chosen['summary']: chosen['summary_sha256'],
        str(Path(chosen['summary']).parent / 'identity.json'): chosen['identity_sha256'],
        chosen['model_path']: chosen['model_sha256']})
    directory = args.output / 'models'; directory.mkdir(exist_ok=True)
    identity = {'schema': SCHEMA, 'builder_module': 'stronger_metadata_review',
        'source_sha256': sha(__file__), 'builder_sha256': sha(builder.__file__),
        'policy_module': 'metadata_review_hybrid', 'policy_sha256': sha(previous.policy.__file__),
        'registration_sha256': sha(args.output / 'registration.json'),
        'base_selection': chosen, 'A_references': evidence, 'pending_optional_A': pending,
        'parent_manifests': {cohort: {'directory': str(path.resolve()), 'sha256': sha(path / 'complete.json')}
                             for cohort, path in parents.items()},
        'mentor_refitted': False, 'transfer_refitted': False, 'metadata_adapter_refitted': False,
        'development_used_for_base_selection': True, 'test_used': False}
    identity_path = directory / 'identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Construction inputs changed; preserve this output and use a new area')
    write(identity_path, identity)
    complete_path = directory / 'complete.json'
    if complete_path.exists():
        complete = read(complete_path)
        if complete['identity_sha256'] != sha(identity_path):
            raise ValueError('Changed completed construction identity')
        for item in complete['models']:
            if sha(directory / (item['key'] + '.pkl')) != item['sha256']:
                raise ValueError('Changed constructed metadata model')
        for item in complete['observer_assets']:
            fitting.verify_hashes({item['recipe_path']: item['recipe_sha256'], **item['weights']})
        return complete
    base = pickle.loads(Path(chosen['model_path']).read_bytes())
    models, all_assets = [], {}
    for cohort, parent_directory in parents.items():
        complete, _ = previous.model_assets(parent_directory)
        if read(parent_directory / 'identity.json')['cohort'] != cohort:
            raise ValueError('Swapped metadata root cohorts')
        for spec in complete['models']:
            parent = pickle.loads((parent_directory / (spec['key'] + '.pkl')).read_bytes())
            model = builder.build(base, chosen['module'], chosen['candidate'].get('settings', {}), parent,
                {'construction_identity_sha256': sha(identity_path), 'base_selection': chosen,
                 'parent_metadata_directory': str(parent_directory.resolve()),
                 'parent_model_sha256': spec['sha256'], 'cohort': cohort})
            all_assets.update(validate_assets(model))
            key = cohort + '-' + spec['key']; path = directory / (key + '.pkl')
            if path.exists():
                # A crash may leave a fully written prior model. It must agree
                # structurally; never overwrite an unexplained model artifact.
                saved = pickle.loads(path.read_bytes())
                if (saved.fit_receipt != model.fit_receipt
                        or builder.retained_state(saved) != builder.retained_state(model)
                        or fitting.preserved_digest(saved) != fitting.preserved_digest(model)):
                    raise ValueError('Changed partially constructed model')
            else:
                previous.atomic_pickle(path, model)
            saved = pickle.loads(path.read_bytes())
            if (builder.retained_state(saved) != builder.retained_state(parent)
                    or fitting.preserved_digest(saved) != fitting.preserved_digest(model)):
                raise ValueError('Saved hybrid changed metadata/mentor/transfer values')
            models.append({'key': key, 'sha256': sha(path), 'parent_metadata_directory': str(parent_directory.resolve()),
                'parent_sha256': spec['sha256'], 'cohort': cohort, 'smoothing': spec['smoothing'],
                'metadata_review': True, 'policy_module': 'metadata_review_hybrid',
                'retained_state_sha256': builder.retained_state(saved),
                'base_and_child_state_sha256': fitting.preserved_digest(saved)})
    complete = {'models': models, 'base_A': chosen['candidate'], 'base_selection': chosen,
        'identity_sha256': sha(identity_path), 'parent_manifests': identity['parent_manifests'],
        'observer_assets': list(all_assets.values()), 'builder_module': 'stronger_metadata_review',
        'policy_module': 'metadata_review_hybrid', 'neural_calls': 0, 'cloud_calls': 0,
        'mentor_refitted': False, 'transfer_refitted': False, 'metadata_adapter_refitted': False,
        'test_used': False}
    write(complete_path, complete)
    return complete


def cache_values(payload):
    value = pickle.loads(payload)
    if not isinstance(value, dict):
        raise ValueError('Malformed heterogeneous observation cache')
    for asset, entries in value.items():
        if (not isinstance(asset, tuple) or len(asset) != 3 or not all(isinstance(v, str) for v in asset)
                or len(asset[0]) != 64 or not isinstance(entries, dict)):
            raise ValueError('Malformed class-aware cache identity')
        option = asset[1:] == ('science_option_observer', 'ScienceOptionObserver')
        expected = 4 if option else 5
        for key, outputs in entries.items():
            if (not isinstance(key, tuple) or len(key) != expected or key[0] not in ('solver', 'review')
                    or not all(isinstance(v, str) for v in key) or len(key[3]) != 64
                    or any(c not in '0123456789abcdef' for c in key[3])):
                raise ValueError('Changed observer class/input-cache semantics')
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
                raise ValueError('Malformed cached observation')
            for values, width in zip(outputs[:3], (4, 8, 8)):
                if np.asarray(values).shape != (width,) or not np.isfinite(values).all():
                    raise ValueError('Nonfinite or malformed neural observation')
            if not isinstance(outputs[3], (int, np.integer)) or not 1 <= outputs[3] <= 512:
                raise ValueError('Invalid actual encoder token slots')
    return value


def merge_imported(caches, key, imported):
    current = caches.setdefault(key, {})
    for item, outputs in imported.items():
        if item in current and any(not np.array_equal(a, b) for a, b in zip(outputs, current[item])):
            raise ValueError('Resumed heterogeneous cache changed an imported observation')
        current[item] = outputs


def attach(model, caches, pool, device):
    for _, observer in observer_slots(model):
        observer.device = device
        observer._cache = caches.setdefault(share_key(observer), {})
    model._asset_pool = pool


def save_caches(path, caches, pool):
    for key, observer in pool.items():
        caches[key] = observer._cache
    previous.atomic_pickle(path, caches)


def seal_payload(output):
    names = ['registration.json', 'identity.json', 'cache-import.json', 'summary.json', 'search.json',
             'dev-observations.jsonl', 'private-neural-cache.pkl', 'private-neural-caches.pkl',
             'raw-generations.jsonl', 'models/identity.json', 'models/complete.json']
    if (output / 'transport-errors.jsonl').exists():
        names.append('transport-errors.jsonl')
    return {'schema': SCHEMA, 'source_sha256': sha(__file__),
            'files': {name: sha(output / name) for name in names}}


def finalize(output):
    path = output / 'stronger-metadata-replay-complete.json'; expected = seal_payload(output)
    if path.exists() and read(path) != expected:
        raise ValueError('Changed completed stronger metadata replay seal')
    write(path, expected)


def final_summary(candidates, chosen, final_a, baselines, b):
    strongest = b['B']; selected = study.select(candidates, strongest['dev']['mean_compute'])
    a = final_a['candidate']; baseline = max(v['accuracy'] for v in baselines.values())
    result = {'development_questions': 256, 'baselines': baselines, **selected,
        'A': a, 'inherited_A': chosen['candidate'], 'base_selection': chosen,
        'final_A_selection': final_a, 'B': strongest, 'ordinary_B': b['ordinary_B'], 'matched_B': b['matched_B'],
        'inherited_base_is_strongest_completed_A': chosen['model_sha256'] == final_a['model_sha256']
            and chosen['candidate'].get('settings', {}) == final_a['candidate'].get('settings', {}),
        'A_above_best_baseline': a['dev']['accuracy'] - baseline,
        'B_above_best_baseline': strongest['dev']['accuracy'] - baseline,
        'C_plus_10_points': False, 'C_compute_below_B': False,
        'test_used': False, 'final_gap_claim': False,
        'policy_module': 'metadata_review_hybrid', 'builder_module': 'stronger_metadata_review'}
    c = selected['selected_C']
    if c is not None:
        margin = c['dev']['accuracy'] - max(a['dev']['accuracy'], strongest['dev']['accuracy'])
        result.update(C_above_higher_A_B=margin, C_plus_10_points=margin >= .1 - 1e-12,
            C_compute_below_B=c['dev']['mean_compute'] < strongest['dev']['mean_compute'],
            C_compute_saved_vs_B=1. - c['dev']['mean_compute'] / strongest['dev']['mean_compute'])
    result['development_gaps_met'] = (result['A_above_best_baseline'] >= .1 - 1e-12
        and result['B_above_best_baseline'] >= .1 - 1e-12 and result['C_plus_10_points']
        and result['C_compute_below_B'] and result['inherited_base_is_strongest_completed_A'])
    return result


def run_locked(args):
    contract = register(args); panel = contract['dev_questions_sha256']
    previous.wait_completed(args.source)
    inherited.wait(args.source / 'metadata-replay-complete.json')
    inherited.wait(args.b_reference)
    for folder in (args.mixed_models, args.newsource_models):
        inherited.wait(folder / 'complete.json')
    chosen, evidence, pending = references(args.registry, panel, True)
    # On resume a newly completed optional A must not rewrite the inherited
    # model identity. It still participates in the final strongest-A comparison.
    constructed_identity = args.output / 'models/identity.json'
    if constructed_identity.exists():
        saved = read(constructed_identity)
        for item in saved['A_references']:
            if sha(item['summary']) != item['summary_sha256']:
                raise ValueError('A completed prerequisite summary changed')
        chosen, evidence, pending = saved['base_selection'], saved['A_references'], saved['pending_optional_A']
    fitting.verify_hashes(contract['source_files'])
    if sha(args.data / 'splits/dev-questions.jsonl') != panel:
        raise ValueError('Changed development question panel')
    responses, source_receipt = previous.completed_bank(args.source, panel)
    source_models, recipe = previous.model_assets(args.source_models)
    source_identity = read(args.source / 'identity.json')
    if (source_identity['models_sha256'] != sha(args.source_models / 'complete.json')
            or source_identity['source_sha256'] != sha(previous.__file__)
            or read(args.source_models / 'identity.json')['cohort'] != 'newsource'):
        raise ValueError('Require the sealed new-source metadata bank and its exact raw cache')
    b = read(args.b_reference); inherited.validate_b(b, read(args.b_reference.parent / 'identity.json'), panel)
    cache_bytes = (args.source / 'private-neural-cache.pkl').read_bytes()
    raw_bytes = (args.source / 'raw-generations.jsonl').read_bytes()
    imported = inherited.cache_values(cache_bytes)
    receipt = {'schema': SCHEMA, 'source_sha256': sha(__file__), 'response_source': source_receipt,
        'neural_cache_sha256': hashlib.sha256(cache_bytes).hexdigest(),
        'raw_import_sha256': hashlib.sha256(raw_bytes).hexdigest(),
        'cache_model_complete_sha256': sha(args.source_models / 'complete.json'),
        'neural_recipe_sha256': recipe, 'B_reference_sha256': sha(args.b_reference),
        'dev_questions_sha256': panel, 'selection_by_correctness': False, 'test_used': False}
    # Must precede model construction: the imported helper deliberately refuses
    # unregistered output contents. It retains every source node and raw receipt.
    if not (args.output / 'cache-import.json').exists() and (args.output / 'models').exists():
        raise ValueError('Missing bank import for already constructed models')
    previous.import_prefix(args.output, responses, cache_bytes, raw_bytes, receipt)
    complete = construct(args, chosen, evidence, pending,
                         {'mixed': args.mixed_models, 'newsource': args.newsource_models})
    model_directory = args.output / 'models'
    identity = {'schema': SCHEMA, 'source_sha256': sha(__file__), 'builder_module': 'stronger_metadata_review',
        'builder_sha256': sha(builder.__file__), 'policy_module': 'metadata_review_hybrid',
        'hybrid_sha256': sha(previous.policy.__file__), 'models_directory': str(model_directory.resolve()),
        'models_sha256': sha(model_directory / 'complete.json'), 'base_selection': chosen,
        'dev_questions_sha256': panel, 'ordinary_plain_completion_sha256': sha(args.data / 'dev-plain-complete.json'),
        'registration_sha256': sha(args.output / 'registration.json'),
        'cache_import_sha256': sha(args.output / 'cache-import.json'),
        'penalties': PENALTIES, 'thresholds': THRESHOLDS, 'subject_utility_penalties': UTILITIES,
        'test_used': False, 'include_reviews': False}
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Changed stronger metadata development identity')
    write(identity_path, identity)
    if (args.output / 'summary.json').exists():
        finalize(args.output)
        print('Verified sealed stronger metadata replay; no inference repeated.', flush=True)
        return
    bank = args.output / 'dev-observations.jsonl'; caches_path = args.output / 'private-neural-caches.pkl'
    caches = cache_values(caches_path.read_bytes()) if caches_path.exists() else {}
    source_spec = source_models['models'][0]
    source_model = pickle.loads((args.source_models / (source_spec['key'] + '.pkl')).read_bytes())
    old_key = share_key(source_model.mentor.classifier)
    if old_key[0] != recipe:
        raise ValueError('Source raw cache observer differs from its registered recipe')
    merge_imported(caches, old_key, imported)
    policy_lock, client_lock = threading.RLock(), threading.Lock()
    pool, client, candidates = {}, None, []

    def request(endpoint, payload=None):
        nonlocal client
        if not args.collect:
            raise RuntimeError('Missing paid development node; explicitly enable --collect after review')
        with client_lock:
            if client is None:
                import modal
                candidate = study.resilient(modal.Cls.from_name(study.APP, 'Ollama')(), args.output / 'transport-errors.jsonl')
                tags = {r['name']: r['digest'] for r in candidate('/api/tags')['models']}
                if tags.get(study.old.MODEL) != study.old.MODEL_DIGEST or tags.get(study.old.JUDGE) != study.old.JUDGE_DIGEST:
                    raise ValueError('Changed frozen compact solver or reviewer')
                client = candidate
        return client(endpoint, payload)

    with ExitStack() as stack:
        streams = {}
        for name in ('dev-observations.jsonl', 'raw-generations.jsonl'):
            stream = stack.enter_context((args.output / name).open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); streams[name] = stream
        records = previous.strict_records(bank)
        raw_records = previous.strict_records(args.output / 'raw-generations.jsonl')
        rows, trees, judges = study.development_view(args.data, bank)
        if len(rows) != 256:
            raise ValueError('Require the complete registered256question development panel')
        previous.validate_nodes(rows, trees, judges, records); previous.coverage(records, raw_records)
        journal = previous.RequestJournal(request, streams['raw-generations.jsonl'], raw_records, slots=2)
        acquire = previous.Acquisition(rows, trees, judges, streams['dev-observations.jsonl'], journal, None)
        for spec in complete['models']:
            path = model_directory / (spec['key'] + '.pkl')
            if sha(path) != spec['sha256']:
                raise ValueError('Changed constructed model before replay')
            model = pickle.loads(path.read_bytes()); attach(model, caches, pool, args.device)
            groups = {subject: [] for subject in ('math', 'chemistry', 'physics')}

            def evaluate(settings):
                values = previous.evaluate(model, settings, rows, trees, judges, acquire, policy_lock)
                return values, {'model': spec['key'], 'model_path': str(path.resolve()),
                    'policy_module': 'metadata_review_hybrid', 'builder_module': 'stronger_metadata_review',
                    'cohort': spec['cohort'], 'smoothing': spec['smoothing'],
                    'settings': settings, 'dev': previous.measured(values)}

            try:
                for penalty, threshold in product(PENALTIES, THRESHOLDS):
                    settings = {'penalty': penalty, 'threshold': threshold}
                    values, candidate = evaluate(settings); candidates.append(candidate)
                    for subject in groups:
                        groups[subject].append({'settings': settings, 'dev': previous.measured(
                            [v for r, v in zip(rows, values) if r['subject'] == subject])})
                for utility in UTILITIES:
                    settings = {'by_subject': {subject: max(group, key=lambda r: (
                        r['dev']['accuracy'] - utility * r['dev']['mean_compute'] / 1000,
                        r['dev']['accuracy'], -r['dev']['mean_compute']))['settings'] for subject, group in groups.items()}}
                    _, candidate = evaluate(settings); candidate['selection_utility_penalty'] = utility
                    candidates.append(candidate)
            finally:
                save_caches(caches_path, caches, pool)
            write(args.output / 'search.json', candidates)
            print('Completed strongest-A metadata C', spec['key'], flush=True)
        final_a, final_refs, pending_final = references(args.registry, panel, False)
        for item in evidence:
            if sha(item['summary']) != item['summary_sha256']:
                raise ValueError('A selection prerequisite changed during development')
        baselines = {name: study.statistics([study.fixed(row, trees[row['id']], *parameters) for row in rows])
                     for name, parameters in study.BASELINES.items()}
        if baselines != b['baselines']:
            raise ValueError('Original fixed baselines changed on the shared panel')
        summary = final_summary(candidates, chosen, final_a, baselines, b)
        summary.update(A_references=final_refs, pending_optional_A=pending_final,
            bank_sha256=sha(bank), neural_caches_sha256=sha(caches_path),
            raw_generations_sha256=sha(args.output / 'raw-generations.jsonl'),
            raw_coverage=previous.coverage(previous.strict_records(bank), previous.strict_records(args.output / 'raw-generations.jsonl')),
            new_development_nodes=acquire.added, new_physical_generations=len(journal.known) - len(raw_records),
            B_reference_sha256=sha(args.b_reference), models_directory=str(model_directory.resolve()),
            B_comparison='Strongest ordinary/matched B, with lower compute breaking an accuracy tie.',
            cost_accounting='Distinct recipe/input encoder passes, all physical LLM phases, metadata CPU timing separate.',
            A_reference_scope='All seven required families plus optional science-v3 completed by final selection.')
        fitting.verify_hashes(contract['source_files'])
        if sha(args.b_reference) != receipt['B_reference_sha256']:
            raise ValueError('B reference changed during replay')
        for item in complete['observer_assets']:
            fitting.verify_hashes({item['recipe_path']: item['recipe_sha256'], **item['weights']})
        write(args.output / 'summary.json', summary)
    finalize(args.output)
    print({'complete': True, 'selected_C': summary['selected_C'],
           'development_gaps_met': summary['development_gaps_met']}, flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.replay.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'registry', 'mixed-models', 'newsource-models', 'source',
                 'source-models', 'b-reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='mps')
    parser.add_argument('--collect', action='store_true', help='Permit only missing development responses through the frozen protocol')
    run(parser.parse_args())
