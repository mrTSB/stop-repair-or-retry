"""Same256-DEV replay of six expanded Cs, preserving strongest A/B/prior C.

Explicit registration precedes execution. The fixed21+5 grid gives156
candidates. Only an explicit --collect registration can acquire a genuinely
missing paid development node; no test record is opened by this driver.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
import copy
import fcntl
import hashlib
from itertools import product
from pathlib import Path
import pickle
import sys
import threading

import numpy as np

import expanded_development_models as builder
import expanded_mentor_io as io
import fit_expanded_mentor as fitting
import normalize_expanded_metadata as normalized
import development_stronger_metadata as previous
import metadata_review_hybrid as metadata_runtime
import stronger_base_hybrid as raw_runtime
import expanded_plain_feature_cache as feature_cache

inherited, common, study = previous.inherited, previous.previous, previous.study
SCHEMA = 'round2-expanded-mentor-development-v1'
PENALTIES, THRESHOLDS, UTILITIES = previous.PENALTIES, previous.THRESHOLDS, previous.UTILITIES
REQUIRED_PRIOR_C = {'development-mentor-newsource-v1', 'development-mentor-newsource-crossfit-v2'}


def sealed_source(directory, panel, plain_complete_sha256):
    payload, receipt = inherited.completed_bank(directory, panel)
    identity = io.read(directory/'identity.json')
    if identity['source_sha256'] != io.sha(previous.__file__):
        raise ValueError('Import the completed strongest-A metadata development bank')
    if identity.get('ordinary_plain_completion_sha256') != plain_complete_sha256:
        raise ValueError('Source cache ancestor context differs from the original plain grid')
    marker = directory/'stronger-metadata-replay-complete.json'
    if not marker.exists() or io.read(marker) != previous.seal_payload(directory):
        raise ValueError('Require the immutable strongest-A metadata bank/cache seal')
    models_directory = directory/'models'; complete = io.read(models_directory/'complete.json')
    if identity['models_sha256'] != io.sha(models_directory/'complete.json'):
        raise ValueError('Source bank/model identity changed')
    for record in complete['models']:
        path = models_directory/(record['key']+'.pkl')
        if io.sha(path) != record['sha256']: raise ValueError('Changed source model')
    for asset in complete['observer_assets']:
        io.verify({asset['recipe_path']: asset['recipe_sha256'], **asset['weights']})
    receipt.update(stronger_metadata_seal_sha256=io.sha(marker),
        model_complete_sha256=io.sha(models_directory/'complete.json'),
        heterogeneous_cache_sha256=io.sha(directory/'private-neural-caches.pkl'))
    return payload, receipt, complete


def prior_reference(summary_path, models_directory, panel, baselines):
    directory = summary_path.parent
    _, bank = inherited.completed_bank(directory, panel)
    summary, identity = io.read(summary_path), io.read(directory/'identity.json')
    source = identity['source_sha256']
    if source == io.sha(previous.__file__):
        if io.read(directory/'stronger-metadata-replay-complete.json') != previous.seal_payload(directory):
            raise ValueError('Unsealed prior strongest-A C')
        module = 'metadata_review_hybrid'
    elif source == io.sha(study.__file__):
        if identity.get('review') != {'thinking_cap': 1024, 'answer_cap': 512, 'guided_cap': 128}:
            raise ValueError('Prior C used a different physical review/solver protocol')
        module = 'hybrid_repairability'
    else:
        raise ValueError('Unsupported prior C protocol; require an explicit audited source')
    if summary['baselines'] != baselines or identity['models_sha256'] != io.sha(models_directory/'complete.json'):
        raise ValueError('Prior C changed panel controls or fitted models')
    complete = io.read(models_directory/'complete.json')
    specs = {r['key']: r for r in complete['models']}
    candidate = summary.get('selected_C')
    if candidate is None: return None, bank
    spec = specs[candidate['model']]; path = models_directory/(spec['key']+'.pkl')
    if io.sha(path) != spec['sha256']: raise ValueError('Prior C model changed')
    return {**candidate, 'model_path': str(path.resolve()), 'model_sha256': spec['sha256'],
        'policy_module': module, 'retained_from_previous': True, 'source_summary': str(summary_path.resolve()),
        'source_summary_sha256': io.sha(summary_path),
        'inherited_base_model_sha256': identity.get('base_selection', {}).get('model_sha256'),
        'inherited_base_settings': identity.get('base_selection', {}).get('candidate', {}).get('settings')}, {**bank, 'model_path': str(path.resolve()),
        'model_sha256': spec['sha256'], 'model_complete_sha256': io.sha(models_directory/'complete.json')}


def merge_plain_features(caches, asset, converted, receipt):
    if (len(converted) != 768 or receipt['converted_nodes'] != 768
            or receipt['overlap_with_shared_cache'] != receipt['bitwise_identical_overlap']
            or receipt['max_absolute_logit_difference'] != 0.):
        raise ValueError('C augmentation requires the whole768 cache and bitwise-identical shared overlap')
    if asset not in caches: raise ValueError('Augment only an existing matching observer recipe/class')
    previous.merge_imported(caches, asset, converted)


def augment_plain_features(args, panel, source_models, caches):
    if args.plain_features is None: return None
    path = args.source/'models'/(source_models['models'][0]['key']+'.pkl')
    model = pickle.loads(path.read_bytes()); observer = model.mentor.classifier
    if type(observer) is not feature_cache.original.perception.Observer:
        raise ValueError('Whole768 source may augment only the exact old-V1 observer class')
    asset = previous.share_key(observer)
    if asset not in caches: raise ValueError('Missing matching old-V1 source cache')
    recipe = io.read(Path(observer.directory)/'recipe.json')
    rows, trees = study.plain_data(args.data, 'dev')
    converted, receipt = feature_cache.convert(args.plain_features, observer, recipe, rows, trees,
        caches[asset], panel, io.sha(args.data/'dev-plain-complete.json'))
    merge_plain_features(caches, asset, converted, receipt)
    return {**receipt, 'cache_asset': list(asset), 'all_other_assets_preserved': True,
            'runtime_compute_charge_unchanged': True}


def contract(args):
    panel = io.sha(args.data/'splits/dev-questions.jsonl')
    fitted = normalized.verify_complete(args.models)
    if (fitted['paired_review_questions'], fitted['metadata_records'], len(fitted['models'])) != (320, 448, 6):
        raise ValueError('Only the completed expanded320/448 six-candidate fit is supported')
    chosen, refs, pending = previous.references(args.registry, panel, False)
    _, source_receipt, source_models = sealed_source(args.source, panel, io.sha(args.data/'dev-plain-complete.json'))
    source_caches = previous.cache_values((args.source/'private-neural-caches.pkl').read_bytes())
    augmented = augment_plain_features(args, panel, source_models, source_caches)
    b = io.read(args.b_reference)
    inherited.validate_b(b, io.read(args.b_reference.parent/'identity.json'), panel)
    requested = list(args.prior_c) + [(args.source/'summary.json', args.source/'models')]
    if not REQUIRED_PRIOR_C <= {path.parent.name for path, _ in requested}:
        raise ValueError('Retain both original and crossfit new-source C references explicitly')
    prior, evidence = [], []
    for path, models in requested:
        candidate, receipt = prior_reference(path, models, panel, b['baselines'])
        evidence.append(receipt)
        if candidate is not None: prior.append(candidate)
    return {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'paths': {name: str(getattr(args, name).resolve()) for name in ('data', 'registry', 'models', 'source', 'b_reference', 'output')},
        'prior_c_paths': [[str(p.resolve()), str(m.resolve())] for p, m in args.prior_c],
        'dev_questions_sha256': panel, 'development_questions': 256,
        'normalized_fit_complete_sha256': io.sha(args.models/'complete.json'),
        'base_selection': chosen, 'A_references': refs, 'pending_optional_A': pending,
        'source_receipt': source_receipt, 'B_reference_sha256': io.sha(args.b_reference),
        'plain_feature_source': str(args.plain_features.resolve()) if args.plain_features is not None else None,
        'plain_feature_augmentation': augmented,
        'B_identity_sha256': io.sha(args.b_reference.parent/'identity.json'),
        'prior_C': prior, 'prior_C_evidence': evidence,
        'penalties': PENALTIES, 'thresholds': THRESHOLDS, 'subject_utilities': UTILITIES,
        'model_count': 6, 'candidate_count': 6*(len(PENALTIES)*len(THRESHOLDS)+len(UTILITIES)),
        'device': args.device, 'allow_missing_development_acquisition': args.collect,
        'maximum_cloud_gpus': 2, 'maximum_request_slots': 2,
        'test_used': False, 'final_gap_claim': False}


def register(args):
    if args.output.exists(): raise ValueError('Use a new expanded development directory')
    record = contract(args)
    args.output.mkdir(parents=True); io.write(args.output/'registration.json', record)
    print({'registered': True, 'registration_sha256': io.sha(args.output/'registration.json')}, flush=True)


def validate_registration(args):
    record = io.read(args.output/'registration.json')
    if io.sha(args.output/'registration.json') != args.registration_sha256:
        raise ValueError('Explicit development registration SHA changed')
    io.verify(record['source_files'])
    if record['plain_feature_source'] != (str(args.plain_features.resolve()) if args.plain_features is not None else None):
        raise ValueError('Changed registered C whole768 cache augmentation')
    if record['plain_feature_augmentation'] is not None: io.verify(record['plain_feature_augmentation']['files'])
    if record['paths'] != {name: str(getattr(args, name).resolve()) for name in record['paths']}:
        raise ValueError('Registered development paths changed')
    if (record['device'] != args.device or record['allow_missing_development_acquisition'] != args.collect
        or record['dev_questions_sha256'] != io.sha(args.data/'splits/dev-questions.jsonl')
        or record['normalized_fit_complete_sha256'] != io.sha(args.models/'complete.json')
        or record['B_reference_sha256'] != io.sha(args.b_reference)
        or record['B_identity_sha256'] != io.sha(args.b_reference.parent/'identity.json')
        or record['prior_c_paths'] != [[str(p.resolve()), str(m.resolve())] for p, m in args.prior_c]):
        raise ValueError('Registered development inputs/settings changed')
    for ref in record['A_references']:
        if io.sha(ref['summary']) != ref['summary_sha256']: raise ValueError('Completed A reference changed')
    for candidate in record['prior_C']:
        io.verify({candidate['source_summary']: candidate['source_summary_sha256'],
                   candidate['model_path']: candidate['model_sha256']})
    for evidence in record['prior_C_evidence']:
        directory = Path(evidence['directory'])
        io.verify({str(directory/'summary.json'): evidence['summary_sha256'],
                   str(directory/'identity.json'): evidence['identity_sha256'],
                   str(directory/'dev-observations.jsonl'): evidence['bank_sha256']})
        if 'model_path' in evidence:
            io.verify({str(Path(evidence['model_path']).parent/'complete.json'): evidence['model_complete_sha256']})
    if sealed_source(args.source, record['dev_questions_sha256'], io.sha(args.data/'dev-plain-complete.json'))[1] != record['source_receipt']:
        raise ValueError('Registered sealed source bank/cache changed')
    return record


def dispatch(model, module, row, tree, judges, settings):
    if module == 'metadata_review_hybrid':
        if not isinstance(model, metadata_runtime.Planner): raise ValueError('Wrong metadata runtime class')
        return metadata_runtime.evaluate(row, tree, judges, model, settings, 'C')
    if module == 'stronger_base_hybrid':
        if isinstance(model, metadata_runtime.Planner) or not isinstance(model, raw_runtime.Planner):
            raise ValueError('Never send metadata C through the semantic runtime')
        return raw_runtime.evaluate(row, tree, judges, model, settings, 'C')
    raise ValueError('Unsupported expanded C runtime')


def evaluate(model, module, settings, rows, trees, judges, acquisition, policy_lock):
    def one(row):
        while True:
            try:
                with policy_lock:
                    return dispatch(model, module, row, trees[row['id']], judges, settings)
            except study.MissingObservation as missing:
                acquisition.acquire(row, missing)
    with ThreadPoolExecutor(max_workers=2) as executor: return list(executor.map(one, rows))


def measured(items):
    result = study.statistics(items)
    result.update(mean_metadata_calls=float(np.mean([r.get('metadata_calls', 0) for r in items])),
        mean_metadata_cpu_ms=float(1000*np.mean([r.get('metadata_cpu_seconds', 0.) for r in items])),
        mean_judge_physical_calls=float(np.mean([r['judge_physical_calls'] for r in items])),
        mean_review_neural_calls=float(np.mean([sum(c['kind'] == 'review' for c in r['verifier_components']) for r in items])))
    return result


def search_models(specs, resolve, callback, persist):
    candidates = []
    for spec in specs:
        model = resolve(spec); groups = {s: [] for s in ('math', 'chemistry', 'physics')}
        def result(settings):
            items, subjects = callback(model, spec, settings)
            if len(items) != 256 or Counter(subjects) != {'math': 128, 'chemistry': 64, 'physics': 64}:
                raise ValueError('Each candidate requires the entire256-question50/25/25 panel')
            candidate = {'model': spec['key'], 'policy_module': spec['policy_module'],
                'builder_module': builder.__name__, 'smoothing': spec['smoothing'],
                'semantic_review': spec['semantic_review'], 'metadata_review': spec['metadata_review'],
                'settings': settings, 'dev': measured(items), 'retained_from_previous': False}
            return items, subjects, candidate
        for penalty, threshold in product(PENALTIES, THRESHOLDS):
            settings = {'penalty': penalty, 'threshold': threshold}
            items, subjects, candidate = result(settings); candidates.append(candidate)
            for subject in groups:
                groups[subject].append({'settings': settings, 'dev': measured([v for v, s in zip(items, subjects) if s == subject])})
        for utility in UTILITIES:
            settings = {'by_subject': {s: max(group, key=lambda r: (
                r['dev']['accuracy'] - utility*r['dev']['mean_compute']/1000,
                r['dev']['accuracy'], -r['dev']['mean_compute']))['settings'] for s, group in groups.items()}}
            _, _, candidate = result(settings); candidate['selection_utility_penalty'] = utility
            candidates.append(candidate)
        persist(spec, model, candidates)
    if len(candidates) != 156: raise ValueError('Incomplete registered six-model156-candidate grid')
    return candidates


def summary(candidates, registered, final_a, baselines, b):
    own = study.select(candidates, b['B']['dev']['mean_compute'])
    retained = [*candidates, *registered['prior_C']]
    result = previous.final_summary(retained, registered['base_selection'], final_a, baselines, b)
    result.update(schema=SCHEMA, selected_expanded_C=own['selected_C'],
        retained_previous_C=bool(result['selected_C'] and result['selected_C'].get('retained_from_previous')),
        policy_module='explicit per candidate', builder_module=builder.__name__,
        expanded_training_questions=512, expanded_reviewed_roots=320, expanded_metadata_records=448,
        complete_new_candidates=len(candidates), prior_C=registered['prior_C'],
        test_used=False, final_gap_claim=False)
    selected = result['selected_C']
    if selected and selected.get('retained_from_previous'):
        result['inherited_base_is_strongest_completed_A'] = (
            selected.get('inherited_base_model_sha256') == final_a['model_sha256']
            and selected.get('inherited_base_settings') == final_a['candidate'].get('settings', {}))
        result['development_gaps_met'] &= result['inherited_base_is_strongest_completed_A']
    return result


def seal_payload(directory):
    names = ['registration.json', 'identity.json', 'cache-import.json', 'summary.json', 'search.json',
        'dev-observations.jsonl', 'raw-generations.jsonl', 'private-neural-cache.pkl',
        'private-neural-caches.pkl', 'models/identity.json', 'models/complete.json']
    if (directory/'transport-errors.jsonl').exists(): names.append('transport-errors.jsonl')
    return {'schema': SCHEMA, 'source_sha256': io.sha(__file__),
            'files': {name: io.sha(directory/name) for name in names}}


def run(args):
    record = validate_registration(args)
    with io.lock(args.output/'.replay.lock'):
        marker = args.output/'expanded-replay-complete.json'
        if marker.exists():
            if io.read(marker) != seal_payload(args.output): raise ValueError('Changed completed expanded replay')
            return
        panel = record['dev_questions_sha256']
        responses, source_receipt, source_models = sealed_source(args.source, panel, io.sha(args.data/'dev-plain-complete.json'))
        if source_receipt != record['source_receipt']: raise ValueError('Registered whole source bank changed')
        raw = (args.source/'raw-generations.jsonl').read_bytes()
        raw_cache = (args.source/'private-neural-cache.pkl').read_bytes()
        source_caches = previous.cache_values((args.source/'private-neural-caches.pkl').read_bytes())
        augmentation = augment_plain_features(args, panel, source_models, source_caches)
        if augmentation != record['plain_feature_augmentation']:
            raise ValueError('Registered C augmentation inputs or overlap changed')
        receipt = {'schema': SCHEMA, 'source_receipt': source_receipt,
            'neural_cache_sha256': hashlib.sha256(raw_cache).hexdigest(),
            'raw_import_sha256': hashlib.sha256(raw).hexdigest(), 'dev_questions_sha256': panel,
            'registration_sha256': args.registration_sha256, 'selection_by_correctness': False}
        common.import_prefix(args.output, responses, raw_cache, raw, receipt)
        completed = builder.construct(args.output, args.models, record['base_selection'],
            record['A_references'], record['pending_optional_A'], args.registration_sha256)
        identity = {'schema': SCHEMA, 'source_sha256': io.sha(__file__), 'builder_module': builder.__name__,
            'models_directory': str((args.output/'models').resolve()),
            'models_sha256': io.sha(args.output/'models/complete.json'), 'dev_questions_sha256': panel,
            'registration_sha256': args.registration_sha256, 'base_selection': record['base_selection'],
            'ordinary_plain_completion_sha256': io.sha(args.data/'dev-plain-complete.json'),
            'review': {'thinking_cap': 1024, 'answer_cap': 512, 'guided_cap': 128}, 'test_used': False}
        fitting.save_json(args.output/'identity.json', identity)
        caches_path = args.output/'private-neural-caches.pkl'
        caches = previous.cache_values(caches_path.read_bytes()) if caches_path.exists() else {}
        for key, values in source_caches.items(): previous.merge_imported(caches, key, values)
        source_model = pickle.loads((args.source/'models'/(source_models['models'][0]['key']+'.pkl')).read_bytes())
        previous.merge_imported(caches, previous.share_key(source_model.mentor.classifier), inherited.cache_values(raw_cache))
        policy_lock, client_lock = threading.RLock(), threading.Lock()
        pool, client = {}, None
        def request(endpoint, payload=None):
            nonlocal client
            if not args.collect: raise RuntimeError('Missing paid DEV node; --collect requires separate reviewed registration')
            with client_lock:
                if client is None:
                    import modal
                    candidate = study.resilient(modal.Cls.from_name(study.APP, 'Ollama')(), args.output/'transport-errors.jsonl')
                    tags = {r['name']: r['digest'] for r in candidate('/api/tags')['models']}
                    if tags.get(study.old.MODEL) != study.old.MODEL_DIGEST or tags.get(study.old.JUDGE) != study.old.JUDGE_DIGEST:
                        raise ValueError('Changed frozen compact/reviewer assets')
                    client = candidate
            return client(endpoint, payload)
        device_lock = io.lock('/private/tmp/srr-training-mps.lock') if args.device == 'mps' else nullcontext()
        with device_lock, ExitStack() as stack:
            streams = {}
            for name in ('dev-observations.jsonl', 'raw-generations.jsonl'):
                stream = stack.enter_context((args.output/name).open('a'))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); streams[name] = stream
            bank = args.output/'dev-observations.jsonl'
            records = common.strict_records(bank); raw_records = common.strict_records(args.output/'raw-generations.jsonl')
            rows, trees, judges = study.development_view(args.data, bank)
            if len(rows) != 256 or len({r['id'] for r in rows}) != 256:
                raise ValueError('Require the complete256-question development panel')
            common.validate_nodes(rows, trees, judges, records); common.coverage(records, raw_records)
            journal = common.RequestJournal(request, streams['raw-generations.jsonl'], raw_records, slots=2)
            acquisition = common.Acquisition(rows, trees, judges, streams['dev-observations.jsonl'], journal, None)
            def resolve(spec):
                path = args.output/'models'/(spec['key']+'.pkl')
                if io.sha(path) != spec['sha256']: raise ValueError('Changed constructed candidate')
                model = pickle.loads(path.read_bytes()); previous.attach(model, caches, pool, args.device)
                return model
            def callback(model, spec, settings):
                return evaluate(model, spec['policy_module'], settings, rows, trees, judges, acquisition, policy_lock), [r['subject'] for r in rows]
            def save_caches():
                for key, observer in pool.items(): caches[key] = observer._cache
                io.atomic_bytes(caches_path, pickle.dumps(caches, protocol=5))
            def persist(spec, model, candidates):
                for candidate in candidates:
                    path = args.output/'models'/(candidate['model']+'.pkl')
                    candidate.update(model_path=str(path.resolve()), model_sha256=io.sha(path))
                save_caches(); io.write(args.output/'search.json', candidates)
                print({'completed_model': spec['key'], 'candidates': len(candidates)}, flush=True)
            try: candidates = search_models(completed['models'], resolve, callback, persist)
            finally: save_caches()
            final_a, refs, pending = previous.references(args.registry, panel, False)
            b = io.read(args.b_reference)
            inherited.validate_b(b, io.read(args.b_reference.parent/'identity.json'), panel)
            baselines = {name: study.statistics([study.fixed(r, trees[r['id']], *parameters) for r in rows])
                         for name, parameters in study.BASELINES.items()}
            if baselines != b['baselines']: raise ValueError('The five fixed baselines changed')
            result = summary(candidates, record, final_a, baselines, b)
            result.update(A_references=refs, pending_optional_A=pending, bank_sha256=io.sha(bank),
                plain_feature_augmentation=augmentation,
                neural_caches_sha256=io.sha(caches_path), raw_generations_sha256=io.sha(args.output/'raw-generations.jsonl'),
                raw_coverage=common.coverage(common.strict_records(bank), common.strict_records(args.output/'raw-generations.jsonl')),
                new_development_nodes=acquisition.added, new_physical_generations=len(journal.known)-len(raw_records),
                B_reference_sha256=io.sha(args.b_reference),
                cost_accounting='All physical LLM phases and distinct encoder components; metadata CPU time separately')
            validate_registration(args)
            for asset in completed['observer_assets']: io.verify({asset['recipe_path']: asset['recipe_sha256'], **asset['weights']})
            io.write(args.output/'summary.json', result)
        io.write(marker, seal_payload(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run'))
    for name in ('data', 'registry', 'models', 'source', 'b_reference', 'output'):
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--prior-c', type=Path, nargs=2, action='append', required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--plain-features', type=Path, help='Explicit bitwise-compatible whole768 old-V1 cache augmentation')
    parser.add_argument('--collect', action='store_true'); parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    if args.mode == 'register': register(args)
    else:
        if not args.registration_sha256: parser.error('Execution requires reviewed registration SHA')
        run(args)


if __name__ == '__main__': main()
