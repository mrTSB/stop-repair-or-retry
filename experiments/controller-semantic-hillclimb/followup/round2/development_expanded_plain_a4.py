"""Registered pure four-state A replay, using existing paid plain outputs only.

Fourteen fixed settings (seven penalties, discrete/continuous belief updates)
use the completed weighted TRAIN-only regular model. No LLM or encoder calls
are allowed: every required raw neural observation must already be sealed in
the shared development cache. Runtime compute still charges every paid pass.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from itertools import product
from pathlib import Path
import pickle
import sys

import numpy as np

import expanded_mentor_io as io
import fit_expanded_mentor as fitting
import development_stronger_metadata as previous
import prospective_collect as protocol
import dynamic_grounded_pomdp as policy
import expanded_plain_feature_cache as feature_cache
from prepare_mixed_mentor import plain_data

study, inherited = previous.study, previous.inherited
SCHEMA = 'round2-expanded-pure-plain-A4-development-v1'
PENALTIES = previous.PENALTIES
SETTINGS = [{'penalty': p, 'continuous_filter': continuous, 'context_weight': 0.}
            for p, continuous in product(PENALTIES, (False, True))]


def fitted_model(directory):
    completed = fitting.verify_complete(directory, io.sha(directory/'registration.json'))
    record = completed['regular_A4_artifact']; path = directory/(record['key']+'.pkl')
    if record['policy_module'] != 'dynamic_grounded_pomdp' or io.sha(path) != record['sha256']:
        raise ValueError('Changed registered pure-A4 archive')
    model = pickle.loads(path.read_bytes())
    if (not isinstance(model, policy.Planner) or type(model.classifier) is not policy.Observer
            or model.review_observer is not None or model.context_weight != 0.
            or set(model.dynamics) != {'math', 'chemistry', 'physics'}
            or any(np.asarray(d.prior).shape != (4,) for d in model.dynamics.values())
            or model.fit_receipt['training_questions'] != 512
            or model.fit_receipt['guided_in_unreviewed_counts'] is not False
            or model.classifier.calibration is None or model.classifier._model is not None):
        raise ValueError('Require the sealed TRAIN-only four-state plain POMDP')
    recipe_path = Path(model.classifier.directory)/'recipe.json'; recipe = io.read(recipe_path)
    if recipe['input_profile'] != 'full-options-openstax-v1' or recipe['max_length'] != 512:
        raise ValueError('Changed old full observation profile')
    assets = {str(recipe_path.resolve()): io.sha(recipe_path),
              **{str((recipe_path.parent/name).resolve()): digest for name, digest in recipe['weights'].items()}}
    io.verify(assets)
    return model, {**record, 'model_path': str(path.resolve())}, recipe, assets


def cache_source(directory, panel, plain_complete_sha256):
    _, bank = inherited.completed_bank(directory, panel)
    identity = io.read(directory/'identity.json')
    if identity.get('ordinary_plain_completion_sha256') != plain_complete_sha256:
        raise ValueError('Source cache ancestor context differs from the original plain grid')
    if (identity['source_sha256'] != io.sha(previous.__file__)
            or io.read(directory/'stronger-metadata-replay-complete.json') != previous.seal_payload(directory)):
        raise ValueError('Require the sealed strongest-A metadata development cache')
    payload = (directory/'private-neural-caches.pkl').read_bytes()
    caches = previous.cache_values(payload)
    return caches, {**bank, 'seal_sha256': io.sha(directory/'stronger-metadata-replay-complete.json'),
                   'cache_sha256': hashlib.sha256(payload).hexdigest()}


def contract(args):
    panel = io.sha(args.data/'splits/dev-questions.jsonl')
    model, record, recipe, assets = fitted_model(args.models)
    caches, source = cache_source(args.source, panel, io.sha(args.data/'dev-plain-complete.json'))
    key = previous.share_key(model.classifier)
    if key not in caches: raise ValueError('No matching class/recipe observation cache')
    rows, trees = plain_data(args.data, 'dev')
    if args.plain_features is None:
        supplemental = None; complete_cache = caches[key]
    else:
        complete_cache, supplemental = feature_cache.convert(args.plain_features, model.classifier, recipe,
            rows, trees, caches[key], panel, io.sha(args.data/'dev-plain-complete.json'))
    aligned_cache(model.classifier, complete_cache, rows, trees)
    chosen, refs, pending = previous.references(args.registry, panel, False)
    return {'schema': SCHEMA, 'source_files': io.sources(sys.modules[__name__]),
        'paths': {n: str(getattr(args, n).resolve()) for n in ('data', 'models', 'source', 'registry', 'output')},
        'dev_questions_sha256': panel, 'plain_complete_sha256': io.sha(args.data/'dev-plain-complete.json'),
        'fit_complete_sha256': io.sha(args.models/'complete.json'), 'model': record, 'observer_assets': assets,
        'cache_source': source, 'cache_asset': list(key), 'prior_A': chosen,
        'plain_feature_source': str(args.plain_features.resolve()) if args.plain_features is not None else None,
        'plain_feature_conversion': supplemental,
        'A_references': refs, 'pending_optional_A': pending, 'settings': SETTINGS,
        'development_questions': 256, 'candidate_count': 14, 'role': 'A',
        'policy_module': 'dynamic_grounded_pomdp', 'state_count': 4,
        'new_encoder_calls': 0, 'new_cloud_calls': 0, 'logical_review_calls': 0,
        'paid_compute_includes_cached_encoder_passes': True,
        'test_used': False, 'final_gap_claim': False}


def register(args):
    if args.output.exists(): raise ValueError('Use a new pure-A4 development directory')
    record = contract(args); args.output.mkdir(parents=True)
    io.write(args.output/'registration.json', record)
    print({'registered': True, 'registration_sha256': io.sha(args.output/'registration.json')}, flush=True)


def validate_registration(args):
    path = args.output/'registration.json'; record = io.read(path)
    if io.sha(path) != args.registration_sha256: raise ValueError('Explicit pure-A4 registration SHA changed')
    io.verify(record['source_files']); io.verify(record['observer_assets'])
    if record['plain_feature_source'] != (str(args.plain_features.resolve()) if args.plain_features is not None else None):
        raise ValueError('Changed explicitly registered whole768 feature source')
    if record['plain_feature_conversion'] is not None: io.verify(record['plain_feature_conversion']['files'])
    io.verify({record['model']['model_path']: record['model']['sha256']})
    if (record['paths'] != {n: str(getattr(args, n).resolve()) for n in record['paths']}
            or record['dev_questions_sha256'] != io.sha(args.data/'splits/dev-questions.jsonl')
            or record['plain_complete_sha256'] != io.sha(args.data/'dev-plain-complete.json')
            or record['fit_complete_sha256'] != io.sha(args.models/'complete.json')
            or record['settings'] != SETTINGS):
        raise ValueError('Changed pure-A4 registered inputs or grid')
    for ref in record['A_references']:
        resolved = ref['resolved']
        io.verify({ref['summary']: ref['summary_sha256'], resolved['model_path']: resolved['model_sha256'],
            str(Path(ref['summary']).parent/'identity.json'): resolved['identity_sha256']})
    if cache_source(args.source, record['dev_questions_sha256'], io.sha(args.data/'dev-plain-complete.json'))[1] != record['cache_source']:
        raise ValueError('Changed pure-A4 registered cache')
    return record


def aligned_cache(observer, cache, rows, trees):
    """Every root/first child is required regardless of its correctness/validity."""
    if (len(rows) != 256 or len({r['id'] for r in rows}) != 256
            or Counter(r['subject'] for r in rows) != {'math': 128, 'chemistry': 64, 'physics': 64}):
        raise ValueError('Require the entire256-question50/25/25 panel')
    selected, alignment = {}, []
    for row in rows:
        tree = trees[row['id']]
        if set(tree) != set(study.PLAIN): raise ValueError('Only all seven fixed plain branches are allowed')
        safe = {**row, 'answer': ''}
        for path, node in tree.items(): protocol.validate_node(safe, 'plain', 'solver', path, node, tree, {})
        for path in ('', 'repair', 'retry'):
            node = tree[path]; key = observer.key(safe, path, node, 'solver')
            if key not in cache: raise ValueError('Missing precomputed paid observation; no inference authorized')
            selected[key] = cache[key]
            alignment.append({'id': row['id'], 'path': path, 'prompt_sha256': node['prompt_sha256'],
                'text_sha256': hashlib.sha256(node['text'].encode()).hexdigest(),
                'token_slots': int(cache[key][3])})
    if len(selected) != 768: raise ValueError('Incomplete distinct768-node plain cache')
    return selected, alignment


@contextmanager
def cache_only(observer, cache, recipe):
    """Disable both encoder loading and new forwards while retaining actual cost."""
    original = {name: (name in vars(observer), getattr(observer, name)) for name in ('load', 'values', '_cache', '_recipe')}
    def load():
        if observer._model is not None: raise ValueError('No resident neural model is allowed in cache-only replay')
    def values(row, path, tree, judge=None):
        if row.get('answer') or judge is not None or path not in ('', 'repair', 'retry'):
            raise ValueError('Reference/review/terminal observation reached pure-A4 encoder interface')
        key = observer.key(row, path, tree[path], 'solver')
        if key not in cache: raise ValueError('Unpaid or missing observation; new inference is disabled')
        return cache[key]
    observer.load, observer.values, observer._cache, observer._recipe = load, values, cache, recipe
    try: yield
    finally:
        for name, (owned, value) in original.items():
            if owned: setattr(observer, name, value)
            else: delattr(observer, name)


def replay_one(row, tree, model, settings):
    if settings['context_weight'] != 0. or model.review_observer is not None:
        raise ValueError('Pure-A4 cannot use learned transition heads or review observations')
    safe = {**row, 'answer': ''}
    result = policy.evaluate(safe, tree, {}, model, settings, 'A')
    if (result['judge_calls'] != 0 or result['judge_tokens'] != 0 or result['solver_calls'] > 3
            or any(d['queried'] or d['path'] not in ('', 'repair', 'retry') for d in result['decisions'])
            or result['verifier_calls'] != len(result['decisions']) or result['path'] not in study.PLAIN):
        raise ValueError('Pure-A4 exceeded its plain paid-prefix/horizon contract')
    expected_slots = sum(model.classifier.values(safe, d['path'], tree)[3] for d in result['decisions'])
    if result['verifier_token_slots'] != expected_slots:
        raise ValueError('Cached neural passes lost their actual runtime slot charge')
    result['correct'] = int(fitting.core.correct(tree[result['path']], row['answer']))
    return result


def select(candidates, final_a, baselines):
    own = max(candidates, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
    prior = {**final_a['candidate'], 'model_path': final_a['model_path'], 'model_sha256': final_a['model_sha256'],
             'policy_module': final_a['module'], 'retained_from_previous': True,
             'source_summary': final_a['summary'], 'source_summary_sha256': final_a['summary_sha256']}
    winner = max([prior, own], key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))
    highest = max(v['accuracy'] for v in baselines.values())
    return {'schema': SCHEMA, 'development_questions': 256, 'baselines': baselines,
        'selected_A4': own, 'selected_A': winner, 'retained_previous_A': winner.get('retained_from_previous', False),
        'A_above_highest_baseline': winner['dev']['accuracy']-highest,
        'A_plus_10_points': winner['dev']['accuracy'] >= highest+.1-1e-12,
        'new_encoder_calls': 0, 'new_cloud_calls': 0, 'test_used': False, 'final_gap_claim': False,
        'C_bridge_supported': False, 'architecture': 'Four-state hidden difficulty/current correctness POMDP; plain-only dynamics; no LLM'}


def seal_payload(directory):
    names = ('registration.json', 'identity.json', 'cache-alignment.json', 'selected-neural-cache.pkl', 'search.json', 'summary.json')
    return {'schema': SCHEMA, 'source_sha256': io.sha(__file__), 'files': {n: io.sha(directory/n) for n in names}}


def run(args):
    record = validate_registration(args)
    with io.lock(args.output/'.replay.lock'):
        marker = args.output/'plain-A4-replay-complete.json'
        if marker.exists():
            if io.read(marker) != seal_payload(args.output): raise ValueError('Changed completed pure-A4 replay')
            return
        model, spec, recipe, _ = fitted_model(args.models)
        caches, _ = cache_source(args.source, record['dev_questions_sha256'], io.sha(args.data/'dev-plain-complete.json'))
        rows, trees = plain_data(args.data, 'dev')
        source_cache = caches[tuple(record['cache_asset'])]
        if args.plain_features is not None:
            source_cache, conversion = feature_cache.convert(args.plain_features, model.classifier, recipe,
                rows, trees, source_cache, record['dev_questions_sha256'], io.sha(args.data/'dev-plain-complete.json'))
            if conversion != record['plain_feature_conversion']:
                raise ValueError('Registered feature conversion changed')
        cache, alignment = aligned_cache(model.classifier, source_cache, rows, trees)
        fitting.save_json(args.output/'cache-alignment.json', alignment)
        path = args.output/'selected-neural-cache.pkl'
        payload = pickle.dumps(cache, protocol=5)
        if path.exists():
            old = inherited.cache_values(path.read_bytes())
            if set(old) != set(cache) or any(any(not np.array_equal(a, b) for a, b in zip(v, old[k])) for k, v in cache.items()):
                raise ValueError('Changed selected sealed cache on resume')
        else: io.atomic_bytes(path, payload)
        fitting.save_json(args.output/'identity.json', {'schema': SCHEMA, 'source_sha256': io.sha(__file__),
            'registration_sha256': args.registration_sha256, 'model': spec,
            'dev_questions_sha256': record['dev_questions_sha256'], 'cache_sha256': io.sha(path),
            'cache_alignment_sha256': io.sha(args.output/'cache-alignment.json'),
            'new_encoder_calls': 0, 'new_cloud_calls': 0, 'test_used': False})
        candidates = []
        with cache_only(model.classifier, cache, recipe):
            for settings in SETTINGS:
                values = [replay_one(r, trees[r['id']], model, settings) for r in rows]
                if len(values) != 256: raise ValueError('Partial development evaluation')
                candidates.append({'model': spec['key'], 'model_path': spec['model_path'], 'model_sha256': spec['sha256'],
                    'policy_module': 'dynamic_grounded_pomdp', 'settings': settings,
                    'dev': study.statistics(values), 'retained_from_previous': False})
                io.write(args.output/'search.json', candidates)
        if len(candidates) != 14: raise ValueError('Incomplete registered14-candidate grid')
        final_a, refs, pending = previous.references(args.registry, record['dev_questions_sha256'], False)
        baselines = {name: study.statistics([study.fixed(r, trees[r['id']], *parameters) for r in rows])
                     for name, parameters in study.BASELINES.items()}
        if io.read(args.source/'summary.json')['baselines'] != baselines: raise ValueError('Fixed baseline controls changed')
        result = select(candidates, final_a, baselines)
        result.update(A_references=refs, pending_optional_A=pending, final_A_reference=final_a,
            raw_neural_observations_reused=len(cache), physical_encoder_executions=0,
            charged_neural_observations='One paid encoder pass per observed decision node; no forced-terminal pass')
        validate_registration(args)
        io.write(args.output/'summary.json', result); io.write(marker, seal_payload(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run'))
    for name in ('data', 'models', 'source', 'registry', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--plain-features', type=Path, help='Explicit completed768-row old-V1 cache variant; no inference')
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    if args.mode == 'register': register(args)
    elif not args.registration_sha256: parser.error('Execution requires reviewed registration SHA')
    else: run(args)


if __name__ == '__main__': main()
