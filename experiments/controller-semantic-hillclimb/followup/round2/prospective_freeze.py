"""Freeze a development-qualified Round2 policy set before any1024-panel inference.

This is a freeze/verification stage only. It neither collects test responses
nor opens test reference answers. A subsequent collector and grader must verify
this manifest and supply their own completed inference and audit receipts.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import pickle
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
import banks
from study_common import BASELINES
import extended_acquire as old

A_MODULES = {'repairability_pomdp': ('development_repairability',),
             'continuous_repairability': ('development_continuous', 'development_factorized', 'development_refined_observer'),
             'subject_science_pomdp': ('development_science_option',),
             'science_fusion_pomdp': ('development_science_fusion',)}
C_MODULES = {'hybrid_repairability': ('development_mixed_mentor',),
             'stronger_base_hybrid': ('development_stronger_base',)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def runtime_environment():
    return {'python': list(sys.version_info[:3]), 'packages': {name: version(name) for name in
            ('numpy', 'torch', 'transformers', 'safetensors', 'scikit-learn', 'sentencepiece', 'protobuf')}}


def best(values):
    if not values:
        raise ValueError('No eligible development candidate')
    return max(values, key=lambda r: (r['dev']['accuracy'], -r['dev']['mean_compute']))


def check_metrics(value, n=256):
    if value['n'] != n or type(value['correct']) is not int or not 0 <= value['correct'] <= n:
        raise ValueError('Metrics must cover the entire registered development panel')
    if not math.isclose(value['accuracy'], value['correct'] / n, abs_tol=1e-12):
        raise ValueError('Accuracy disagrees with the integer correct count')
    if not math.isfinite(value['mean_compute']) or value['mean_compute'] <= 0:
        raise ValueError('Invalid development compute measurement')


def check_targets(selected, baselines, n=256):
    if set(baselines) != set(BASELINES):
        raise ValueError('Exactly the five unchanged fixed baselines are required')
    for value in [*baselines.values(), *(r['dev'] for r in selected.values())]:
        check_metrics(value, n)
    if selected['A']['dev']['mean_judge_calls'] != 0:
        raise ValueError('A must be a POMDP without LLM review calls')
    if selected['B']['dev']['mean_judge_calls'] <= 0 or selected['C']['dev']['mean_judge_calls'] <= 0:
        raise ValueError('B must use its LLM and selected C must acquire paid LLM observations')
    a, b, c = (selected[f]['dev'] for f in ('A', 'B', 'C'))
    maximum = max(r['accuracy'] for r in baselines.values())
    gaps = {'A_above_highest_baseline': a['accuracy'] - maximum,
            'B_above_highest_baseline': b['accuracy'] - maximum,
            'C_above_highest_A_B': c['accuracy'] - max(a['accuracy'], b['accuracy'])}
    passed = {k: v >= .1 - 1e-12 for k, v in gaps.items()}
    passed['C_compute_below_B'] = c['mean_compute'] < b['mean_compute']
    return {'gaps': gaps, 'criteria': passed, 'all_required_passed': all(passed.values()),
            'C_compute_fraction_saved_vs_B': 1. - c['mean_compute'] / b['mean_compute']}


def check_development_identity(directory, expected_sha):
    identity = read_json(directory / 'identity.json')
    recorded = [identity[k] for k in ('dev_questions_sha256', 'development_questions_sha256',
                'development_question_sha256', 'dev_question_sha256') if k in identity]
    if not recorded or any(value != expected_sha for value in recorded):
        raise ValueError('Development candidates use a different panel')
    if identity.get('test_used', False):
        raise ValueError('Test outcomes influenced development selection')
    return identity


def check_no_test_used(summary):
    markers = [summary[k] for k in ('test_used', 'fresh_data_used') if k in summary]
    if not markers or any(markers):
        raise ValueError('Development summary lacks an explicit unused-test declaration')
    if summary.get('test_inference_started', False) or summary.get('fresh_test_inference_started', False):
        raise ValueError('The prospective test panel has already been used')


def source_manifest(root=ROOT):
    # This deliberately freezes the full experiment source surface, including
    # dynamic imports and archived prompt/parser implementations. Future tools
    # may be added, but changing any recorded file invalidates this freeze.
    files = sorted((root / 'experiments').rglob('*.py'))
    files += sorted((root / 'src').rglob('*.py')) if (root / 'src').exists() else []
    result = {str(p.relative_to(root)): sha(p) for p in files}
    if not result:
        raise ValueError('No project runtime source found')
    return result


def observer_slots(model):
    """Find every live observer in single, subject-routed, and hybrid planners."""
    seen_models, seen_observers, result = set(), set(), []
    def visit_observer(observer, path):
        if id(observer) in seen_observers:
            return
        seen_observers.add(id(observer))
        components = getattr(observer, 'components', None)
        if components is not None:
            for index, component in enumerate(components):
                visit_observer(component, path + '.components.' + str(index))
        elif hasattr(observer, 'directory'):
            result.append((path, observer))
        else:
            raise ValueError('Unknown observer assets')
    def visit(current, path):
        if id(current) in seen_models:
            return
        seen_models.add(id(current))
        observer = getattr(current, 'classifier', None)
        if observer is not None:
            visit_observer(observer, path + '.classifier')
        for name in ('base', 'mentor', 'regular'):
            child = getattr(current, name, None)
            if child is not None:
                visit(child, path + '.' + name)
        children = getattr(current, 'models', None)
        if isinstance(children, dict):
            for key, child in sorted(children.items()):
                visit(child, path + '.models.' + key)
    visit(model, 'model')
    if not result:
        raise ValueError('Selected planner has no recognized observer')
    return result


def validate_role(model, family, module):
    expected = {'repairability_pomdp': ('repairability_pomdp', 'Planner'),
                'continuous_repairability': ('repairability_pomdp', 'Planner'),
                'subject_science_pomdp': ('subject_science_pomdp', 'SubjectPlanner'),
                'science_fusion_pomdp': ('subject_science_pomdp', 'SubjectPlanner'),
                'hybrid_repairability': ('hybrid_repairability', 'Planner'),
                'stronger_base_hybrid': ('stronger_base_hybrid', 'Planner')}[module]
    if (type(model).__module__, type(model).__name__) != expected:
        raise ValueError('Selected pickle does not implement its declared policy family')
    if family == 'A' and module in ('subject_science_pomdp', 'science_fusion_pomdp'):
        if set(model.models) != {'math', 'chemistry', 'physics'} or set(model.settings) != set(model.models):
            raise ValueError('Subject-routed A must cover each observed subject exactly once')
        for child in model.models.values():
            validate_role(child, 'A', 'continuous_repairability')
        if any(subject not in child.dynamics for subject, child in model.models.items()):
            raise ValueError('Subject-routed A selected a planner missing that subject')
    elif family == 'A':
        if getattr(model, 'review_observer', None) is not None:
            raise ValueError('A cannot acquire LLM observations')
        if set(model.dynamics) != {'math', 'chemistry', 'physics'} and module != 'continuous_repairability':
            raise ValueError('A dynamics do not cover all subjects')
    elif family == 'C':
        if model.fit_receipt.get('state_count') != 16 or model.fit_receipt.get('maximum_reviews') != 1:
            raise ValueError('C must be the declared two-mode POMDP with one optional root LLM review')
        if module == 'stronger_base_hybrid':
            validate_role(model.base, 'A', model.base_module)
            if not model.fit_receipt.get('parent_transfer_unchanged'):
                raise ValueError('Stronger-base C changed the training-only mentor transfer')


def verify_weight_folder(folder):
    recipe = read_json(folder / 'recipe.json')
    if not recipe.get('weights') or not recipe.get('parameters') or not recipe.get('max_length'):
        raise ValueError('Observer recipe lacks weights or an explicit compute contract')
    for name, digest in recipe['weights'].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()) or sha(path) != digest:
            raise ValueError('Observer weight shard differs from its recipe')
    return recipe


def archive_model(model, family, output, assets):
    mappings = []
    for slot, observer in observer_slots(model):
        folder = Path(observer.directory).resolve()
        recipe = verify_weight_folder(folder)
        key = sha(folder / 'recipe.json')
        relative = 'observers/' + key
        if key not in assets:
            shutil.copytree(folder, output / relative)
            files = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()}
            for name, digest in files.items():
                if sha(output / relative / name) != digest:
                    raise ValueError('Observer copy failed verification')
            assets[key] = {'directory': relative, 'source_directory': str(folder), 'recipe': recipe,
                           'files': files, 'fit_receipts': {str(p): sha(p) for name in
                           ('training-summary.json', 'contract.json', 'neural-receipt.json')
                           if (p := folder.parent / name).exists()}}
        observer.directory = relative
        observer.device = 'cpu'
        mappings.append({'slot': slot, 'asset': key, 'observer_class': type(observer).__module__ + '.' + type(observer).__name__})
    destination = output / 'models' / (family + '.pkl')
    destination.write_bytes(pickle.dumps(model, protocol=5))
    return {'file': str(destination.relative_to(output)), 'sha256': sha(destination), 'observers': mappings}


def development_registry(directory, dev_sha):
    """Retain every completed same-panel A/B study, including the matched-review B."""
    result = {'A': [], 'B': [], 'evidence_files': [], 'matched_B_complete': False}
    for identity_path in sorted(directory.rglob('identity.json')):
        identity = read_json(identity_path)
        digests = [identity[k] for k in ('dev_questions_sha256', 'development_questions_sha256',
                   'development_question_sha256', 'dev_question_sha256') if k in identity]
        if not digests or any(d != dev_sha for d in digests):
            continue
        path = identity_path.parent / 'summary.json'
        matched = 'ordinary_B_source' in identity and 'root_stop_rules' in identity
        ordinary_b = (identity.get('source_sha256') == sha(HERE / 'development_b.py') or
                      identity.get('family') == 'ordinary independent LLM judge')
        if matched and not path.exists():
            raise ValueError('Registered matched-review B is unfinished; keep the test panel closed')
        if not path.exists():
            continue
        summary = read_json(path)
        if 'selected_A' not in summary and not (('B' in summary) and (matched or ordinary_b)):
            continue
        check_development_identity(path.parent, dev_sha)
        check_no_test_used(summary)
        result['evidence_files'].extend(str(p.resolve()) for name in ('identity.json', 'summary.json', 'search.json')
                                        if (p := path.parent / name).exists())
        if 'selected_A' in summary:
            candidate = summary['selected_A']
            check_metrics(candidate['dev'])
            result['A'].append({'summary': str(path.resolve()), 'candidate': candidate})
        if 'B' not in summary or not (matched or ordinary_b):
            continue
        search = read_json(path.parent / 'search.json')
        if any(r.get('family') != 'B' for r in search):
            raise ValueError('B search contains another controller family')
        if matched:
            if identity['source_sha256'] != sha(HERE / 'matched_review_b.py'):
                raise ValueError('Matched B changed after evaluation')
            expected = {json.dumps({'root_stop': root, 'second_stop': stop, 'second_action': action}, sort_keys=True)
                        for root in ('action', 'agreement', .8, .95, .995, 1.)
                        for stop in ('agreement', 'never') for action in ('reuse', 'retry', 'root_action')}
            chosen = best(search)
            if summary['matched_B'] != chosen:
                raise ValueError('Matched B summary weakened its accuracy winner')
            ordinary_path = Path(identity['ordinary_B_source'])
            ordinary = read_json(ordinary_path)
            check_development_identity(ordinary_path.parent, dev_sha)
            retained = {**ordinary['B'], 'protocol': 'ordinary-late', 'source_summary_sha256': sha(ordinary_path)}
            if summary['ordinary_B'] != retained or summary['B'] != best([retained, chosen]):
                raise ValueError('Matched B summary failed to preserve the strongest independent B')
            module, protocol = 'matched_review_b', 'matched-retained-1024-512-128'
            result['matched_B_complete'] = True
        else:
            if identity['source_sha256'] != sha(HERE / 'development_b.py') or any(not r.get('late') for r in search):
                raise ValueError('Unknown independent ordinary B evaluation')
            expected = {json.dumps({'judge_stop': stop, 'reuse_review': reuse}, sort_keys=True)
                        for stop in ('action', 'agreement', .8, .95, .995, 1.) for reuse in (False, True)}
            chosen = best(search)
            if summary['B'] != chosen:
                raise ValueError('Ordinary B summary weakened its strongest development result')
            module, protocol = 'late_pairs', 'ordinary-late'
        if len(search) != len(expected) or {json.dumps(r['settings'], sort_keys=True) for r in search} != expected:
            raise ValueError('Preserve every registered independent B setting')
        check_metrics(chosen['dev'])
        result['B'].append({'summary': str(path.resolve()), 'candidate': {**chosen, 'module': module, 'protocol': protocol}})
    if not result['A'] or not result['B'] or not result['matched_B_complete']:
        raise ValueError('Complete same-panel A, ordinary B, and matched full-review B before freezing')
    return result


def fitting_chain(directory, registry_root, seen=None):
    """Verify the original mixed fit or its training-only emission refinement."""
    seen = set() if seen is None else seen
    directory = directory.resolve()
    if directory in seen:
        raise ValueError('C fitting provenance contains a cycle')
    seen.add(directory)
    complete, identity = read_json(directory / 'complete.json'), read_json(directory / 'identity.json')
    if sha(directory / 'identity.json') != complete['identity_sha256']:
        raise ValueError('C fitting identity changed')
    files = [p for name in ('identity.json', 'complete.json', 'metadata.jsonl', 'neural.npz',
                           'neural-receipt.json', 'calibration-oof.npz') if (p := directory / name).exists()]
    if 'model_files' in identity:
        if identity['source_sha256'] != sha(HERE / 'prepare_mixed_mentor.py'):
            raise ValueError('Mixed mentor fitting source changed')
        if any(sha(path) != digest for path, digest in identity['model_files'].items()):
            raise ValueError('C fitting policy dependencies changed')
    elif 'stronger_base_parent_complete_sha256' in identity:
        if identity['source_sha256'] != sha(HERE / 'development_stronger_base.py') or identity['hybrid_sha256'] != sha(HERE / 'stronger_base_hybrid.py'):
            raise ValueError('Stronger-base construction source changed')
        if any(identity.get(k) is not False for k in ('development_or_test_used_for_mentor_refit', 'mentor_refitted', 'transfer_refitted', 'test_used')):
            raise ValueError('Stronger-base construction must preserve the training-only mentor')
        parent = Path(identity['parent_mentor_directory'])
        if sha(parent / 'complete.json') != identity['stronger_base_parent_complete_sha256'] or sha(parent / 'identity.json') != identity['parent_identity_sha256']:
            raise ValueError('Stronger-base parent fitting evidence changed')
        parent_complete, parent_files = fitting_chain(parent, registry_root, seen)
        parent_models = {r['sha256']: parent / (r['key'] + '.pkl') for r in parent_complete['models']}
        if any(sha(path) != digest for digest, path in parent_models.items()):
            raise ValueError('Stronger-base parent mentor model changed')
        if any(record['parent_sha256'] not in parent_models for record in complete['models']):
            raise ValueError('Stronger-base model has an unregistered mentor parent')
        selected = identity['base_selection']
        for name, digest in ((selected['summary'], selected['summary_sha256']),
                             (str(Path(selected['summary']).parent / 'identity.json'), selected['identity_sha256']),
                             (selected['model_path'], selected['model_sha256'])):
            if sha(name) != digest:
                raise ValueError('Stronger-base selected A evidence changed')
            files.append(Path(name))
        if read_json(selected['summary'])['selected_A'] != selected['candidate'] or complete['base_A'] != selected['candidate']:
            raise ValueError('Stronger-base A selection differs from its recorded development winner')
        if complete['parent_mentor_complete_sha256'] != identity['stronger_base_parent_complete_sha256']:
            raise ValueError('Stronger-base completion points to another mentor fit')
        files += parent_files + list(parent_models.values())
    elif 'source_complete_sha256' in identity:
        if identity['source_sha256'] != sha(HERE / 'crossfit_mentor_emissions.py'):
            raise ValueError('Excluded-calibration fitting source changed')
        module = importlib.import_module('mentor_pomdp')
        if identity['parent_mentor_source_sha256'] != sha(module.__file__):
            raise ValueError('Original mentor runtime changed')
        parents = [p.parent for p in registry_root.rglob('complete.json')
                   if p.parent.resolve() != directory and sha(p) == identity['source_complete_sha256']
                   and (p.parent / 'identity.json').exists()
                   and sha(p.parent / 'identity.json') == identity['source_identity_sha256']]
        if not parents:
            raise ValueError('Cannot verify the original mixed-mentor fitting evidence')
        parent = sorted(parents)[0]
        parent_identity = read_json(parent / 'identity.json')
        if parent_identity['tree_sha256'] != identity['tree_sha256']:
            raise ValueError('Calibration refinement changed its training tree')
        for name, field in (('metadata.jsonl', 'neural_metadata_sha256'), ('neural.npz', 'neural_outputs_sha256'),
                            ('neural-receipt.json', 'neural_receipt_sha256')):
            if sha(parent / name) != identity[field]:
                raise ValueError('Calibration refinement changed its neural fitting data')
        files += fitting_chain(parent, registry_root, seen)[1]
    else:
        raise ValueError('Unknown C fitting provenance')
    return complete, files


def infer_a_module(candidate, identity, fallback=None):
    if candidate.get('policy_module'):
        module = candidate['policy_module']
    else:
        recorded = identity.get('source_sha256')
        found = [module for module, drivers in A_MODULES.items() if any(
            (recorded.get(driver + '.py') if isinstance(recorded, dict) else recorded) == sha(HERE / (driver + '.py'))
            for driver in drivers)]
        module = found[0] if len(found) == 1 else fallback
    if module not in A_MODULES:
        raise ValueError('Cannot resolve the retained A runtime module')
    return module


def resolve_a_model(candidate, identity, summary, preferred, registry):
    if 'model_path' in candidate:
        path = Path(candidate['model_path'])
    else:
        filename = candidate.get('pickle') or candidate['model'] + '.pkl'
        digest = identity.get('model_sha256', {}).get(filename)
        local = [preferred / filename, summary.parent / filename]
        matches = [p for p in local if p.exists() and (digest is None or sha(p) == digest)]
        if not matches and digest is not None:
            matches = [p for p in registry.rglob(filename) if sha(p) == digest]
        if not matches:
            raise ValueError('Cannot resolve recorded A model assets')
        path = matches[0]
    expected = identity.get('model_sha256', {}).get(path.name)
    if not path.is_file() or (expected is not None and sha(path) != expected):
        raise ValueError('Selected A changed after development evaluation')
    return path.resolve()


def select_inputs(args):
    dev_sha = sha(args.data / 'splits/dev-questions.jsonl')
    # Refined/fusion studies retain stronger existing A candidates explicitly.
    # Resolve that recorded chain before locating its model and runtime module.
    forwarded, seen = [], set()
    while True:
        path = args.a_summary.resolve()
        if path in seen:
            raise ValueError('Cyclic retained-A provenance')
        seen.add(path)
        summary = read_json(path)
        candidate = summary['selected_A']
        if 'source_summary' not in candidate:
            break
        target = Path(candidate['source_summary'])
        if sha(target) != candidate['source_summary_sha256']:
            raise ValueError('Retained stronger A source summary changed')
        forwarded.append(str(path))
        args.a_summary = target
    paths = {'A': args.a_summary, 'B': args.b_summary, 'C': args.c_summary}
    summaries = {family: read_json(path) for family, path in paths.items()}
    identities = {family: check_development_identity(path.parent, dev_sha) for family, path in paths.items()}
    for summary in summaries.values():
        check_no_test_used(summary)
    registry = development_registry(args.data.parent, dev_sha)
    registry['forwarded_A_summaries'] = forwarded
    a = summaries['A']['selected_A']
    best_a = best([r['candidate'] for r in registry['A']])
    if (a['dev']['accuracy'], -a['dev']['mean_compute']) != (best_a['dev']['accuracy'], -best_a['dev']['mean_compute']):
        raise ValueError('A must be the strongest completed same-panel A; do not lower A to ease the C gap')
    if not any(r['summary'] == str(args.a_summary.resolve()) for r in registry['A']):
        raise ValueError('Selected A is outside the registered experiment area')
    args.a_module = infer_a_module(a, identities['A'], getattr(args, 'a_module', None))
    if args.a_module not in ('subject_science_pomdp', 'science_fusion_pomdp') and a != best(read_json(args.a_summary.parent / 'search.json')):
        raise ValueError('A summary does not retain its strongest development candidate')
    b_record = max(registry['B'], key=lambda r: (r['candidate']['dev']['accuracy'], -r['candidate']['dev']['mean_compute']))
    b = b_record['candidate']
    b_path = Path(b_record['summary'])
    summaries['B'] = read_json(b_path)
    identities['B'] = check_development_identity(b_path.parent, dev_sha)
    c_search = read_json(args.c_summary.parent / 'search.json')
    c = best([r for r in c_search if r['dev']['mean_judge_calls'] > 0 and r['dev']['mean_compute'] < b['dev']['mean_compute']])
    c_module = c.get('policy_module', 'hybrid_repairability')
    if c_module not in C_MODULES:
        raise ValueError('Unknown hybrid C runtime module')
    baselines = summaries['B']['baselines']
    for summary in (summaries['A'], summaries['C']):
        if set(summary['baselines']) != set(baselines):
            raise ValueError('Different fixed baselines across development studies')
        for name, values in baselines.items():
            for key in ('n', 'correct', 'accuracy', 'mean_compute'):
                if not math.isclose(values[key], summary['baselines'][name][key], abs_tol=1e-9):
                    raise ValueError('Development baseline outcomes differ across policies')
    selected = {'A': {**a, 'family': 'A', 'module': args.a_module, 'protocol': 'plain', 'settings': a.get('settings', {}),
                      'source_summary': str(args.a_summary.resolve())},
                'B': {**b, 'family': 'B', 'source_summary': str(b_path.resolve())},
                'C': {**c, 'family': 'C', 'module': c_module, 'protocol': 'mixed-strong-root',
                      'source_summary': str(args.c_summary.resolve())}}
    checks = check_targets(selected, baselines)
    if not checks['all_required_passed']:
        raise ValueError('Development targets are not met; reserved test remains closed: ' + json.dumps(checks))
    expected_sources = {'A': A_MODULES[args.a_module],
                        'B': ('matched_review_b',) if b['module'] == 'matched_review_b' else ('development_b',),
                        'C': C_MODULES[c_module]}
    for family, modules in expected_sources.items():
        recorded = identities[family]['source_sha256']
        if not any((recorded.get(module + '.py') if isinstance(recorded, dict) else recorded) == sha(HERE / (module + '.py')) for module in modules):
            raise ValueError('A development driver changed after its recorded evaluation')
        source_map = identities[family]['source_sha256']
        if isinstance(source_map, dict) and any(sha(HERE / name) != digest for name, digest in source_map.items()):
            raise ValueError('A development dependency changed after evaluation')
        policy_module = importlib.import_module(selected[family]['module'])
        for name in ('policy_sha256', 'pomdp_sha256', 'hybrid_sha256'):
            expected_module = importlib.import_module('repairability_pomdp') if name == 'pomdp_sha256' and family == 'A' else policy_module
            if name == 'policy_sha256' and selected[family]['module'] == 'science_fusion_pomdp':
                expected_module = importlib.import_module('continuous_repairability')
            if name in identities[family] and identities[family][name] != sha(expected_module.__file__):
                raise ValueError('Selected policy source differs from its development evaluation')
        for name, filename in (('factorized_sha256', 'factorized_repairability.py'),
                               ('core_sha256', 'repairability_pomdp.py'),
                               ('numeric_consistency_sha256', 'numeric_consistency.py'),
                               ('fusion_sha256', 'science_fusion_pomdp.py'),
                               ('receipt_validator_sha256', 'development_mixed_mentor.py')):
            if name in identities[family] and identities[family][name] != sha(HERE / filename):
                raise ValueError('Selected development dependency changed: ' + filename)
    a_path = resolve_a_model(a, identities['A'], args.a_summary, args.a_models, args.data.parent)
    model_paths = {'A': a_path,
                   'C': args.c_models / (c['model'] + '.pkl')}
    c_fitting, fit_files = fitting_chain(args.c_models, args.data.parent)
    if identities['C']['models_sha256'] != sha(args.c_models / 'complete.json'):
        raise ValueError('C development used a different completed model collection')
    if c_module == 'stronger_base_hybrid':
        base = read_json(args.c_models / 'identity.json')['base_selection']
        if base['model_sha256'] != sha(a_path) or base['module'] != args.a_module or base['candidate'].get('settings', {}) != a.get('settings', {}):
            raise ValueError('Stronger-base C must inherit the strongest selected A with its recorded settings')
    registry['evidence_files'].extend(str(p.resolve()) for p in fit_files)
    c_record = next(r for r in c_fitting['models'] if r['key'] == c['model'])
    if sha(model_paths['C']) != c_record['sha256']:
        raise ValueError('Selected C changed after fitting')
    recorded_models = identities['A'].get('model_sha256', {})
    if recorded_models and recorded_models.get(model_paths['A'].name) != sha(model_paths['A']):
        raise ValueError('Selected A changed after development evaluation')
    registry['A_model_origin'] = {
        'selected_pickle_sha256': sha(a_path),
        'historical_evaluation_pickle_digest_available': a_path.name in recorded_models,
        'historical_evaluation_pickle_digest': recorded_models.get(a_path.name),
        'independent_training_reconstruction_performed': False,
        'validation_required': 'Replay every selected development decision with the frozen model before test collection. This confirms current behavior and costs; missing historical pickle hashes cannot establish historical byte identity.'}
    return selected, baselines, checks, model_paths, identities, registry


def freeze(args):
    if args.output.exists():
        raise ValueError('Never overwrite a prospective freeze')
    selection = read_json(args.data / 'splits/selection.json')
    reserved = selection['splits']['test']
    if reserved['n'] != 1024:
        raise ValueError('Keep the entire reserved1024-question test panel')
    test_path = args.data / 'splits/test-questions.jsonl'
    if sha(test_path) != reserved['sha256']:
        raise ValueError('The reserved question file changed')
    for path in args.data.glob('test*'):
        if path.is_file() and path.stat().st_size:
            raise ValueError('Prospective test data already exist; do not reselect on this panel')
    for split in ('train', 'dev'):
        complete = read_json(args.data / (split + '-ordinary-complete.json'))
        for kind, suffix in (('base', 'observations'), ('late', 'late-observations')):
            if sha(args.data / (split + '-' + suffix + '.jsonl')) != complete[kind + '_bank_sha256']:
                raise ValueError('Finish and seal paired training/development collection before freezing')
    selected, baselines, checks, model_paths, identities, registry = select_inputs(args)
    # Runtime declarations are checked before unpickling selected local models.
    for module in (*A_MODULES, *C_MODULES):
        importlib.import_module(module)
    planners = {family: pickle.loads(path.read_bytes()) for family, path in model_paths.items()}
    for family, model in planners.items():
        validate_role(model, family, selected[family]['module'])
        for _, observer in observer_slots(model):
            recipe = verify_weight_folder(Path(observer.directory))
            protected = recipe.get('protected_split_sha256')
            if protected and any(protected[split] != sha(args.data / 'splits' / (split + '-questions.jsonl')) for split in ('train', 'dev', 'test')):
                raise ValueError('Auxiliary observer was trained against different protected splits')
    mentor_complete = read_json(args.mentor_bank / 'complete.json')
    if mentor_complete['bank_sha256'] != sha(args.mentor_bank / 'train.jsonl') or mentor_complete['contract_sha256'] != sha(args.mentor_bank / 'contract.json'):
        raise ValueError('Mixed mentor fitting bank changed')
    runtime = source_manifest()
    evidence_paths = set()
    evidence_paths.update(Path(p) for p in registry['evidence_files'])
    for path in (args.a_summary, args.b_summary, args.c_summary):
        evidence_paths.update(p for name in ('summary.json', 'identity.json', 'search.json') if (p := path.parent / name).exists())
    for directory in (args.a_models, args.c_models, args.mentor_bank):
        evidence_paths.update(p for name in ('identity.json', 'complete.json', 'contract.json', 'training-neural.npz',
            'training-neural.jsonl', 'metadata.jsonl', 'neural.npz', 'neural-receipt.json') if (p := directory / name).exists())
    a_directory = args.a_summary.parent
    a_summary = read_json(args.a_summary)
    for field, filename in (('oof_provenance_sha256', 'oof-provenance.json'),
                             ('component_alignment_sha256', 'component-alignment.json')):
        if field in a_summary and sha(a_directory / filename) != a_summary[field]:
            raise ValueError('Selected A training alignment evidence changed')
    for directory in {a_directory, model_paths['A'].parent}:
        evidence_paths.update(p for name in ('identity.json', 'oof-provenance.json', 'oof-alignment.jsonl',
            'component-alignment.json', 'training-neural.npz', 'training-neural.jsonl') if (p := directory / name).exists())
    oof_path = a_directory / 'oof-provenance.json'
    if oof_path.exists():
        oof = read_json(oof_path)
        if oof.get('alignment_sha256') != sha(a_directory / 'oof-alignment.jsonl'):
            raise ValueError('Selected A excluded-fold alignment changed')
    evidence_paths.update(p for name in ('train-observations.jsonl', 'train-late-observations.jsonl',
        'dev-observations.jsonl', 'dev-late-observations.jsonl', 'train-plain-complete.json', 'dev-plain-complete.json',
        'train-ordinary-complete.json', 'dev-ordinary-complete.json', 'splits/selection.json') if (p := args.data / name).exists())
    evidence_paths.update((args.mentor_bank / 'train.jsonl', *(model_paths.values())))
    evidence_paths.update(p for name in ('pyproject.toml', 'uv.lock') if (p := ROOT / name).exists())
    evidence = {str(p.resolve()): sha(p) for p in sorted(evidence_paths)}
    args.output.mkdir(parents=True)
    (args.output / 'models').mkdir()
    assets, descriptors = {}, {}
    for family, model in planners.items():
        descriptors[family] = archive_model(model, family, args.output, assets)
        descriptors[family]['selected_source_sha256'] = sha(model_paths[family])
    corpora = {}
    for asset in assets.values():
        digest = asset['recipe'].get('corpus_sha256')
        if digest is None or digest in corpora:
            continue
        source = ROOT / '.cache/semantic-textbooks'
        if sha(source / 'passages.jsonl') != digest:
            raise ValueError('Observer textbook corpus differs from its training recipe')
        destination = args.output / 'corpora' / digest
        destination.mkdir(parents=True)
        files = {}
        for name in ('manifest.json', 'passages.jsonl', 'tfidf.pkl'):
            shutil.copy2(source / name, destination / name)
            files[name] = sha(source / name)
            if sha(destination / name) != files[name]:
                raise ValueError('Failed to archive textbook retrieval assets')
        corpora[digest] = {'directory': str(destination.relative_to(args.output)), 'files': files}
    for relative, digest in runtime.items():
        target = args.output / 'runtime' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if sha(target) != digest:
            raise ValueError('Runtime source copy changed')
    selection_archives = {}
    for family, summary in (('A', args.a_summary), ('B', Path(selected['B']['source_summary'])), ('C', args.c_summary)):
        target = args.output / 'selection-evidence' / family
        target.mkdir(parents=True)
        for name in ('summary.json', 'identity.json', 'search.json'):
            source = summary.parent / name
            if source.exists():
                shutil.copy2(source, target / name)
                selection_archives[str((target / name).relative_to(args.output))] = sha(target / name)
    payload = {'schema': 'round2-prospective-freeze-v1', 'stage': 'frozen; awaiting verified development audit and blind collection',
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'questions': 1024,
               'reserved_question_sha256': reserved['sha256'], 'selection_manifest_sha256': sha(args.data / 'splits/selection.json'),
               'data_directory': str(args.data.resolve()), 'project_directory': str(ROOT),
               'selected': selected, 'development_baselines': baselines, 'development_checks': checks,
               'development_identities': identities, 'runtime_sources': runtime, 'evidence': evidence,
               'runtime_environment': runtime_environment(),
               'runtime_execution': 'Use hash-verified project sources at project_directory; archived source is a preservation copy. Observer checkpoints and textbook retrieval assets are relocated to their verified archive copies.',
               'completed_A_B_registry': registry,
               'selection_rule': 'Strongest completed same-panel A and independent B (ordinary or matched full-review); highest-accuracy queried C strictly cheaper than that B. No test outcomes used.',
               'selection_archives': selection_archives,
               'models': descriptors, 'observer_assets': assets, 'corpora': corpora, 'baselines': BASELINES,
               'solver_digest': old.MODEL_DIGEST, 'reviewer_digest': old.JUDGE_DIGEST,
               'targets': {'units': 'percentage points', 'A_B_above_best_baseline': .1,
                           'C_above_best_A_B': .1, 'C_compute_strictly_below_B': True},
               'inference_contract': {'solver': 'Same frozen compact4Bweights', 'additional_solver_calls': 2,
                   'A': 'POMDP with discriminative observations and no LLM review',
                   'B': 'Strongest independent ordinary or matched full-review LLM development policy',
                   'C': 'Repairability POMDP plus one optional strong retained root LLM review',
                   'references': 'Blank references before policy or model calls; grades remain closed until all1024decisions complete',
                   'compute': '(4*solver tokens +3*all physical review tokens +sum(encoder billions*actual slots))/3.09',
                   'sensitivity': 'Also report35Btotalreview parameter proxy',
                   'offline_training_in_inference_compute': False},
               'remaining_required_stages': ['verified blind collector', 'all1024completed inference receipts',
                   'independent cost and paid-prefix audits', 'single prospective grading and paired confidence intervals'],
               'test_reference_answers_opened': False, 'test_inference_started': False}
    (args.output / 'frozen.json').write_text(json.dumps(payload, indent=2) + '\n')
    freeze_sha = sha(args.output / 'frozen.json')
    (args.output / 'frozen.sha256').write_text(freeze_sha + '\n')
    verify(args.output, freeze_sha)
    print(json.dumps({'freeze_sha256': sha(args.output / 'frozen.json'), 'development_checks': checks,
                      'test_inference_started': False, 'questions': 1024}), flush=True)


def verify(output, expected_sha=None):
    if expected_sha is not None and sha(output / 'frozen.json') != expected_sha:
        raise ValueError('Frozen manifest changed from the externally recorded digest')
    frozen = read_json(output / 'frozen.json')
    if frozen['schema'] != 'round2-prospective-freeze-v1' or frozen['questions'] != 1024:
        raise ValueError('Unknown prospective freeze')
    if not check_targets(frozen['selected'], frozen['development_baselines'])['all_required_passed']:
        raise ValueError('Frozen development targets do not pass')
    if 'runtime_environment' in frozen and frozen['runtime_environment'] != runtime_environment():
        raise ValueError('Python or inference package versions differ from the freeze')
    root, data = Path(frozen['project_directory']), Path(frozen['data_directory'])
    if sha(data / 'splits/test-questions.jsonl') != frozen['reserved_question_sha256']:
        raise ValueError('Reserved question file changed')
    for relative, digest in frozen['runtime_sources'].items():
        if sha(root / relative) != digest or sha(output / 'runtime' / relative) != digest:
            raise ValueError('Frozen runtime source changed: ' + relative)
    for path, digest in frozen['evidence'].items():
        if sha(path) != digest:
            raise ValueError('Frozen fitting or development evidence changed: ' + path)
    for relative, digest in frozen.get('selection_archives', {}).items():
        if sha(output / relative) != digest:
            raise ValueError('Archived development selection evidence changed')
    for descriptor in frozen['models'].values():
        if sha(output / descriptor['file']) != descriptor['sha256']:
            raise ValueError('Frozen planner changed')
    for asset in frozen['observer_assets'].values():
        folder = output / asset['directory']
        for name, digest in asset['files'].items():
            if sha(folder / name) != digest:
                raise ValueError('Frozen observer asset changed')
        for path, digest in asset['fit_receipts'].items():
            if sha(path) != digest:
                raise ValueError('Observer fitting receipt changed')
        verify_weight_folder(folder)
    for digest, corpus in frozen.get('corpora', {}).items():
        for name, expected in corpus['files'].items():
            if sha(output / corpus['directory'] / name) != expected:
                raise ValueError('Archived textbook retrieval assets changed')
        if sha(output / corpus['directory'] / 'passages.jsonl') != digest:
            raise ValueError('Archived corpus does not match the observer profile')
    return frozen


def load_models(output, device='cpu', expected_sha=None):
    frozen = verify(output, expected_sha)
    for module in (*A_MODULES, *C_MODULES):
        importlib.import_module(module)
    result = {}
    for family, descriptor in frozen['models'].items():
        model = pickle.loads((output / descriptor['file']).read_bytes())
        validate_role(model, family, frozen['selected'][family]['module'])
        slots = dict(observer_slots(model))
        if set(slots) != {r['slot'] for r in descriptor['observers']}:
            raise ValueError('Frozen planner observer topology changed')
        for mapping in descriptor['observers']:
            observer = slots[mapping['slot']]
            observer.directory = str((output / frozen['observer_assets'][mapping['asset']]['directory']).resolve())
            observer.device = device
            asset = frozen['observer_assets'][mapping['asset']]
            corpus_sha = asset['recipe'].get('corpus_sha256')
            if corpus_sha:
                from evidence import Retriever
                corpus = frozen['corpora'][corpus_sha]
                observer._retriever = Retriever(output / corpus['directory'])
        result[family] = model
    return frozen, result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'verify'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-freeze-sha256')
    for name in ('data', 'a-summary', 'b-summary', 'c-summary', 'a-models', 'c-models', 'mentor-bank'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--a-module', choices=tuple(A_MODULES))
    args = p.parse_args()
    if args.phase == 'verify':
        verify(args.output, args.expected_freeze_sha256)
        print('Prospective freeze verified; no test inference or grading performed.')
    else:
        if any(getattr(args, name.replace('-', '_')) is None for name in
               ('data', 'a-summary', 'b-summary', 'c-summary', 'a-models', 'c-models', 'mentor-bank', 'a-module')):
            p.error('All selection, model, and data arguments are required for freeze')
        freeze(args)
