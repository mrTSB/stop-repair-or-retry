"""Registered CPU-only metadata review replacement; no development execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import review_metadata_diagnostic as diagnostic
import review_metadata_spec as spec
from paid_review_metadata import Adapter, digest, signal as metadata_signal
import metadata_review_hybrid as runtime
import prepare_mixed_mentor as prepare
from prepare_mixed_mentor import sha, fingerprint
from fit_new_source_mentor import no_neural_inference, source_files
import mentor_pomdp as mentor

RECIPE = {'schema': 'paid-metadata-review-fit-v1', 'model': 'metadata_logistic',
    'features': spec.FEATURES, 'C': 1., 'max_iter': 1000, 'seed': spec.SEED,
    'training_weights': 'Equal total weight per question across root/critical/independent siblings',
    'runtime_calibration': 'Fixed-grid weighted BCE on raw outer held-question predictions, then fit classifier to all320 records',
    'temperature_grid': spec.TEMPERATURES,
    'root_emission_calibration': 'Original strict nested outer-held probabilities; never final full-fit probabilities',
    'root_cohorts': {'mixed': 192, 'newsource': 128}, 'smoothing': [8., 32.],
    'parent_variants': 'Only semantic_review=True; original models retained',
    'auxiliary_protocols': 'Correctness supervision only; never root transition/emission counts',
    'changed': ['root review likelihood', 'review-conditioned first transition', 'review cost'],
    'preserved': ['A8', 'truth transfer', 'child calibration', 'child emissions', 'initial/prior/plain dynamics',
                  'terminal transitions', 'mentor comparison marginal', 'first/last solver costs'],
    'cost': 'Actual physical review token totals*3/3.09, same local/global shrinkage4 and unseen default2800, no review encoder',
    'include_reviews': False, 'neural_calls': 0, 'cloud_calls': 0,
    'development_or_test_question_response_values_read': False, 'policy_module': 'metadata_review_hybrid'}


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def verify_hashes(mapping):
    for path, value in mapping.items():
        if sha(path) != value:
            raise ValueError('Source/input digest changed: ' + str(path))


def load_diagnostic(args):
    directory = args.diagnostic
    summary = read(directory / 'summary.json'); original = read(directory / 'registration.json')
    implementation = read(directory / 'implementation.json')
    checks = {'registration.json': 'registration_sha256', 'implementation.json': 'implementation_sha256',
              'private-features.jsonl': 'private_features_sha256', 'private-predictions.npz': 'predictions_sha256',
              'fold-receipts.json': 'fold_receipts_sha256'}
    for name, key in checks.items():
        if sha(directory / name) != summary[key]:
            raise ValueError('Changed completed diagnostic artifact: ' + name)
    if not summary['strict_nested_question_calibration'] or summary['development_or_test_used']:
        raise ValueError('Require the completed training-only nested diagnostic')
    if sha(spec.__file__) != original['specification_sha256']:
        raise ValueError('Frozen feature specification changed')
    verify_hashes(implementation['source_files']); verify_hashes(implementation['input_evidence'])
    records, provenance = diagnostic.load_records(args, original)
    saved = [json.loads(line) for line in (directory / 'private-features.jsonl').read_text().splitlines()]
    if records != saved or fingerprint(records) != implementation['records_sha256']:
        raise ValueError('Paid feature/reference alignment differs from the completed diagnostic')
    with np.load(directory / 'private-predictions.npz') as archive:
        nested = archive['metadata_logistic'].copy()
    receipts = read(directory / 'fold-receipts.json')['metadata_logistic']
    raw = np.full(len(records), np.nan)
    if len(receipts) != 3 or len(nested) != len(records):
        raise ValueError('Incomplete nested prediction grid')
    ids = {r['id'] for r in records}
    model_paths = []
    for fold, receipt in enumerate(receipts):
        held_ids = {qid for qid in ids if diagnostic.fold(qid) == fold}
        if receipt['outer_fold'] != fold or set(receipt['held_ids']) != held_ids or set(receipt['training_ids']) != ids - held_ids:
            raise ValueError('Outer question exclusion differs from the fixed recipe')
        if [inner['fold'] for inner in receipt['inner']] != [0, 1, 2]:
            raise ValueError('Incomplete nested calibration folds')
        for inner in receipt['inner']:
            inner_held = {qid for qid in ids - held_ids if diagnostic.fold(qid, True) == inner['fold']}
            if set(inner['held_ids']) != inner_held or set(inner['training_ids']) != ids - held_ids - inner_held:
                raise ValueError('Nested calibration used a held question')
        path = directory / f'metadata_logistic-outer{fold}.pkl'; model_paths.append(path)
        if sha(path) != receipt['model_sha256']:
            raise ValueError('Changed outer metadata classifier')
        fitted = pickle.loads(path.read_bytes())
        indexes = [i for i, r in enumerate(records) if r['id'] in held_ids]
        held = [records[i] for i in indexes]
        raw[indexes] = diagnostic.predict(fitted, held)
        np.testing.assert_allclose(diagnostic.calibrated(raw[indexes], receipt['temperature'], held), nested[indexes],
                                   rtol=0, atol=1e-15)
    if not np.isfinite(raw).all() or not np.isfinite(nested).all():
        raise ValueError('Nonfinite held-question probabilities')
    evidence = {**implementation['input_evidence'],
        **{str(path.resolve()): sha(path) for path in model_paths +
           [directory / name for name in [*checks, 'summary.json', 'numerical-audit.json']]}}
    return records, raw, nested, evidence, original, provenance


def parent_artifacts(args, cohort):
    refined = getattr(args, cohort + '_parent')
    raw = args.parent_models if cohort == 'mixed' else args.newsource_raw
    files = [refined / 'identity.json', refined / 'complete.json', raw / 'identity.json', raw / 'complete.json']
    complete, identity = read(files[1]), read(files[0])
    raw_complete, raw_identity = read(files[3]), read(files[2])
    if complete['identity_sha256'] != sha(files[0]) or raw_complete['identity_sha256'] != sha(files[2]):
        raise ValueError('Changed parent completion/identity')
    if complete['parent_complete_sha256'] != sha(raw / 'complete.json'):
        raise ValueError('Refined parent does not descend from the declared raw fit')
    if complete['development_or_test_used'] or raw_complete['development_or_test_used']:
        raise ValueError('Parent fit must use training only')
    if cohort == 'mixed':
        if identity['source_identity_sha256'] != sha(raw / 'identity.json'):
            raise ValueError('Mixed refinement identity mismatch')
        import crossfit_mentor_emissions
        if identity['source_sha256'] != sha(crossfit_mentor_emissions.__file__):
            raise ValueError('Changed parent emission fitting source')
    else:
        if identity['new_source_parent_identity_sha256'] != sha(raw / 'identity.json'):
            raise ValueError('New-source refinement identity mismatch')
        verify_hashes(identity['source_files']); verify_hashes(raw_identity['source_files'])
        if raw_identity['parent_complete_sha256'] != sha(args.parent_models / 'complete.json'):
            raise ValueError('New-source raw fit is not anchored to the mixed raw fit')
    selected = [r for r in complete['models'] if r['semantic_review']]
    if [(r['smoothing'], r['semantic_review']) for r in selected] != [(8., True), (32., True)]:
        raise ValueError('Use exactly the two semantic smoothing variants')
    for row in selected:
        path = refined / (row['key'] + '.pkl'); files.append(path)
        if sha(path) != row['sha256']:
            raise ValueError('Changed trained parent pickle')
        raw_spec = next(r for r in raw_complete['models'] if r['key'] == row['key'])
        if row['parent_sha256'] != raw_spec['sha256']:
            raise ValueError('Parent model ancestry differs')
    for name, key in [('metadata.jsonl', 'neural_metadata_sha256'), ('neural.npz', 'neural_outputs_sha256')]:
        path = raw / name; files.append(path)
        if sha(path) != identity[key]:
            raise ValueError('Changed underlying parent observation arrays')
    files.append(refined / 'calibration-oof.npz')
    if cohort == 'newsource' and sha(files[-1]) != identity['calibration_oof_sha256']:
        raise ValueError('Changed parent OOF calibration')
    original_identity = read(args.parent_models / 'identity.json')
    verify_hashes(original_identity['model_files'])
    if original_identity['source_sha256'] != sha(prepare.__file__):
        raise ValueError('Original paid-input preparer source changed')
    original_receipt_path = args.parent_models / 'neural-receipt.json'
    original_receipt = read(original_receipt_path); files.append(original_receipt_path)
    first = pickle.loads((refined / (selected[0]['key'] + '.pkl')).read_bytes())
    for observer in (first.base.classifier, first.mentor.classifier):
        directory = Path(observer.directory); recipe_path = directory / 'recipe.json'
        if sha(recipe_path) != original_receipt['encoder_recipes']['full']:
            raise ValueError('Inherited student/child observer recipe changed')
        recipe = read(recipe_path); files.append(recipe_path)
        for filename, expected in recipe['weights'].items():
            path = directory / filename
            if sha(path) != expected:
                raise ValueError('Inherited student/child observer weights changed')
            files.append(path)
    return complete, raw_identity, {str(p.resolve()): sha(p) for p in files}


def support_and_mask():
    support = np.zeros((8, 8)); mask = np.ones((8, mentor.N_REVIEW))
    for s in range(8):
        for c in (0, 1):
            support[s, mentor.state(2 * (s // 4) + c, s % 2)] = 1.
        c, m = (s // 2) % 2, s % 2
        for r in range(30):
            agree = r % 2
            if (agree and c != m) or (not agree and c == m == 1):
                mask[s, r] = 1e-4
        if m:
            mask[s, 31] = 0.
    return support, mask


def root_records(rows, trees, judges, records, probabilities, cohort):
    lookup = {r['id']: (r, float(p)) for r, p in zip(records, probabilities)
              if r['protocol'] == 'root' and (cohort == 'mixed' or r['bank'] == 'new')}
    selected = [row for row in rows if row['id'] in lookup]
    expected = RECIPE['root_cohorts'][cohort]
    if len(lookup) != expected or len(selected) != expected:
        raise ValueError('Root transition cohort changed')
    result = []
    for row in selected:
        qid = row['id']; tree = trees[qid]; judge = judges[qid, '']; recorded, probability = lookup[qid]
        if recorded['review_sha256'] != fingerprint(judge) or recorded['current_sha256'] != fingerprint(tree['']):
            raise ValueError('Root paid response changed')
        truth = int(mentor.correct(judge, row['answer']))
        if truth != recorded['target']:
            raise ValueError('Root training target differs from archived review label')
        hard = int(np.mean([mentor.correct(tree[p], row['answer']) for p in mentor.core.FIXED]) < .6)
        state = lambda path: mentor.state(2 * hard + int(mentor.correct(tree[path], row['answer'])), truth)
        result.append({'id': qid, 'subject': row['subject'], 'state': state(''),
            'signal': metadata_signal(judge, tree[''], probability), 'nested_probability': probability,
            'outer_fold': diagnostic.fold(qid), 'first_states': [state('guided'), state('retry')],
            'review_cost': (judge['input_tokens'] + judge['output_tokens']) * 3 / 3.09})
    return result


def rebuild(records, subjects, smoothing, encoder_price=0.):
    support, mask = support_and_mask()
    def counts(selected):
        review = .5 * mask.copy(); first = np.zeros((mentor.N_REVIEW, 2, 8, 8))
        cost = np.zeros(8); n = np.zeros(8)
        for row in selected:
            s, signal = row['state'], row['signal']; review[s, signal] += 1
            cost[s] += row['review_cost'] + encoder_price; n[s] += 1
            for action, nxt in enumerate(row['first_states']):
                first[signal, action, s, nxt] += 1
        return review, first, cost, n
    normalize = lambda values: values / values.sum(-1, keepdims=True)
    global_review, global_first, global_cost, global_n = counts(records)
    pooled_first = normalize(global_first.sum(0) + support[None])
    pooled_cost = np.divide(global_cost, global_n, out=np.full_like(global_cost, 2800.), where=global_n > 0)
    result = {}
    for subject in subjects:
        review, first, cost, n = counts([r for r in records if r['subject'] == subject])
        result[subject] = (normalize(review + smoothing * normalize(global_review)),
                           normalize(first + smoothing * pooled_first[None]), (cost + 4. * pooled_cost) / (n + 4.))
    return result


def preserved_digest(parent):
    """Canonical component values, independent of pickle aliases/array layout."""
    caches = {'_model', '_tokenizer', '_recipe', '_cache', '_retriever', '_backgrounds', '_alphas'}
    active = set()
    def encode(value):
        if value is None or type(value) in (bool, int, str):
            return [type(value).__name__, value]
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                raise ValueError('Nonfinite protected model scalar')
            return ['float', float(value).hex()]
        if isinstance(value, np.integer):
            return ['int', int(value)]
        if isinstance(value, np.bool_):
            return ['bool', bool(value)]
        if isinstance(value, np.ndarray):
            if value.dtype.fields:
                return ['structured-array', list(value.shape), [[name, encode(value[name])] for name in value.dtype.names]]
            if value.dtype.hasobject:
                return ['object-array', list(value.shape), encode(value.tolist())]
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError('Nonfinite protected model array')
            return ['array', value.dtype.str, list(value.shape), hashlib.sha256(value.tobytes(order='C')).hexdigest()]
        if isinstance(value, np.dtype):
            return ['dtype', value.str]
        if isinstance(value, Path):
            return ['path', str(value)]
        if isinstance(value, type):
            return ['type', value.__module__, value.__qualname__]
        marker = id(value)
        if marker in active:
            raise ValueError('Cyclic protected model structure')
        active.add(marker)
        try:
            if isinstance(value, dict):
                pairs = [[encode(k), encode(v)] for k, v in value.items()]
                return ['dict', sorted(pairs, key=lambda item: json.dumps(item[0], sort_keys=True))]
            if isinstance(value, (list, tuple)):
                return [type(value).__name__, [encode(v) for v in value]]
            if hasattr(value, '__dict__'):
                return ['object', type(value).__module__, type(value).__qualname__,
                        [[name, encode(item)] for name, item in sorted(vars(value).items()) if name not in caches]]
            raise ValueError('Unsupported protected model value: ' + str(type(value)))
        finally:
            active.remove(marker)
    arrays = {s: {key: value for key, value in vars(d).items()
                  if key not in ('review', 'reviewed_transition', 'review_costs', '_alphas')}
              for s, d in parent.mentor.dynamics.items()}
    return digest(encode((parent.base, parent.transfer, parent.mentor.classifier, parent.mentor.regular, arrays)))


def inputs(args):
    records, raw, nested, evidence, registration, _ = load_diagnostic(args)
    source = Path(registration['source_directories']['root'])
    rows, trees, judges, old_ids, _ = prepare.training_view(argparse.Namespace(
        old_data=args.old_data, new_data=args.new_data, mentor=source))
    parents = {}
    for cohort in RECIPE['root_cohorts']:
        complete, identity, files = parent_artifacts(args, cohort)
        local_tree = trees if cohort == 'mixed' else {qid: t for qid, t in trees.items() if qid not in old_ids}
        if fingerprint(local_tree) != identity['tree_sha256']:
            raise ValueError('Parent training tree differs from the selected source cohort')
        parents[cohort] = complete; evidence.update(files)
    return records, raw, nested, rows, trees, judges, parents, evidence


def run(args):
    outputs = {name: getattr(args, name + '_output') for name in RECIPE['root_cohorts']}
    with no_neural_inference(), threadpool_limits(limits=1):
        records, raw, nested, rows, trees, judges, parents, evidence = inputs(args)
        sources = source_files()
        if args.mode == 'register':
            if any(path.exists() for path in outputs.values()):
                raise ValueError('Require two new output directories')
            contract = {'recipe': RECIPE, 'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                        'source_files': sources, 'input_evidence': evidence,
                        'diagnostic_directory': str(args.diagnostic.resolve()),
                        'outputs': {key: str(value.resolve()) for key, value in outputs.items()},
                        'historical_identity_claim': False}
            for path in outputs.values():
                path.mkdir(parents=True); write(path / 'registration.json', contract)
            print(json.dumps({'registered': True, 'sha256': sha(outputs['mixed'] / 'registration.json')}), flush=True)
            return
        registrations = [read(path / 'registration.json') for path in outputs.values()]
        contract = registrations[0]
        if registrations[1] != contract or contract['recipe'] != RECIPE or sources != contract['source_files']:
            raise ValueError('Registered source/recipe changed')
        if contract['input_evidence'] != evidence or contract['outputs'] != {k: str(v.resolve()) for k, v in outputs.items()}:
            raise ValueError('Registered evidence/output changed')
        verify_hashes(sources); verify_hashes(evidence)
        if any((p / 'identity.json').exists() or (p / 'complete.json').exists() for p in outputs.values()):
            raise ValueError('Never overwrite a started fit')
        temperature_search = [{'temperature': t, 'loss': diagnostic.binary_loss(
            [r['target'] for r in records], diagnostic.calibrated(raw, t, records), diagnostic.weights(records))}
            for t in spec.TEMPERATURES]
        temperature = min(temperature_search, key=lambda r: (r['loss'], r['temperature']))['temperature']
        fitted = diagnostic.train_model('metadata_logistic', records)
        adapter = Adapter(fitted, temperature, {})
        adapter.receipt = {'learned_state_sha256': digest(adapter.state()), 'training_questions': 192,
            'training_records': 320, 'runtime_temperature': temperature,
            'registration_sha256': sha(outputs['mixed'] / 'registration.json'),
            'root_tables_use_nested_oof': True, 'neural_calls': 0}
        adapter.validate()
        for cohort, output in outputs.items():
            parent_dir = getattr(args, cohort + '_parent'); complete = parents[cohort]
            aligned = root_records(rows, trees, judges, records, nested, cohort)
            identity = {'source_sha256': sha(__file__), 'policy_module': 'metadata_review_hybrid',
                'registration_sha256': sha(output / 'registration.json'),
                'parent_models_directory': str(parent_dir.resolve()),
                'parent_complete_sha256': sha(parent_dir / 'complete.json'),
                'parent_identity_sha256': sha(parent_dir / 'identity.json'),
                'cohort': cohort, 'root_records': len(aligned), 'adapter': adapter.receipt,
                'neural_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False}
            write(output / 'identity.json', identity)
            write(output / 'trained-linear.json', {'state': adapter.state(), 'receipt': adapter.receipt,
                'temperature_search': temperature_search, 'final_calibration_training_only': True,
                'root_emissions_do_not_use_final_temperature': True})
            (output / 'root-oof-alignment.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in aligned))
            (output / 'adapter.pkl').write_bytes(pickle.dumps(adapter, protocol=5))
            models = []
            for selected in [r for r in complete['models'] if r['semantic_review']]:
                source_path = parent_dir / (selected['key'] + '.pkl')
                parent = pickle.loads(source_path.read_bytes()); before = preserved_digest(parent)
                if not parent.mentor.fit_receipt['semantic_review'] or parent.mentor.review_observer is None:
                    raise ValueError('Require the registered semantic-review parent')
                recipe = read(Path(parent.mentor.classifier.directory) / 'recipe.json')
                encoder_price = recipe['parameters'] / 1e9 * recipe['max_length'] / 3.09
                expected_costs = rebuild(aligned, parent.mentor.dynamics, selected['smoothing'], encoder_price)
                updated = rebuild(aligned, parent.mentor.dynamics, selected['smoothing'])
                for subject, dynamics in parent.mentor.dynamics.items():
                    np.testing.assert_allclose(dynamics.review_costs, expected_costs[subject][2], rtol=1e-13, atol=1e-10)
                    dynamics.review, dynamics.reviewed_transition, dynamics.review_costs = updated[subject]
                if preserved_digest(parent) != before:
                    raise ValueError('A protected parent component changed')
                parent.mentor.fit_receipt.update(semantic_review=False,
                    replaced_review_observation='paid-metadata-logistic-v1',
                    root_oof_alignment_sha256=sha(output / 'root-oof-alignment.jsonl'),
                    metadata_fit_identity_sha256=sha(output / 'identity.json'),
                    child_calibration_and_emissions_unchanged=True)
                model = runtime.build(parent, adapter, complete['base_A']['settings'],
                    {'parent_model_sha256': selected['sha256'], 'identity_sha256': sha(output / 'identity.json')})
                if preserved_digest(model) != before:
                    raise ValueError('Runtime wrapping changed protected components')
                key = f"metadata-smooth{selected['smoothing']:g}"; path = output / (key + '.pkl')
                path.write_bytes(pickle.dumps(model, protocol=5))
                if preserved_digest(pickle.loads(path.read_bytes())) != before:
                    raise ValueError('Saved runtime changed protected component values')
                models.append({'key': key, 'sha256': sha(path), 'parent_sha256': selected['sha256'],
                    'smoothing': selected['smoothing'], 'semantic_review': False, 'metadata_review': True,
                    'policy_module': 'metadata_review_hybrid', 'preserved_components_sha256': before,
                    'removed_conservative_review_encoder_compute': encoder_price})
                print(json.dumps({'fit': cohort, 'model': key, 'root_records': len(aligned), 'temperature': temperature}), flush=True)
            verify_hashes(sources); verify_hashes(evidence)
            write(output / 'complete.json', {'models': models, 'base_A': complete['base_A'],
                'identity_sha256': sha(output / 'identity.json'), 'registration_sha256': sha(output / 'registration.json'),
                'artifacts': {name: sha(output / name) for name in ('trained-linear.json', 'adapter.pkl', 'root-oof-alignment.jsonl')},
                'fit_questions': complete['fit_questions'], 'paired_review_questions': len(aligned),
                'policy_module': 'metadata_review_hybrid', 'neural_calls': 0, 'cloud_calls': 0,
                'development_or_test_used': False, 'cpu_metadata_timing_reported_separately': True})
            print(json.dumps({'complete': cohort, 'sha256': sha(output / 'complete.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'fit'))
    for name in ('diagnostic', 'old-data', 'new-data', 'parent-models', 'mixed-parent',
                 'newsource-parent', 'newsource-raw', 'mixed-output', 'newsource-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
