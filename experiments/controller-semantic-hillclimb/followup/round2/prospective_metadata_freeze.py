"""Explicit metadata-C prospective provenance; frozen legacy files stay intact.

This extension preserves the original strongest-A/B selection and target gates.
It verifies training artifacts and saved learned values without neural or cloud
inference. A separate registered metadata runner must audit every paid prefix.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np

import prospective_freeze as legacy
import prospective_newsource_freeze as inherited
import fit_metadata_review as fitting
import metadata_review_hybrid as policy
import paid_review_metadata as paid
import review_metadata_diagnostic as diagnostic
import review_metadata_spec as spec

HERE = Path(__file__).resolve().parent
SCHEMA = 'round2-prospective-metadata-C-provenance-v1'
MODULE = 'metadata_review_hybrid'
read, require, fingerprint = inherited.read, inherited.require, inherited.fingerprint
sha = legacy.sha
ORIGINAL_ROLE = legacy.validate_role
CACHE_FIELDS = {'_model', '_tokenizer', '_recipe', '_cache', '_retriever', '_backgrounds', '_alphas', '_asset_pool'}


def structural_digest(value):
    """Saved class/shape/dtype/value identity, never reserialized pickle identity."""
    active = set()
    def encode(value):
        if value is None or type(value) in (bool, int, str):
            return [type(value).__name__, value]
        if isinstance(value, (float, np.floating)):
            require(np.isfinite(value), 'Nonfinite learned scalar')
            return ['float', float(value).hex()]
        if isinstance(value, np.integer): return ['int', int(value)]
        if isinstance(value, np.bool_): return ['bool', bool(value)]
        if isinstance(value, np.ndarray):
            if value.dtype.fields:
                return ['structured-array', list(value.shape), [[name, encode(value[name])] for name in value.dtype.names]]
            if value.dtype.hasobject: return ['object-array', list(value.shape), encode(value.tolist())]
            if np.issubdtype(value.dtype, np.number): require(np.isfinite(value).all(), 'Nonfinite learned array')
            return ['array', value.dtype.str, list(value.shape), hashlib.sha256(value.tobytes(order='C')).hexdigest()]
        if isinstance(value, np.dtype): return ['dtype', value.str]
        if isinstance(value, Path): return ['path', str(value)]
        if isinstance(value, type): return ['type', value.__module__, value.__qualname__]
        marker = id(value); require(marker not in active, 'Cyclic learned structure'); active.add(marker)
        try:
            if isinstance(value, dict):
                pairs = [[encode(k), encode(v)] for k, v in value.items()]
                return ['dict', sorted(pairs, key=lambda item: json.dumps(item[0], sort_keys=True))]
            if isinstance(value, (tuple, list)): return [type(value).__name__, [encode(v) for v in value]]
            if hasattr(value, '__dict__'):
                return ['object', type(value).__module__, type(value).__qualname__,
                        [[key, encode(item)] for key, item in sorted(vars(value).items()) if key not in CACHE_FIELDS]]
            raise ValueError('Unknown learned-state value: ' + str(type(value)))
        finally:
            active.remove(marker)
    return fingerprint(encode(value))


def bind(mapping, files):
    for name, expected in mapping.items():
        inherited.digest_file(name, expected, files)


def validate_role(model, family, module):
    if module != MODULE:
        return ORIGINAL_ROLE(model, family, module)
    require(family == 'C' and type(model) is policy.Planner, 'Metadata runtime is an explicit hybrid C only')
    receipt = model.fit_receipt
    require(receipt.get('policy_module') == MODULE and receipt.get('state_count') == 16
            and receipt.get('maximum_reviews') == 1 and receipt.get('include_reviews') is False
            and receipt.get('parent_transfer_unchanged') is True,
            'Metadata C changed its two-mode POMDP/review contract')
    require(type(model.metadata_adapter) is paid.Adapter, 'Unknown metadata correctness adapter')
    model.metadata_adapter.validate()
    require(model.review_observer is None and model.mentor.review_observer is None
            and model.mentor.fit_receipt.get('semantic_review') is False,
            'Metadata replacement must actually remove the review neural observer')
    ORIGINAL_ROLE(model.base, 'A', model.base_module)
    require(receipt.get('metadata_adapter') == model.metadata_adapter.receipt,
            'Metadata adapter differs from the runtime receipt')
    for _, observer in legacy.observer_slots(model):
        require(getattr(observer, '_model', None) is None, 'Archive unloaded observer assets before freezing')


def validate_adapter(adapter, linear, identity, contract_sha):
    require(type(adapter) is paid.Adapter, 'Unexpected saved metadata adapter type')
    adapter.validate()
    require(adapter.state() == linear['state'] and adapter.receipt == linear['receipt'] == identity['adapter'],
            'Saved scaler/coefficient/temperature state differs from fitting evidence')
    receipt = adapter.receipt
    require(receipt.get('training_questions') == 192 and receipt.get('training_records') == 320
            and receipt.get('registration_sha256') == contract_sha
            and receipt.get('root_tables_use_nested_oof') is True and receipt.get('neural_calls') == 0
            and receipt.get('runtime_temperature') == adapter.temperature,
            'Metadata supervised/calibration cohort contract changed')
    require(linear.get('final_calibration_training_only') is True
            and linear.get('root_emissions_do_not_use_final_temperature') is True,
            'Root POMDP tables must use held-question probabilities')
    require([r['temperature'] for r in linear['temperature_search']] == spec.TEMPERATURES
            and adapter.temperature == min(linear['temperature_search'], key=lambda r: (r['loss'], r['temperature']))['temperature'],
            'Changed fixed-grid metadata temperature selection')


def diagnostic_evidence(directory, files):
    """Validate fixed nested TRAIN-only forecasts, without fitting any models."""
    directory = Path(directory)
    summary, contract = read(directory/'summary.json'), read(directory/'registration.json')
    implementation = read(directory/'implementation.json')
    require(summary.get('strict_nested_question_calibration') is True
            and summary.get('development_or_test_used') is False
            and contract.get('specification_sha256') == sha(spec.__file__)
            and contract.get('features') == spec.FEATURES and contract.get('seed') == spec.SEED,
            'Diagnostic must be the fixed, training-only paid-metadata protocol')
    checks = {'registration.json': 'registration_sha256', 'implementation.json': 'implementation_sha256',
              'private-features.jsonl': 'private_features_sha256', 'private-predictions.npz': 'predictions_sha256',
              'fold-receipts.json': 'fold_receipts_sha256'}
    for name, field in checks.items(): inherited.digest_file(directory/name, summary[field], files)
    files.update((directory/'summary.json', directory/'numerical-audit.json'))
    bind(implementation['source_files'], files); bind(implementation['input_evidence'], files)
    records = [json.loads(line) for line in (directory/'private-features.jsonl').read_text().splitlines()]
    require(fingerprint(records) == implementation['records_sha256'] and len(records) == 320
            and len({r['id'] for r in records}) == 192
            and Counter(r['protocol'] for r in records) == {'root': 192, 'critical': 64, 'independent': 64},
            'Diagnostic omitted or filtered a sibling training observation')
    require(len({(r['id'], r['protocol']) for r in records}) == len(records)
            and all(len(r['features']) == len(spec.FEATURES) and np.isfinite(r['features']).all() for r in records),
            'Invalid or duplicate paid feature rows')
    ids = {r['id'] for r in records}; raw = np.full(len(records), np.nan)
    with np.load(directory/'private-predictions.npz', allow_pickle=False) as archive:
        nested = archive['metadata_logistic'].copy()
    require(nested.shape == (320,) and np.isfinite(nested).all(), 'Invalid nested metadata predictions')
    folds = read(directory/'fold-receipts.json')['metadata_logistic']
    require(len(folds) == 3, 'Incomplete outer question folds')
    for fold, receipt in enumerate(folds):
        held = {qid for qid in ids if diagnostic.fold(qid) == fold}
        require(receipt['outer_fold'] == fold and set(receipt['held_ids']) == held
                and set(receipt['training_ids']) == ids-held, 'Outer metadata calibration leaks a question')
        require([r['fold'] for r in receipt['inner']] == [0, 1, 2], 'Incomplete inner question folds')
        for inner in receipt['inner']:
            local = {qid for qid in ids-held if diagnostic.fold(qid, True) == inner['fold']}
            require(set(inner['held_ids']) == local and set(inner['training_ids']) == ids-held-local,
                    'Nested metadata calibration leaks a question')
        path = inherited.digest_file(directory/f'metadata_logistic-outer{fold}.pkl', receipt['model_sha256'], files)
        indexes = [i for i, r in enumerate(records) if r['id'] in held]
        local = [records[i] for i in indexes]
        raw[indexes] = diagnostic.predict(pickle.loads(path.read_bytes()), local)
        np.testing.assert_allclose(diagnostic.calibrated(raw[indexes], receipt['temperature'], local),
                                   nested[indexes], rtol=0, atol=1e-15)
    require(np.isfinite(raw).all(), 'Incomplete outer held-question forecasts')
    return records, raw, nested


class Provenance(inherited.Provenance):
    def __init__(self, args):
        super().__init__(args)
        self.metadata_receipts = {}

    def chain(self, directory, registry_root, seen=None):
        directory = Path(directory).resolve(); identity = read(directory/'identity.json')
        if identity.get('builder_module') == 'stronger_metadata_review':
            return self.stronger(directory, registry_root, seen)
        if identity.get('policy_module') != MODULE:
            return super().chain(directory, registry_root, seen)
        seen = set() if seen is None else seen
        require(directory not in seen, 'C fitting provenance cycle'); seen.add(directory)
        files = {directory/'identity.json', directory/'complete.json', directory/'registration.json'}
        complete, contract = read(directory/'complete.json'), read(directory/'registration.json')
        require(identity['source_sha256'] == sha(fitting.__file__) and contract['recipe'] == fitting.RECIPE
                and complete['policy_module'] == MODULE
                and complete['identity_sha256'] == sha(directory/'identity.json')
                and identity['registration_sha256'] == complete['registration_sha256'] == sha(directory/'registration.json'),
                'Changed metadata fitting recipe or identity')
        require(identity['cohort'] in ('mixed', 'newsource')
                and all(identity.get(k) == 0 and complete.get(k) == 0 for k in ('neural_calls', 'cloud_calls'))
                and identity.get('development_or_test_used') is False and complete.get('development_or_test_used') is False,
                'Metadata fit used unregistered inference or held-out values')
        require(all(contract['source_files'].get(str(Path(module.__file__).resolve())) == sha(module.__file__)
                    for module in (fitting, policy, paid, diagnostic, spec)),
                'Metadata fit omitted a required source dependency')
        bind(contract['source_files'], files); bind(contract['input_evidence'], files)
        require(set(complete['artifacts']) == {'trained-linear.json', 'adapter.pkl', 'root-oof-alignment.jsonl'},
                'Incomplete metadata fitting artifacts')
        bind({str(directory/name): value for name, value in complete['artifacts'].items()}, files)
        records, raw, nested = diagnostic_evidence(contract['diagnostic_directory'], files)
        linear = read(directory/'trained-linear.json'); adapter = pickle.loads((directory/'adapter.pkl').read_bytes())
        validate_adapter(adapter, linear, identity, sha(directory/'registration.json'))
        searches = [{'temperature': t, 'loss': diagnostic.binary_loss(
            [r['target'] for r in records], diagnostic.calibrated(raw, t, records), diagnostic.weights(records))}
            for t in spec.TEMPERATURES]
        require([r['temperature'] for r in searches] == [r['temperature'] for r in linear['temperature_search']],
                'Changed temperature grid')
        np.testing.assert_allclose([r['loss'] for r in searches], [r['loss'] for r in linear['temperature_search']], rtol=1e-13, atol=1e-14)
        aligned = [json.loads(line) for line in (directory/'root-oof-alignment.jsonl').read_text().splitlines()]
        lookup = {r['id']: (r, float(p)) for r, p in zip(records, nested)
                  if r['protocol'] == 'root' and (identity['cohort'] == 'mixed' or r['bank'] == 'new')}
        expected = fitting.RECIPE['root_cohorts'][identity['cohort']]
        require(len(aligned) == len(lookup) == identity['root_records'] == complete['paired_review_questions'] == expected
                and {r['id'] for r in aligned} == set(lookup), 'Root emission cohort changed')
        for row in aligned:
            recorded, probability = lookup[row['id']]
            require(row['nested_probability'] == probability and row['outer_fold'] == diagnostic.fold(row['id'])
                    and row['subject'] == recorded['subject'], 'Root POMDP tables lost excluded-fold alignment')
        parent = Path(identity['parent_models_directory']).resolve()
        inherited.digest_file(parent/'complete.json', identity['parent_complete_sha256'], files)
        inherited.digest_file(parent/'identity.json', identity['parent_identity_sha256'], files)
        ancestor, parent_files = self.chain(parent, registry_root, seen); files.update(parent_files)
        require(complete['base_A'] == ancestor['base_A'] and complete['fit_questions'] == ancestor['fit_questions'],
                'Metadata replacement changed the inherited base/training cohort')
        parents = {r['sha256']: r for r in ancestor['models']}
        require([(r['key'], r['smoothing'], r['metadata_review']) for r in complete['models']] ==
                [('metadata-smooth8', 8., True), ('metadata-smooth32', 32., True)], 'Changed metadata candidate grid')
        learned = {}
        for entry in complete['models']:
            require(entry['semantic_review'] is False and entry['policy_module'] == MODULE
                    and entry['parent_sha256'] in parents, 'Unknown metadata model parent')
            prior = parents[entry['parent_sha256']]
            require(prior['semantic_review'] is True and prior['smoothing'] == entry['smoothing'], 'Wrong semantic parent variant')
            parent_path = inherited.digest_file(parent/(prior['key']+'.pkl'), prior['sha256'], files)
            model_path = inherited.digest_file(directory/(entry['key']+'.pkl'), entry['sha256'], files)
            original, model = pickle.loads(parent_path.read_bytes()), pickle.loads(model_path.read_bytes())
            validate_role(model, 'C', MODULE)
            require(fitting.preserved_digest(original) == fitting.preserved_digest(model) == entry['preserved_components_sha256'],
                    'Saved metadata model changed a protected parent component')
            require(structural_digest(model.metadata_adapter) == structural_digest(adapter), 'Saved model uses another metadata adapter')
            updated = fitting.rebuild(aligned, original.mentor.dynamics, entry['smoothing'])
            for subject, dynamics in model.mentor.dynamics.items():
                for actual, expected_array in zip((dynamics.review, dynamics.reviewed_transition, dynamics.review_costs), updated[subject]):
                    np.testing.assert_allclose(actual, expected_array, rtol=1e-13, atol=1e-12)
            require(model.mentor.fit_receipt['root_oof_alignment_sha256'] == complete['artifacts']['root-oof-alignment.jsonl'],
                    'Saved model lost metadata OOF provenance')
            learned[entry['key']] = structural_digest(model)
        files.add(Path(__file__).resolve())
        self.metadata_receipts[str(directory)] = {'cohort': identity['cohort'], 'root_records': expected,
            'adapter_training_questions': 192, 'adapter_training_sibling_records': 320,
            'identity_sha256': sha(directory/'identity.json'), 'complete_sha256': sha(directory/'complete.json'),
            'saved_model_structural_sha256': learned, 'nested_question_folds': 3,
            'historical_byte_identity_claim': False, 'neural_calls': 0, 'cloud_calls': 0}
        return complete, sorted(files)

    def stronger(self, directory, registry_root, seen):
        import stronger_metadata_review as builder
        import development_stronger_metadata as driver
        seen = set() if seen is None else seen
        require(directory not in seen, 'Metadata construction provenance cycle'); seen.add(directory)
        identity, complete = read(directory/'identity.json'), read(directory/'complete.json')
        files = {directory/'identity.json', directory/'complete.json', directory.parent/'registration.json',
                 Path(builder.__file__).resolve(), Path(driver.__file__).resolve(), Path(__file__).resolve()}
        require(complete['identity_sha256'] == sha(directory/'identity.json')
                and identity['registration_sha256'] == sha(directory.parent/'registration.json')
                and identity['source_sha256'] == sha(driver.__file__)
                and identity['builder_sha256'] == sha(builder.__file__)
                and identity['policy_sha256'] == sha(policy.__file__)
                and complete['policy_module'] == identity['policy_module'] == MODULE
                and complete['builder_module'] == identity['builder_module'] == 'stronger_metadata_review',
                'Strongest-A metadata construction identity changed')
        require(all(identity.get(k) is False and complete.get(k) is False
                    for k in ('mentor_refitted','transfer_refitted','metadata_adapter_refitted','test_used'))
                and complete.get('neural_calls') == complete.get('cloud_calls') == 0,
                'Metadata base replacement changed retained fitted components or used test data')
        contract = read(directory.parent/'registration.json'); bind(contract['source_files'], files)
        selected = identity['base_selection']
        require(complete['base_selection'] == selected and complete['base_A'] == selected['candidate'],
                'Constructed C lost its selected base identity')
        bind({selected['summary']: selected['summary_sha256'],
              str(Path(selected['summary']).parent/'identity.json'): selected['identity_sha256'],
              selected['model_path']: selected['model_sha256']}, files)
        require(read(selected['summary'])['selected_A'] == selected['candidate'], 'Selected base summary changed')
        base = pickle.loads(Path(selected['model_path']).read_bytes()); ORIGINAL_ROLE(base, 'A', selected['module'])
        require(set(identity['parent_manifests']) == {'mixed','newsource'}
                and complete['parent_manifests'] == identity['parent_manifests'], 'Both metadata parent cohorts must remain registered')
        parents = {}
        for cohort, item in identity['parent_manifests'].items():
            parent = Path(item['directory']).resolve()
            inherited.digest_file(parent/'complete.json', item['sha256'], files)
            ancestor, parent_files = self.chain(parent, registry_root, set(seen)); files.update(parent_files)
            require(read(parent/'identity.json')['cohort'] == cohort, 'Swapped metadata parent cohorts')
            parents[cohort] = (parent, {r['sha256']: r for r in ancestor['models']})
        expected = {(cohort, smoothing) for cohort in ('mixed','newsource') for smoothing in (8.,32.)}
        require(len(complete['models']) == 4 and {(r['cohort'],r['smoothing']) for r in complete['models']} == expected,
                'Preserve all four strongest-base metadata candidates')
        learned = {}
        for record in complete['models']:
            cohort = record['cohort']; parent_dir, choices = parents[cohort]
            require(record['metadata_review'] is True and record['policy_module'] == MODULE
                    and Path(record['parent_metadata_directory']).resolve() == parent_dir
                    and record['parent_sha256'] in choices, 'Unknown constructed metadata parent')
            prior = choices[record['parent_sha256']]
            require(prior['smoothing'] == record['smoothing'], 'Constructed model changed smoothing ancestry')
            prior_path = inherited.digest_file(parent_dir/(prior['key']+'.pkl'), prior['sha256'], files)
            path = inherited.digest_file(directory/(record['key']+'.pkl'), record['sha256'], files)
            original, model = pickle.loads(prior_path.read_bytes()), pickle.loads(path.read_bytes())
            validate_role(model, 'C', MODULE)
            require(builder.retained_state(original) == builder.retained_state(model) == record['retained_state_sha256']
                    and fitting.preserved_digest(model) == record['base_and_child_state_sha256']
                    and structural_digest(model.base) == structural_digest(base),
                    'Saved stronger-base metadata model changed a learned parent/base component')
            require(model.base_module == selected['module'] and model.base_settings == selected['candidate'].get('settings', {})
                    and model.fit_receipt.get('builder_module') == 'stronger_metadata_review'
                    and model.fit_receipt.get('parent_retained_state_sha256') == record['retained_state_sha256'],
                    'Saved runtime lost explicit metadata/base-selection provenance')
            learned[record['key']] = structural_digest(model)
        for asset in complete['observer_assets']:
            bind({asset['recipe_path']: asset['recipe_sha256'], **asset['weights']}, files)
        self.metadata_receipts[str(directory)] = {'kind':'strongest-base-metadata',
            'identity_sha256':sha(directory/'identity.json'), 'complete_sha256':sha(directory/'complete.json'),
            'base_selection':selected, 'saved_model_structural_sha256':learned,
            'metadata_mentor_transfer_preserved':True, 'historical_byte_identity_claim':False}
        return complete, sorted(files)

    @contextmanager
    def installed(self):
        original_select = legacy.select_inputs
        def select(args):
            result = original_select(args)
            selected, _, _, paths, _, registry = result
            require(selected['C']['module'] == MODULE and self.metadata_receipts,
                    'This extension requires an explicitly fitted metadata C')
            identity = read(args.c_summary.parent/'identity.json')
            stronger = identity.get('builder_module') == 'stronger_metadata_review'
            if stronger:
                summary = read(args.c_summary)
                base = read(args.c_models/'identity.json')['base_selection']
                require(summary.get('inherited_base_is_strongest_completed_A') is True
                        and base['model_sha256'] == sha(paths['A']) and base['module'] == selected['A']['module']
                        and base['candidate'].get('settings', {}) == selected['A']['settings'],
                        'Metadata C must retain the current strongest selected A before freezing')
            driver = importlib.import_module('development_stronger_metadata' if stronger else 'development_metadata_review')
            marker = args.c_summary.parent/('stronger-metadata-replay-complete.json' if stronger else 'metadata-replay-complete.json')
            require(marker.exists() and read(marker) == driver.seal_payload(args.c_summary.parent),
                    'Finish and seal the selected metadata development comparison before freezing')
            registry['evidence_files'].append(str(marker.resolve()))
            registry['evidence_files'].extend(str((args.c_summary.parent/name).resolve()) for name in read(marker)['files'])
            model = pickle.loads(paths['C'].read_bytes()); validate_role(model, 'C', MODULE)
            registry['metadata_C_provenance'] = {'schema': SCHEMA, 'validator_sha256': sha(__file__),
                'fitting_chains': self.metadata_receipts, 'newsource_ancestor_chains': self.receipts,
                'selected_adapter': {'state': model.metadata_adapter.state(), 'receipt': model.metadata_adapter.receipt},
                'selected_model_structural_sha256': structural_digest(model),
                'required_runner': 'prospective_metadata_parallel.py',
                'required_audit': 'metadata_paid_audit.py',
                'selection_and_targets_unchanged': True,
                'no_test_question_response_values_read_by_extension': True,
                'neural_calls': 0, 'cloud_calls': 0, 'historical_byte_identity_claim': False}
            registry['evidence_files'].append(str(Path(__file__).resolve()))
            return result
        modules = {**legacy.C_MODULES, MODULE: ('development_metadata_review', 'development_stronger_metadata')}
        with patch.object(legacy, 'C_MODULES', modules), patch.object(legacy, 'validate_role', validate_role), \
                patch.object(legacy, 'fitting_chain', self.chain), patch.object(legacy, 'select_inputs', select):
            yield


def freeze(args):
    with Provenance(args).installed():
        legacy.freeze(args)
    verify(args.output, sha(args.output/'frozen.json'))


def verify(output, expected_sha=None):
    frozen = legacy.verify(output, expected_sha)
    receipt = frozen['completed_A_B_registry'].get('metadata_C_provenance', {})
    require(frozen['selected']['C']['module'] == MODULE and receipt.get('schema') == SCHEMA
            and receipt.get('validator_sha256') == sha(__file__)
            and receipt.get('required_runner') == 'prospective_metadata_parallel.py',
            'Missing explicit metadata-C prospective provenance')
    require(paid.digest(receipt['selected_adapter']['state']) == receipt['selected_adapter']['receipt']['learned_state_sha256'],
            'Frozen metadata learned state changed')
    return frozen


def load_models(output, device='cpu', expected_sha=None):
    frozen = verify(output, expected_sha)
    with patch.object(legacy, 'validate_role', validate_role):
        _, models = legacy.load_models(output, device, expected_sha)
    adapter = frozen['completed_A_B_registry']['metadata_C_provenance']['selected_adapter']
    require(models['C'].metadata_adapter.state() == adapter['state'] and models['C'].metadata_adapter.receipt == adapter['receipt'],
            'Archived runtime metadata state differs from its frozen learned state')
    return frozen, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'verify'))
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--expected-freeze-sha256')
    for name in ('data', 'a-summary', 'b-summary', 'c-summary', 'a-models', 'c-models', 'mentor-bank', 'verifier', 'legacy-examples'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--a-module', choices=tuple(legacy.A_MODULES)); args = parser.parse_args()
    if args.phase == 'verify':
        verify(args.output, args.expected_freeze_sha256)
    else:
        required = ('data','a_summary','b_summary','c_summary','a_models','c_models','mentor_bank','verifier','legacy_examples','a_module')
        if any(getattr(args, name) is None for name in required): parser.error('Require all registered selection/training paths')
        freeze(args)


if __name__ == '__main__': main()
