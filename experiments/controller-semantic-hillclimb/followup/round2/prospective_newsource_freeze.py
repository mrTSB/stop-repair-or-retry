"""Strict provenance extension for the source-matched retained-mentor fits.

The old freeze, selection rules, runtime classes, and test gates are unchanged.
This entrypoint temporarily extends only fitting provenance dispatch and adds
its validation evidence to the manifest before the old freezer writes it.
No fitting, inference, development evaluation, or test grading occurs here.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np

import prospective_freeze as legacy

HERE = Path(__file__).resolve().parent
FITTER = HERE / 'fit_new_source_mentor.py'
CORE = HERE / 'new_source_mentor_core.py'
SCHEMA = 'round2-newsource-mentor-provenance-v1'
MODEL_SETTINGS = {(f'smooth{s:g}-review{int(review)}', float(s), review)
                  for s in (8, 32) for review in (False, True)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path, expected, files):
    path = Path(path).resolve()
    require(legacy.sha(path) == expected, 'New-source provenance file changed: ' + str(path))
    files.add(path)
    return path


def source_evidence(identity, files):
    require(identity.get('source_sha256') == legacy.sha(FITTER), 'New-source fitting entrypoint changed')
    sources = identity.get('source_files', {})
    required = [FITTER, CORE, HERE / 'crossfit_mentor_emissions.py', HERE / 'prepare_mixed_mentor.py']
    require(all(sources.get(str(path.resolve())) == legacy.sha(path) for path in required),
            'New-source fit omitted a required source dependency')
    for name, expected in sources.items():
        path = Path(name).resolve()
        project_python = path.is_relative_to(legacy.ROOT) and path.suffix == '.py'
        environment_extension = (path.is_relative_to(legacy.ROOT / '.venv')
                                 and 'site-packages' in path.parts and path.suffix in ('.so', '.dylib', '.pyd'))
        # The completed fitter's actual import inventory includes a repo-local
        # pyarrow binary. Preserve and verify it as an environment dependency;
        # neither rewrite the old receipt nor silently drop its recorded hash.
        require(project_python or environment_extension, 'Unknown fitting source path')
        digest_file(path, expected, files)
    files.add(Path(__file__).resolve())


def complete_models(directory, complete, refined, files):
    require(complete.get('new_source_only') is True and complete.get('fit_questions') == 512
            and complete.get('paired_review_questions') == 128
            and complete.get('neural_calls') == 0 and complete.get('cloud_calls') == 0
            and complete.get('development_or_test_used') is False,
            'Source-matched fit must retain all512 new training rows and128 pairs without inference')
    records = complete.get('models', [])
    require(len(records) == 4 and {(r['key'], r['smoothing'], r['semantic_review']) for r in records} == MODEL_SETTINGS
            and all(type(r['semantic_review']) is bool for r in records), 'Preserve all four registered new-source models')
    for record in records:
        digest_file(directory / (record['key'] + '.pkl'), record['sha256'], files)
        require(('parent_sha256' in record) == refined, 'Changed raw/refined model provenance')
    return {record['key']: record for record in records}


def check_selection(identity, contract, public_rows, legacy_ids):
    require(len(public_rows) == 512 and len({r['id'] for r in public_rows}) == 512
            and Counter(r['subject'] for r in public_rows) == {'math': 256, 'chemistry': 128, 'physics': 128},
            'Changed full new training membership')
    selected, excluded = identity['selected_pairs'], identity['excluded_pairs']
    original = contract['selection']
    require(len(original) == 192 and len({r['id'] for r in original}) == 192
            and selected == [r for r in original if r['bank'] == 'new']
            and excluded == [r for r in original if r['bank'] == 'old']
            and len(selected) == 128 and len(excluded) == 64,
            'Source selection must retain every registered new pair and exclude every old pair uniformly')
    by_id = {r['id']: r for r in public_rows}
    ids, chosen, omitted = set(by_id), {r['id'] for r in selected}, {r['id'] for r in excluded}
    require(chosen <= ids and not omitted & ids and not legacy_ids & ids
            and omitted <= legacy_ids
            and Counter(r['subject'] for r in selected) == {'math': 64, 'chemistry': 32, 'physics': 32}
            and all(by_id[r['id']]['subject'] == r['subject'] for r in selected),
            'New-source training/observer membership or subject mapping changed')
    return ids, chosen


def verify_neural_subset(directory, parent, identity, ids, selected, files):
    from difficulty_pomdp import PLAIN
    metadata_path = directory / 'metadata.jsonl'
    outputs_path = directory / 'neural.npz'
    receipt_path = directory / 'neural-receipt.json'
    receipt = read(receipt_path)
    files.add(receipt_path.resolve())
    digest_file(metadata_path, receipt['metadata_sha256'], files)
    digest_file(outputs_path, receipt['outputs_sha256'], files)
    require(receipt['parent_neural_receipt_sha256'] == identity['parent_neural_receipt_sha256']
            and receipt['input_sha256'] == identity['input_sha256'] and receipt['n'] == identity['input_rows'] == 4224,
            'Subset neural receipt differs from the source-matched fit')
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    parent_metadata = [json.loads(line) for line in (parent / 'metadata.jsonl').read_text().splitlines()]
    key = lambda r: (r['id'], r['kind'], r['path'])
    parent_index = {key(row): i for i, row in enumerate(parent_metadata)}
    require(len(parent_index) == len(parent_metadata), 'Duplicate parent neural metadata')
    wanted = {(qid, 'solver', path) for qid in ids for path in PLAIN}
    wanted |= {(qid, 'solver', path) for qid in selected for path in ('guided', 'guided/reuse', 'guided/retry', 'retry/reuse')}
    wanted |= {(qid, 'original-review', '') for qid in selected}
    require(len(metadata) == 4224 and {key(row) for row in metadata} == wanted,
            'Subset omitted, added, or duplicated a registered training observation')
    indexes = []
    for row in metadata:
        index = parent_index.get(key(row))
        require(index is not None and row == parent_metadata[index] and row['fold'] == 'full',
                'Subset label/prompt/input/fold alignment differs from the archived parent')
        indexes.append(index)
    require(sum(r['id'] in ids for r in parent_metadata) == len(metadata)
            and fingerprint(indexes) == identity['reused_parent_indexes_sha256'], 'Changed exact parent subset indexes')
    with np.load(outputs_path, allow_pickle=False) as subset, np.load(parent / 'neural.npz', allow_pickle=False) as full:
        require(set(subset.files) == set(full.files) == {'state', 'plain', 'guided'}, 'Changed neural output fields')
        for name, width in (('state', 4), ('plain', 8), ('guided', 8)):
            require(subset[name].shape == (len(metadata), width)
                    and full[name].shape == (len(parent_metadata), width)
                    and np.isfinite(subset[name]).all()
                    and np.array_equal(subset[name], full[name][indexes]), 'Subset neural arrays differ from parent outputs')
    return metadata


def verify_folds(identity, metadata, path, files):
    solver = [r for r in metadata if r['kind'] == 'solver']
    folds = [int(hashlib.sha256(r['id'].encode()).hexdigest()[:8], 16) % 3 for r in solver]
    expected = []
    for fold in (0, 1, 2):
        fit = [r for r, f in zip(solver, folds) if f != fold and r['train'] and r['valid']]
        held = [r for r, f in zip(solver, folds) if f == fold]
        fit_ids, held_ids = {r['id'] for r in fit}, {r['id'] for r in held}
        require(fit_ids and held_ids and not fit_ids & held_ids, 'Invalid whole-question calibration fold')
        expected.append({'fold': fold, 'fit_questions': len(fit_ids), 'held_questions': len(held_ids),
                         'fit_rows': len(fit), 'held_rows': len(held), 'question_overlap': 0,
                         'fit_ids_sha256': fingerprint(sorted(fit_ids)), 'held_ids_sha256': fingerprint(sorted(held_ids))})
    require(identity['folds'] == expected, 'Question-excluded calibration fold receipts changed')
    digest_file(path, identity['calibration_oof_sha256'], files)
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {'probabilities'}, 'Unexpected calibration output fields')
        p = archive['probabilities']
        require(p.shape == (len(solver), 4) and np.isfinite(p).all() and (p >= 0).all()
                and (p <= 1).all() and np.allclose(p.sum(1), 1.), 'Invalid question-excluded probabilities')
    return {'solver_rows': len(solver), 'folds': 3, 'question_overlap': 0,
            'neural_training_exclusion': 'All new questions excluded from old full observer supervision.',
            'calibration_exclusion': 'Three whole-question folds for emission forecasts; no nested neural/calibration CV claim.'}


def load_model(path):
    # Only reached after complete_models has verified every registered pickle.
    model = pickle.loads(path.read_bytes())
    legacy.validate_role(model, 'C', 'hybrid_repairability')
    return model


def preserved_state(model):
    """Canonical learned values, independent of pickle aliases/NumPy strides.

The selected components match the fitter's preservation contract. Runtime
memoization and loaded encoder assets are excluded and separately hash-bound;
the allowed changed emission arrays and mentor provenance are not selected.
"""
    caches = {'_model', '_tokenizer', '_recipe', '_cache', '_retriever', '_backgrounds', '_alphas'}
    active, counts = set(), {'arrays': 0, 'array_values': 0}
    def encode(value):
        if value is None or type(value) in (bool, int, str):
            return [type(value).__name__, value]
        if isinstance(value, (float, np.floating)):
            require(np.isfinite(value), 'Nonfinite preserved model scalar')
            return ['float', float(value).hex()]
        if isinstance(value, np.integer):
            return ['int', int(value)]
        if isinstance(value, np.bool_):
            return ['bool', bool(value)]
        if isinstance(value, np.ndarray):
            counts['arrays'] += 1; counts['array_values'] += value.size
            if value.dtype.fields:
                return ['structured-array', list(value.shape), [[name, encode(value[name])] for name in value.dtype.names]]
            if value.dtype.hasobject:
                return ['object-array', list(value.shape), encode(value.tolist())]
            if np.issubdtype(value.dtype, np.number):
                require(np.isfinite(value).all(), 'Nonfinite preserved model array')
            return ['array', value.dtype.str, list(value.shape), hashlib.sha256(value.tobytes(order='C')).hexdigest()]
        if isinstance(value, np.dtype):
            return ['dtype', value.str]
        if isinstance(value, Path):
            return ['path', str(value)]
        if isinstance(value, type):
            return ['type', value.__module__, value.__qualname__]
        marker = id(value)
        require(marker not in active, 'Cyclic preserved model structure')
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
            raise ValueError('Unsupported preserved model value type: ' + str(type(value)))
        finally:
            active.remove(marker)
    dynamics = {subject: {name: value for name, value in vars(d).items()
                         if name not in ('emission', 'mentor', '_alphas')}
                for subject, d in model.mentor.dynamics.items()}
    structure = encode((model.base, model.transfer, model.mentor.regular,
                        model.mentor.classifier, model.mentor.review_observer, dynamics))
    return fingerprint(structure), counts


class Provenance:
    def __init__(self, args):
        self.args = args
        self.original_chain = legacy.fitting_chain
        self.receipts = {}
        self.verified_metadata = {}

    def chain(self, directory, registry_root, seen=None):
        directory = Path(directory).resolve()
        identity = read(directory / 'identity.json')
        raw = 'parent_models_directory' in identity and 'selected_pairs' in identity
        refined = 'new_source_parent_identity_sha256' in identity
        if not (raw or refined):
            # Legacy recursion resolves this adapter while the scope is active,
            # so stronger-base wrappers can inherit either new-source variant.
            return self.original_chain(directory, registry_root, seen)
        seen = set() if seen is None else seen
        require(directory not in seen, 'C fitting provenance contains a cycle')
        seen.add(directory)
        files = {directory / 'complete.json', directory / 'identity.json'}
        complete = read(directory / 'complete.json')
        require(complete['identity_sha256'] == legacy.sha(directory / 'identity.json'), 'New-source fitting identity changed')
        source_evidence(identity, files)
        specs = complete_models(directory, complete, refined, files)
        if raw:
            details = self.raw(directory, identity, complete, specs, files, registry_root, seen)
        else:
            details = self.refined(directory, identity, complete, specs, files, registry_root, seen)
        self.receipts[str(directory)] = {'kind': 'question-excluded-emissions' if refined else 'new-source-raw',
            'identity_sha256': legacy.sha(directory / 'identity.json'),
            'complete_sha256': legacy.sha(directory / 'complete.json'), 'models_checked': 4,
            'fit_questions': 512, 'paired_review_questions': 128, **details}
        return complete, sorted(files)

    def raw(self, directory, identity, complete, specs, files, registry_root, seen):
        args = self.args
        require(identity.get('fit_questions') == 512 and identity.get('paired_review_questions') == 128
                and identity.get('neural_inference_calls') == 0 and identity.get('cloud_calls') == 0
                and identity.get('development_or_test_used') is False
                and identity.get('all_retained_neural_rows_from_old_full_unseen_model') is True
                and identity.get('smoothing') == [8., 32.] and identity.get('semantic_review') == [False, True],
                'Changed new-source training-only fitting contract')
        parent = Path(identity['parent_models_directory']).resolve()
        for filename, field in (('complete.json', 'parent_complete_sha256'), ('identity.json', 'parent_identity_sha256'),
                                ('neural-receipt.json', 'parent_neural_receipt_sha256'),
                                ('metadata.jsonl', 'parent_neural_metadata_sha256'), ('neural.npz', 'parent_neural_sha256')):
            digest_file(parent / filename, identity[field], files)
        original, inherited = self.chain(parent, registry_root, seen)
        files.update(inherited)
        require(original['development_or_test_used'] is False and complete['base_A'] == original['base_A'],
                'Source refit changed its declared original A or used heldout data')
        parent_identity, neural = read(parent / 'identity.json'), read(parent / 'neural-receipt.json')
        require(neural['metadata_sha256'] == identity['parent_neural_metadata_sha256']
                and neural['outputs_sha256'] == identity['parent_neural_sha256']
                and neural['encoder_recipes']['full'] == identity['encoder_recipe_sha256'], 'Parent neural completion changed')
        for name in ('mentor_bank_sha256', 'mentor_contract_sha256', 'legacy_examples_sha256', 'base_model_sha256'):
            require(identity[name] == parent_identity[name], 'New-source parent input anchor changed: ' + name)
        require(identity['new_train_question_sha256'] == parent_identity['new_train_split_sha256'], 'Changed original new-source training panel')
        for path, expected in ((args.mentor_bank / 'contract.json', identity['mentor_contract_sha256']),
                               (args.mentor_bank / 'train.jsonl', identity['mentor_bank_sha256']),
                               (args.data / 'splits/train-questions.jsonl', identity['new_train_question_sha256']),
                               (args.legacy_examples, identity['legacy_examples_sha256']),
                               (identity['base_model_path'], identity['base_model_sha256']),
                               (args.verifier / 'full/recipe.json', identity['encoder_recipe_sha256']),
                               (args.verifier / 'training-summary.json', identity['encoder_training_summary_sha256'])):
            digest_file(path, expected, files)
        bank_complete = read(args.mentor_bank / 'complete.json')
        require((bank_complete['bank_sha256'], bank_complete['contract_sha256']) ==
                (identity['mentor_bank_sha256'], identity['mentor_contract_sha256']), 'Registered mentor bank is not complete')
        files.add((args.mentor_bank / 'complete.json').resolve())
        recipe = read(args.verifier / 'full/recipe.json')
        temperature = read(args.verifier / 'training-summary.json')['temperature_selected_on_training_oof']
        require(recipe['fold'] == 'full' and recipe['input_profile'] == 'full-options-openstax-v1'
                and recipe['data_sha256'] == identity['legacy_examples_sha256']
                and recipe['corpus_sha256'] == identity['corpus_sha256']
                and np.isfinite(temperature) and temperature > 0 and temperature == identity['inherited_temperature'],
                'Old-only observer profile or inherited calibration changed')
        import mentor_pomdp
        corpus = Path(mentor_pomdp.perception.CORPUS)
        for name in ('passages.jsonl', 'tfidf.pkl'):
            digest_file(corpus / name, identity['corpus_sha256' if name == 'passages.jsonl' else 'tfidf_sha256'], files)
        from prospective_collect import parse_public_line
        public = [parse_public_line(line) for line in (args.data / 'splits/train-questions.jsonl').read_text().splitlines()]
        old_ids = {json.loads(line)['id'] for line in args.legacy_examples.read_text().splitlines()}
        ids, chosen = check_selection(identity, read(args.mentor_bank / 'contract.json'), public, old_ids)
        metadata = verify_neural_subset(directory, parent, identity, ids, chosen, files)
        self.verified_metadata[directory] = metadata
        anchor = original['models'][0]
        require(identity['temperature_parent_model_sha256'] == anchor['sha256'], 'Inherited-temperature parent changed')
        anchor_path = digest_file(parent / (anchor['key'] + '.pkl'), anchor['sha256'], files)
        require(load_model(anchor_path).mentor.classifier.temperature == temperature, 'Original saved observer temperature changed')
        for spec in specs.values():
            model = load_model(directory / (spec['key'] + '.pkl'))
            receipt, mentor = model.fit_receipt, model.mentor.fit_receipt
            require(receipt.get('fit_module') == 'new_source_mentor_core'
                    and receipt.get('new_source_only') is True and receipt.get('paired_source_questions') == 128
                    and receipt.get('new_source_fit_identity_sha256') == complete['identity_sha256']
                    and mentor.get('new_source_only') is True and mentor.get('new_questions') == 512
                    and mentor.get('legacy_questions') == 0 and mentor.get('paired_source_questions') == 128
                    and mentor.get('new_source_fit_identity_sha256') == complete['identity_sha256']
                    and model.mentor.classifier.temperature == temperature
                    and Path(model.mentor.classifier.directory).resolve() == (args.verifier / 'full').resolve(),
                    'Raw saved planner does not bind the registered source fit or observer')
        return {'exact_parent_neural_rows': len(metadata), 'inherited_temperature': temperature,
                'all_registered_source_pairs_retained': True, 'old_observer_supervision_disjoint': True,
                'new_neural_inference': False, 'metadata_and_array_subset_verified': True}

    def refined(self, directory, identity, complete, specs, files, registry_root, seen):
        require(identity.get('fit_questions') == 512 and identity.get('paired_review_questions') == 128
                and identity.get('emission_records') == 256 and identity.get('neural_calls') == 0
                and identity.get('cloud_calls') == 0 and identity.get('development_or_test_used') is False
                and identity.get('raw_neural_inputs_question_disjoint') is True
                and identity.get('inherited_temperature_never_used_new_question_labels') is True
                and identity.get('runtime_calibration_changed_from_parent') is False
                and identity.get('transitions_costs_transfer_changed_from_parent') is False,
                'Changed new-source question-excluded refinement contract')
        candidates = [p.parent.resolve() for p in Path(registry_root).rglob('complete.json')
                      if p.parent.resolve() != directory and legacy.sha(p) == complete['parent_complete_sha256']
                      and (p.parent / 'identity.json').exists()
                      and legacy.sha(p.parent / 'identity.json') == identity['new_source_parent_identity_sha256']]
        require(len(candidates) == 1, 'Cannot uniquely resolve the exact raw new-source parent')
        parent = candidates[0]
        raw_complete, inherited = self.chain(parent, registry_root, seen)
        files.update(inherited)
        require(str(parent) in self.receipts and self.receipts[str(parent)]['kind'] == 'new-source-raw'
                and complete['base_A'] == raw_complete['base_A'], 'Refinement requires the verified raw source-matched fit')
        for name, field in (('metadata.jsonl', 'neural_metadata_sha256'), ('neural.npz', 'neural_outputs_sha256')):
            digest_file(parent / name, identity[field], files)
        folds = verify_folds(identity, self.verified_metadata[parent], directory / 'calibration-oof.npz', files)
        original = {r['key']: r for r in raw_complete['models']}
        preservation = {}
        for key, spec in specs.items():
            require(spec['parent_sha256'] == original[key]['sha256'], 'Refined model points to the wrong raw model')
            raw_model = load_model(parent / (key + '.pkl'))
            model = load_model(directory / (key + '.pkl'))
            historical = spec.get('preserved_non_emission_sha256', '')
            require(len(historical) == 64 and all(c in '0123456789abcdef' for c in historical),
                    'Missing original in-memory preservation receipt')
            before, before_counts = preserved_state(raw_model)
            after, after_counts = preserved_state(model)
            require(before == after and before_counts == after_counts,
                    'Saved refinement changed a non-emission model component')
            preservation[key] = {'canonical_state_sha256': before, **before_counts,
                'recorded_in_memory_pickle_digest': historical, 'historical_byte_identity_claim': False}
            receipt = model.fit_receipt.get('emission_refinement', {})
            require(receipt.get('parent_model_sha256') == spec['parent_sha256']
                    and receipt.get('identity_sha256') == complete['identity_sha256']
                    and receipt.get('question_excluded_calibration_emissions') is True
                    and model.mentor.fit_receipt.get('source_refinement_identity_sha256') == complete['identity_sha256']
                    and model.mentor.fit_receipt.get('question_excluded_calibration_emissions') is True
                    and model.mentor.fit_receipt.get('calibration_folds') == 3,
                    'Saved refined planner lost its question-exclusion provenance')
        return {'parent_complete_sha256': complete['parent_complete_sha256'],
                'non_emission_components_preserved': True, 'calibration_folds_verified': folds,
                'preservation': preservation,
                'preservation_method': 'Exact class/structure/dtype/shape/value digests of the two SHA-verified saved models; excludes runtime caches and permitted emission/provenance changes.',
                'historical_byte_identity_claim': False}

    @contextmanager
    def installed(self):
        original_select = legacy.select_inputs
        def select(args):
            result = original_select(args)
            require(self.receipts, 'This extension requires an actual registered new-source mentor ancestor')
            registry = result[-1]
            registry['new_source_mentor_provenance'] = {
                'schema': SCHEMA, 'validator_sha256': legacy.sha(__file__),
                'fitting_chains': self.receipts,
                'no_development_or_test_question_values_read_by_extension': True,
                'no_neural_or_cloud_inference': True,
                'selection_and_target_checks': 'Unchanged legacy select_inputs checks; no threshold or comparator changes.'}
            registry['evidence_files'].append(str(Path(__file__).resolve()))
            return result
        with patch.object(legacy, 'fitting_chain', self.chain), patch.object(legacy, 'select_inputs', select):
            yield


def freeze(args):
    # No second manifest is patched after signing: evidence enters the original
    # selector result and is archived before its final digest is computed.
    with Provenance(args).installed():
        legacy.freeze(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-freeze-sha256')
    for name in ('data', 'a-summary', 'b-summary', 'c-summary', 'a-models', 'c-models', 'mentor-bank', 'verifier', 'legacy-examples'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--a-module', choices=tuple(legacy.A_MODULES))
    args = parser.parse_args()
    if args.phase == 'verify':
        frozen = legacy.verify(args.output, args.expected_freeze_sha256)
        receipt = frozen['completed_A_B_registry'].get('new_source_mentor_provenance', {})
        require(receipt.get('schema') == SCHEMA and receipt.get('validator_sha256') == legacy.sha(__file__),
                'Missing source-matched provenance extension')
        print('New-source prospective provenance verified; no inference or grading performed.')
    else:
        required = ('data', 'a_summary', 'b_summary', 'c_summary', 'a_models', 'c_models',
                    'mentor_bank', 'verifier', 'legacy_examples', 'a_module')
        if any(getattr(args, name) is None for name in required):
            parser.error('All data, model, summary, mentor, observer and legacy-input paths are required')
        freeze(args)


if __name__ == '__main__':
    main()
