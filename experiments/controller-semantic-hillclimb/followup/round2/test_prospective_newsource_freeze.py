"""Synthetic provenance fixtures only; never invoke a real prospective freeze."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import prospective_newsource_freeze as extension
import crossfit_mentor_emissions as crossfit
import mentor_pomdp
from difficulty_pomdp import PLAIN


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return extension.legacy.sha(path)


def fixture(root):
    root = root.resolve()
    data, bank, verifier = root / 'data', root / 'bank', root / 'verifier'
    parent, raw, refined, corpus = (root / name for name in ('parent', 'raw', 'refined', 'corpus'))
    for folder in (data / 'splits', bank, verifier / 'full', parent, raw, refined, corpus):
        folder.mkdir(parents=True, exist_ok=True)
    rows = [{'id': f'q{i}', 'question': 'Synthetic public text', 'domain': 'math' if i < 256 else 'science',
             'subject': 'math' if i < 256 else 'chemistry' if i < 384 else 'physics',
             'answer': 'PRIVATE_TRAIN_REFERENCE'} for i in range(512)]
    (data / 'splits/train-questions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    # Poison files would fail any accidental heldout parse; no reader needs them.
    (data / 'splits/dev-questions.jsonl').write_text('DO NOT READ DEVELOPMENT')
    (data / 'splits/test-questions.jsonl').write_text('DO NOT READ TEST')
    selected = [{'id': r['id'], 'subject': r['subject'], 'bank': 'new'}
                for r in rows[:64] + rows[256:288] + rows[384:416]]
    excluded = [{'id': f'old{i}', 'subject': 'math' if i < 32 else 'chemistry' if i < 48 else 'physics', 'bank': 'old'} for i in range(64)]
    contract = {'selection': selected + excluded}
    write(bank / 'contract.json', contract)
    (bank / 'train.jsonl').write_text('Synthetic opaque mentor bank\n')
    write(bank / 'complete.json', {'bank_sha256': extension.legacy.sha(bank / 'train.jsonl'),
                                  'contract_sha256': extension.legacy.sha(bank / 'contract.json')})
    legacy_examples = root / 'legacy.jsonl'
    legacy_examples.write_text(''.join(json.dumps({'id': r['id']}) + '\n' for r in excluded))
    (corpus / 'passages.jsonl').write_text('Synthetic corpus bytes\n')
    (corpus / 'tfidf.pkl').write_bytes(b'Synthetic retrieval bytes; do not unpickle')
    write(verifier / 'full/recipe.json', {'fold': 'full', 'input_profile': 'full-options-openstax-v1',
          'data_sha256': extension.legacy.sha(legacy_examples), 'corpus_sha256': extension.legacy.sha(corpus / 'passages.jsonl')})
    write(verifier / 'training-summary.json', {'temperature_selected_on_training_oof': 2.})
    base_path = root / 'base.pkl'; base_path.write_bytes(b'Synthetic original A artifact')
    chosen = {r['id'] for r in selected}
    metadata = []
    for r in rows:
        nodes = [('solver', p) for p in PLAIN]
        if r['id'] in chosen:
            nodes += [('solver', p) for p in ('guided', 'guided/reuse', 'guided/retry', 'retry/reuse')]
            nodes += [('original-review', '')]
        for kind, path in nodes:
            metadata.append({'id': r['id'], 'kind': kind, 'path': path, 'fold': 'full', 'label': 0,
                'train': kind == 'solver' and path in ('', 'repair', 'retry', 'guided'), 'valid': True,
                'prompt_sha256': extension.fingerprint([r['id'], kind, path]),
                'input_sha256': extension.fingerprint([r['id'], kind, path, 'input'])})
    old_meta = [{'id': r['id'], 'kind': 'solver', 'path': '', 'fold': 0} for r in excluded]
    parent_metadata = old_meta + metadata
    for folder, values in ((parent, parent_metadata), (raw, metadata)):
        (folder / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in values))
    outputs = {name: np.arange(len(parent_metadata) * width, dtype=np.float32).reshape(len(parent_metadata), width)
               for name, width in (('state', 4), ('plain', 8), ('guided', 8))}
    np.savez_compressed(parent / 'neural.npz', **outputs)
    np.savez_compressed(raw / 'neural.npz', **{name: value[64:] for name, value in outputs.items()})
    write(parent / 'neural-receipt.json', {'metadata_sha256': extension.legacy.sha(parent / 'metadata.jsonl'),
          'outputs_sha256': extension.legacy.sha(parent / 'neural.npz'),
          'encoder_recipes': {'full': extension.legacy.sha(verifier / 'full/recipe.json')}})
    parent_identity = {'mentor_bank_sha256': extension.legacy.sha(bank / 'train.jsonl'),
        'mentor_contract_sha256': extension.legacy.sha(bank / 'contract.json'),
        'legacy_examples_sha256': extension.legacy.sha(legacy_examples), 'base_model_sha256': extension.legacy.sha(base_path),
        'new_train_split_sha256': extension.legacy.sha(data / 'splits/train-questions.jsonl')}
    write(parent / 'identity.json', parent_identity)
    anchor = SimpleNamespace(mentor=SimpleNamespace(classifier=SimpleNamespace(temperature=2.)))
    (parent / 'anchor.pkl').write_bytes(pickle.dumps(anchor, protocol=5))
    original = {'identity_sha256': extension.legacy.sha(parent / 'identity.json'), 'development_or_test_used': False,
                'base_A': {'model': 'original-A'}, 'models': [{'key': 'anchor', 'sha256': extension.legacy.sha(parent / 'anchor.pkl')}]}
    write(parent / 'complete.json', original)
    sources = {str(p.resolve()): extension.legacy.sha(p) for p in (extension.FITTER, extension.CORE,
               extension.HERE / 'crossfit_mentor_emissions.py', extension.HERE / 'prepare_mixed_mentor.py')}
    identity = {'source_sha256': extension.legacy.sha(extension.FITTER), 'source_files': sources,
        'parent_models_directory': str(parent), 'parent_complete_sha256': extension.legacy.sha(parent / 'complete.json'),
        'parent_identity_sha256': extension.legacy.sha(parent / 'identity.json'),
        'parent_neural_receipt_sha256': extension.legacy.sha(parent / 'neural-receipt.json'),
        'parent_neural_metadata_sha256': extension.legacy.sha(parent / 'metadata.jsonl'),
        'parent_neural_sha256': extension.legacy.sha(parent / 'neural.npz'),
        **{k: parent_identity[k] for k in ('mentor_bank_sha256', 'mentor_contract_sha256', 'legacy_examples_sha256', 'base_model_sha256')},
        'new_train_question_sha256': parent_identity['new_train_split_sha256'], 'base_model_path': str(base_path),
        'encoder_recipe_sha256': extension.legacy.sha(verifier / 'full/recipe.json'),
        'encoder_training_summary_sha256': extension.legacy.sha(verifier / 'training-summary.json'),
        'inherited_temperature': 2., 'temperature_parent_model_sha256': original['models'][0]['sha256'],
        'corpus_sha256': extension.legacy.sha(corpus / 'passages.jsonl'), 'tfidf_sha256': extension.legacy.sha(corpus / 'tfidf.pkl'),
        'selected_pairs': selected, 'excluded_pairs': excluded, 'fit_questions': 512, 'paired_review_questions': 128,
        'reused_parent_indexes_sha256': extension.fingerprint(list(range(64, len(parent_metadata)))),
        'input_sha256': extension.fingerprint('synthetic formed inputs'), 'input_rows': len(metadata),
        'all_retained_neural_rows_from_old_full_unseen_model': True, 'neural_inference_calls': 0,
        'cloud_calls': 0, 'development_or_test_used': False, 'smoothing': [8., 32.], 'semantic_review': [False, True]}
    write(raw / 'identity.json', identity)
    write(raw / 'neural-receipt.json', {'metadata_sha256': extension.legacy.sha(raw / 'metadata.jsonl'),
        'outputs_sha256': extension.legacy.sha(raw / 'neural.npz'), 'input_sha256': identity['input_sha256'],
        'parent_neural_receipt_sha256': identity['parent_neural_receipt_sha256'], 'n': len(metadata)})
    models = []
    for key, smoothing, semantic in sorted(extension.MODEL_SETTINGS):
        receipt = {'new_source_only': True, 'paired_source_questions': 128,
                   'new_source_fit_identity_sha256': extension.legacy.sha(raw / 'identity.json')}
        classifier = SimpleNamespace(temperature=2., directory=str(verifier / 'full'))
        mentor = SimpleNamespace(classifier=classifier, regular=SimpleNamespace(state=np.arange(4)),
             review_observer=classifier if semantic else None,
             dynamics={'math': SimpleNamespace(prior=np.ones(8), emission=np.ones((2, 8, 3)), mentor=np.ones((2, 8)), _alphas={})},
             fit_receipt={**receipt, 'new_questions': 512, 'legacy_questions': 0})
        model = SimpleNamespace(mentor=mentor, base=SimpleNamespace(x=np.arange(8)), transfer=np.eye(8),
             fit_receipt={**receipt, 'fit_module': 'new_source_mentor_core'})
        path = raw / (key + '.pkl'); path.write_bytes(pickle.dumps(model, protocol=5))
        models.append({'key': key, 'smoothing': smoothing, 'semantic_review': semantic, 'sha256': extension.legacy.sha(path)})
    shared = {'fit_questions': 512, 'paired_review_questions': 128, 'base_A': original['base_A'],
              'new_source_only': True, 'neural_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False}
    write(raw / 'complete.json', {**shared, 'models': models, 'identity_sha256': extension.legacy.sha(raw / 'identity.json')})
    solver = [r for r in metadata if r['kind'] == 'solver']
    folds = [int(hashlib.sha256(r['id'].encode()).hexdigest()[:8], 16) % 3 for r in solver]
    receipts = []
    for fold in range(3):
        fit = [r for r, f in zip(solver, folds) if f != fold and r['train'] and r['valid']]
        held = [r for r, f in zip(solver, folds) if f == fold]
        fit_ids, held_ids = {r['id'] for r in fit}, {r['id'] for r in held}
        receipts.append({'fold': fold, 'fit_questions': len(fit_ids), 'held_questions': len(held_ids),
            'fit_rows': len(fit), 'held_rows': len(held), 'question_overlap': 0,
            'fit_ids_sha256': extension.fingerprint(sorted(fit_ids)), 'held_ids_sha256': extension.fingerprint(sorted(held_ids))})
    np.savez_compressed(refined / 'calibration-oof.npz', probabilities=np.full((len(solver), 4), .25))
    refined_identity = {'source_sha256': identity['source_sha256'], 'source_files': sources,
        'new_source_parent_identity_sha256': extension.legacy.sha(raw / 'identity.json'),
        'neural_metadata_sha256': extension.legacy.sha(raw / 'metadata.jsonl'), 'neural_outputs_sha256': extension.legacy.sha(raw / 'neural.npz'),
        'fit_questions': 512, 'paired_review_questions': 128, 'emission_records': 256, 'folds': receipts,
        'raw_neural_inputs_question_disjoint': True, 'inherited_temperature_never_used_new_question_labels': True,
        'runtime_calibration_changed_from_parent': False, 'transitions_costs_transfer_changed_from_parent': False,
        'calibration_oof_sha256': extension.legacy.sha(refined / 'calibration-oof.npz'),
        'neural_calls': 0, 'cloud_calls': 0, 'development_or_test_used': False}
    write(refined / 'identity.json', refined_identity)
    refined_models = []
    for spec in models:
        model = pickle.loads((raw / (spec['key'] + '.pkl')).read_bytes())
        preserved = crossfit.preserved_digest(model)
        model.mentor.dynamics['math'].emission *= 2
        model.mentor.fit_receipt.update(source_refinement_identity_sha256=extension.legacy.sha(refined / 'identity.json'),
            question_excluded_calibration_emissions=True, calibration_folds=3)
        model.fit_receipt['emission_refinement'] = {'parent_model_sha256': spec['sha256'],
            'identity_sha256': extension.legacy.sha(refined / 'identity.json'), 'question_excluded_calibration_emissions': True}
        path = refined / (spec['key'] + '.pkl'); path.write_bytes(pickle.dumps(model, protocol=5))
        refined_models.append({**spec, 'sha256': extension.legacy.sha(path), 'parent_sha256': spec['sha256'],
                               'preserved_non_emission_sha256': preserved})
    write(refined / 'complete.json', {**shared, 'models': refined_models,
          'identity_sha256': extension.legacy.sha(refined / 'identity.json'),
          'parent_complete_sha256': extension.legacy.sha(raw / 'complete.json')})
    args = SimpleNamespace(data=data, mentor_bank=bank, verifier=verifier, legacy_examples=legacy_examples)
    return SimpleNamespace(args=args, root=root, parent=parent, raw=raw, refined=refined, corpus=corpus,
                           original=original, identity=identity, refined_identity=refined_identity, metadata=metadata, rows=rows, contract=contract)


class NewSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.f = fixture(Path(self.temporary.name))

    def verifier(self):
        result = extension.Provenance(self.f.args)
        result.original_chain = lambda directory, registry, seen=None: (self.f.original,
            [self.f.parent / 'identity.json', self.f.parent / 'complete.json'])
        return result

    def test_complete_raw_and_refined_chain_exact_cache_folds_and_preservation(self):
        verifier = self.verifier()
        original_loads = json.loads
        def guarded(text, *args, **kwargs):
            if 'PRIVATE_TRAIN_REFERENCE' in str(text) or 'DO NOT READ' in str(text):
                self.fail('A hidden reference or heldout question file was decoded')
            return original_loads(text, *args, **kwargs)
        with patch.object(mentor_pomdp.perception, 'CORPUS', self.f.corpus), \
             patch.object(extension, 'load_model', side_effect=lambda path: pickle.loads(path.read_bytes())), \
             patch.object(json, 'loads', side_effect=guarded), \
             patch.object(mentor_pomdp.perception.Observer, 'load', side_effect=AssertionError('No neural calls')):
            complete, files = verifier.chain(self.f.refined, self.f.root)
        self.assertEqual(len(complete['models']), 4)
        self.assertEqual(len(verifier.receipts), 2)
        self.assertEqual(verifier.receipts[str(self.f.raw)]['exact_parent_neural_rows'], 4224)
        self.assertTrue(verifier.receipts[str(self.f.refined)]['non_emission_components_preserved'])
        self.assertIn(self.f.raw / 'neural.npz', files)
        self.assertIn(self.f.refined / 'calibration-oof.npz', files)
        self.assertIn(Path(extension.__file__).resolve(), files)

    def test_source_and_model_tamper_are_rejected_before_model_load(self):
        identity = copy.deepcopy(self.f.identity)
        identity['source_files'][str(extension.CORE.resolve())] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'required source'):
            extension.source_evidence(identity, set())
        complete = extension.read(self.f.raw / 'complete.json')
        (self.f.raw / (complete['models'][0]['key'] + '.pkl')).write_bytes(b'changed')
        with patch.object(extension, 'load_model', side_effect=AssertionError('Must verify digest before unpickle')):
            with self.assertRaisesRegex(ValueError, 'file changed'):
                self.verifier().chain(self.f.raw, self.f.root)

    def test_recorded_environment_binary_is_hashed_not_dropped(self):
        root = self.f.root / 'import-inventory'; here = root / 'experiments'
        here.mkdir(parents=True)
        sources = {}
        for name in ('fit_new_source_mentor.py', 'new_source_mentor_core.py',
                     'crossfit_mentor_emissions.py', 'prepare_mixed_mentor.py'):
            path = here / name; path.write_text('# Synthetic frozen source\n')
            sources[str(path)] = extension.legacy.sha(path)
        binary = root / '.venv/lib/python3.12/site-packages/pyarrow/lib.synthetic.so'
        binary.parent.mkdir(parents=True); binary.write_bytes(b'Synthetic never-loaded binary')
        sources[str(binary)] = extension.legacy.sha(binary)
        identity = {'source_sha256': sources[str(here / 'fit_new_source_mentor.py')], 'source_files': sources}
        with patch.object(extension, 'FITTER', here / 'fit_new_source_mentor.py'), \
             patch.object(extension, 'CORE', here / 'new_source_mentor_core.py'), \
             patch.object(extension, 'HERE', here), patch.object(extension.legacy, 'ROOT', root):
            files = set(); extension.source_evidence(identity, files)
            self.assertIn(binary, files)
            binary.write_bytes(b'changed binary')
            with self.assertRaisesRegex(ValueError, 'file changed'):
                extension.source_evidence(identity, set())

    def test_whole_source_membership_no_correctness_subset_and_nn_exclusion(self):
        legacy_ids = {r['id'] for r in self.f.identity['excluded_pairs']}
        ids, selected = extension.check_selection(self.f.identity, self.f.contract, self.f.rows, legacy_ids)
        self.assertEqual((len(ids), len(selected)), (512, 128))
        changed = copy.deepcopy(self.f.identity); changed['selected_pairs'].pop()
        with self.assertRaisesRegex(ValueError, 'every registered new pair'):
            extension.check_selection(changed, self.f.contract, self.f.rows, legacy_ids)
        with self.assertRaisesRegex(ValueError, 'observer membership'):
            extension.check_selection(self.f.identity, self.f.contract, self.f.rows, legacy_ids | {self.f.rows[0]['id']})

    def test_exact_array_subset_rejects_changed_value_even_with_updated_local_hash(self):
        with np.load(self.f.raw / 'neural.npz') as archive:
            outputs = {k: archive[k].copy() for k in archive.files}
        outputs['state'][1, 1] += 1
        np.savez_compressed(self.f.raw / 'neural.npz', **outputs)
        receipt = extension.read(self.f.raw / 'neural-receipt.json')
        receipt['outputs_sha256'] = extension.legacy.sha(self.f.raw / 'neural.npz')
        write(self.f.raw / 'neural-receipt.json', receipt)
        with self.assertRaisesRegex(ValueError, 'arrays differ'):
            extension.verify_neural_subset(self.f.raw, self.f.parent, self.f.identity,
                {r['id'] for r in self.f.rows}, {r['id'] for r in self.f.identity['selected_pairs']}, set())

    def test_fold_receipts_and_probabilities_are_not_merely_trusted(self):
        extension.verify_folds(self.f.refined_identity, self.f.metadata, self.f.refined / 'calibration-oof.npz', set())
        changed = copy.deepcopy(self.f.refined_identity); changed['folds'][0]['held_ids_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'fold receipts changed'):
            extension.verify_folds(changed, self.f.metadata, self.f.refined / 'calibration-oof.npz', set())
        count = sum(r['kind'] == 'solver' for r in self.f.metadata)
        np.savez_compressed(self.f.refined / 'calibration-oof.npz', probabilities=np.full((count, 4), np.nan))
        changed = copy.deepcopy(self.f.refined_identity)
        changed['calibration_oof_sha256'] = extension.legacy.sha(self.f.refined / 'calibration-oof.npz')
        with self.assertRaisesRegex(ValueError, 'Invalid question-excluded'):
            extension.verify_folds(changed, self.f.metadata, self.f.refined / 'calibration-oof.npz', set())

    def test_non_emission_change_and_parent_cycle_are_rejected(self):
        complete = extension.read(self.f.refined / 'complete.json')
        spec = complete['models'][0]
        path = self.f.refined / (spec['key'] + '.pkl')
        model = pickle.loads(path.read_bytes()); model.transfer[0, 0] = 99
        path.write_bytes(pickle.dumps(model, protocol=5)); spec['sha256'] = extension.legacy.sha(path)
        write(self.f.refined / 'complete.json', complete)
        with patch.object(mentor_pomdp.perception, 'CORPUS', self.f.corpus), \
             patch.object(extension, 'load_model', side_effect=lambda path: pickle.loads(path.read_bytes())):
            with self.assertRaisesRegex(ValueError, 'non-emission'):
                self.verifier().chain(self.f.refined, self.f.root)
        with self.assertRaisesRegex(ValueError, 'cycle'):
            self.verifier().chain(self.f.raw, self.f.root, {self.f.raw.resolve()})

    def test_preservation_is_stable_across_numpy_view_pickle_roundtrip(self):
        x = np.arange(16, dtype=np.float32)
        model = SimpleNamespace(base=SimpleNamespace(x=x[::2]), transfer=np.broadcast_to(x[:8], (3, 8)),
            mentor=SimpleNamespace(regular=SimpleNamespace(state=x), classifier=SimpleNamespace(v=x[1::2]),
                review_observer=None, dynamics={'math': SimpleNamespace(prior=x[::-1], emission=x, mentor=x, _alphas={})}))
        restored = pickle.loads(pickle.dumps(model, protocol=5))
        # This is the actual reason the historical pre-save receipt cannot be
        # reproduced by serializing a later loaded representation.
        self.assertNotEqual(crossfit.preserved_digest(model), crossfit.preserved_digest(restored))
        self.assertEqual(extension.preserved_state(model), extension.preserved_state(restored))
        restored.base.x[0] += 1
        self.assertNotEqual(extension.preserved_state(model), extension.preserved_state(restored))

    def test_scoped_adapter_preserves_selection_failures_and_restores_legacy(self):
        original_chain, original_select = extension.legacy.fitting_chain, extension.legacy.select_inputs
        verifier = self.verifier()
        with patch.object(extension.legacy, 'select_inputs', side_effect=ValueError('Development targets are not met')):
            with verifier.installed():
                with self.assertRaisesRegex(ValueError, 'targets are not met'):
                    extension.legacy.select_inputs(self.f.args)
        self.assertIs(extension.legacy.fitting_chain, original_chain)
        self.assertIs(extension.legacy.select_inputs, original_select)
        registry = {'evidence_files': []}
        verifier.receipts[str(self.f.raw)] = {'kind': 'new-source-raw'}
        expected = ({'A': 'strongestA', 'B': 'strongestB', 'C': 'eligibleC'}, {'Stop': 'baseline'}, {'all_required_passed': True}, {}, {}, registry)
        with patch.object(extension.legacy, 'select_inputs', return_value=expected):
            with verifier.installed():
                result = extension.legacy.select_inputs(self.f.args)
        self.assertIs(result, expected)
        self.assertEqual(registry['new_source_mentor_provenance']['schema'], extension.SCHEMA)
        self.assertIn(str(Path(extension.__file__).resolve()), registry['evidence_files'])
        self.assertIs(extension.legacy.fitting_chain, original_chain)

    def test_legacy_stronger_base_recursion_can_reach_new_refined_ancestor(self):
        wrapper = self.f.root / 'stronger-base'
        write(wrapper / 'identity.json', {'stronger_base_parent_complete_sha256': 'synthetic'})
        verifier = self.verifier()
        delegated = []
        def legacy_chain(directory, registry, seen=None):
            delegated.append(Path(directory))
            if Path(directory) == wrapper:
                return extension.legacy.fitting_chain(self.f.refined, registry, seen)
            return self.f.original, [self.f.parent / 'complete.json']
        verifier.original_chain = legacy_chain
        with patch.object(mentor_pomdp.perception, 'CORPUS', self.f.corpus), \
             patch.object(extension, 'load_model', side_effect=lambda path: pickle.loads(path.read_bytes())):
            with verifier.installed():
                complete, files = extension.legacy.fitting_chain(wrapper, self.f.root)
        self.assertEqual(len(complete['models']), 4)
        self.assertIn(wrapper, delegated)
        self.assertIn(self.f.parent, delegated)
        self.assertIn(self.f.refined / 'calibration-oof.npz', files)
        self.assertEqual(len(verifier.receipts), 2)


if __name__ == '__main__':
    unittest.main()
