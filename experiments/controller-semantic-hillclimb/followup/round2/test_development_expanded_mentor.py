"""Synthetic role, complete-grid, retention and provenance checks; no DEV IO."""
import copy
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import development_expanded_mentor as run
import expanded_development_models as builder
from test_stronger_base_hybrid import FakeObserver, FakeBase, parent
from test_metadata_review import make_adapter


def statistic(items):
    return {'accuracy': float(np.mean([r['correct'] for r in items])),
        'mean_compute': float(np.mean([r['compute'] for r in items])),
        'mean_judge_calls': float(np.mean([r['judge_calls'] for r in items]))}


def item(correct=True, metadata=False, review_nn=False):
    return {'correct': int(correct), 'compute': 100., 'judge_calls': 1., 'judge_physical_calls': 2,
        'metadata_calls': int(metadata), 'metadata_cpu_seconds': .002 if metadata else 0.,
        'verifier_components': [{'kind': 'solver', 'slots': 10, 'parameters_billions': .435}]
            + ([{'kind': 'review', 'slots': 20, 'parameters_billions': .435}] if review_nn else [])}


class ExpandedDevelopmentTests(unittest.TestCase):
    def test_exact_cache_augmentation_keeps_other_assets_and_rejects_collisions(self):
        key=('recipe','dynamic_grounded_pomdp','Observer'); other=('other','science_option_observer','ScienceOptionObserver')
        entries={('solver',f'q{i}','', 'a'*64,'paid'):(np.zeros(4),np.zeros(8),np.zeros(8),32) for i in range(768)}
        caches={key:dict(list(entries.items())[:12]),other:{'other-input':'untouched'}}
        receipt={'converted_nodes':768,'overlap_with_shared_cache':12,'bitwise_identical_overlap':12,'max_absolute_logit_difference':0.}
        run.merge_plain_features(caches,key,entries,receipt)
        self.assertEqual(len(caches[key]),768);self.assertEqual(caches[other],{'other-input':'untouched'})
        changed=copy.deepcopy(entries);first=next(iter(changed));changed[first][0][0]=1.
        with self.assertRaisesRegex(ValueError,'changed an imported'):
            run.merge_plain_features(caches,key,changed,receipt)
        with self.assertRaisesRegex(ValueError,'bitwise-identical'):
            run.merge_plain_features(caches,key,entries,{**receipt,'bitwise_identical_overlap':11})
        with self.assertRaisesRegex(ValueError,'matching observer'):
            run.merge_plain_features({},key,entries,receipt)

    def test_source_ancestor_context_binds_same_plain_completion(self):
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)
            run.io.write(directory/'identity.json',{'source_sha256':run.io.sha(run.previous.__file__),
                'ordinary_plain_completion_sha256':'original'})
            with patch.object(run.inherited,'completed_bank',return_value=(b'',{})):
                with self.assertRaisesRegex(ValueError,'ancestor context'):
                    run.sealed_source(directory,'panel','changed')

    def test_static_source_closure_includes_indirectly_loaded_A_classes(self):
        sources = run.io.sources(run)
        names = {Path(path).name for path in sources}
        self.assertTrue({'factorized_repairability.py', 'nonlinear_repairability.py',
            'subject_science_pomdp.py', 'science_fusion_pomdp.py', 'science_option_observer.py',
            'metadata_review_hybrid.py', 'stronger_base_hybrid.py'} <= names)
        self.assertFalse(any(name.startswith('test_') for name in names))

    def test_both_builders_keep_trained_mentor_transfer_and_metadata_exact(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            old_observer = FakeObserver(directory/'old', 435_000_000)
            selected_observer = FakeObserver(directory/'selected', 600_000_000)
            selected = {'module': 'continuous_repairability', 'candidate': {'settings': {'penalty': .08}}}
            anchor = FakeBase(selected_observer)
            raw_parent = parent(old_observer); raw_parent.base = FakeBase(old_observer)
            for dynamics in raw_parent.mentor.dynamics.values():
                dynamics.review_costs = np.ones(8)*100.
            before = builder.retained_state(raw_parent)
            raw, module, receipt = builder.build(anchor, selected, raw_parent,
                {'metadata_review': False, 'policy_module': 'hybrid_repairability'}, {})
            self.assertEqual(module, 'stronger_base_hybrid')
            self.assertEqual(builder.retained_state(raw), before)
            self.assertEqual(receipt, before); self.assertIs(raw.base, anchor)
            meta_parent = run.metadata_runtime.build(copy.deepcopy(raw_parent), make_adapter(), {'penalty': .08}, {})
            before = builder.retained_state(meta_parent)
            model, module, _ = builder.build(anchor, selected, meta_parent,
                {'metadata_review': True, 'policy_module': 'metadata_review_hybrid'}, {})
            self.assertEqual(module, 'metadata_review_hybrid')
            self.assertEqual(builder.retained_state(model), before)
            self.assertIsNone(model.review_observer)

    def test_explicit_metadata_dispatch_cannot_fall_back_to_semantic(self):
        with TemporaryDirectory() as temporary:
            observer = FakeObserver(Path(temporary)/'model', 435_000_000)
            original = parent(observer); original.base = FakeBase(observer)
            model = run.metadata_runtime.build(original, make_adapter(), {'penalty': .08}, {})
            with patch.object(run.metadata_runtime, 'evaluate', return_value='metadata') as call:
                self.assertEqual(run.dispatch(model, 'metadata_review_hybrid', {}, {}, {}, {}), 'metadata')
                self.assertEqual(call.call_args.args[-1], 'C')
            with self.assertRaisesRegex(ValueError, 'Never send'):
                run.dispatch(model, 'stronger_base_hybrid', {}, {}, {}, {})

    def test_policy_lock_released_before_missing_paid_acquisition(self):
        lock = threading.Lock(); rows = [{'id': 'q'}]; trees = {'q': {}}
        class Acquire:
            def acquire(self, row, missing):
                if lock.locked(): raise AssertionError('Policy lock held during cloud acquisition')
                trees['q']['guided'] = {}; self.called = True
        acquire = Acquire()
        def dispatch(*args):
            self.assertTrue(lock.locked())
            if 'guided' not in trees['q']: raise run.study.MissingObservation('solver', 'guided')
            return item()
        with patch.object(run, 'dispatch', side_effect=dispatch):
            result = run.evaluate(object(), 'stronger_base_hybrid', {}, rows, trees, {}, acquire, lock)
        self.assertEqual(len(result), 1); self.assertTrue(acquire.called)

    def test_complete156_candidate_grid_evaluates_all256_each_time(self):
        specs = [{'key': f'model{i}', 'smoothing': 8. if i < 3 else 32.,
            'semantic_review': i % 3 == 1, 'metadata_review': i % 3 == 2,
            'policy_module': 'metadata_review_hybrid' if i % 3 == 2 else 'stronger_base_hybrid'} for i in range(6)]
        calls, saved = [], []
        subjects = ['math']*128 + ['chemistry']*64 + ['physics']*64
        def callback(model, spec, settings):
            calls.append((spec['key'], settings))
            return [item(i % 2, spec['metadata_review'], spec['semantic_review']) for i in range(256)], subjects
        with patch.object(run.study, 'statistics', side_effect=statistic):
            candidates = run.search_models(specs, lambda spec: object(), callback,
                lambda spec, model, candidates: saved.append((spec['key'], len(candidates))))
        self.assertEqual(len(calls), 156); self.assertEqual(len(candidates), 156)
        self.assertEqual([n for _, n in saved], [26, 52, 78, 104, 130, 156])
        self.assertEqual(sum('by_subject' in settings for _, settings in calls), 30)
        self.assertTrue(all(r['dev']['mean_judge_physical_calls'] == 2 for r in candidates))
        self.assertTrue(all(r['dev']['mean_review_neural_calls'] == int(r['semantic_review']) for r in candidates))
        with self.assertRaisesRegex(ValueError, 'entire256'):
            run.search_models(specs, lambda spec: object(),
                lambda *args: ([item()]*255, subjects[:-1]), lambda *args: None)

    def test_strongest_B_and_prior_C_retained_without_claiming_old_base_is_current(self):
        a = {'model_sha256': 'current-A', 'candidate': {'settings': {}, 'dev': {'accuracy': .6, 'mean_compute': 80.}}}
        ordinary = {'dev': {'accuracy': .65, 'mean_compute': 400.}}
        matched = {'dev': {'accuracy': .7, 'mean_compute': 500.}}
        b = {'B': matched, 'ordinary_B': ordinary, 'matched_B': matched}
        own = {'model': 'new', 'dev': {'accuracy': .66, 'mean_compute': 100., 'mean_judge_calls': 1},
               'retained_from_previous': False}
        prior = {'model': 'previous175', 'dev': {'accuracy': 175/256, 'mean_compute': 150., 'mean_judge_calls': 1},
                 'retained_from_previous': True, 'inherited_base_model_sha256': None, 'inherited_base_settings': None}
        record = {'base_selection': a, 'prior_C': [prior]}
        result = run.summary([own], record, a, {'Repair': {'accuracy': .5}}, b)
        self.assertEqual(result['selected_C']['model'], 'previous175')
        self.assertEqual(result['selected_expanded_C']['model'], 'new')
        self.assertTrue(result['retained_previous_C'])
        self.assertEqual(result['B'], matched)
        self.assertFalse(result['inherited_base_is_strongest_completed_A'])
        self.assertFalse(result['development_gaps_met'])
        self.assertFalse(result['test_used']); self.assertFalse(result['final_gap_claim'])

    def test_prior_C_requires_same_baselines_model_and_review_caps(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); directory = root/'prior'; models = root/'models'
            directory.mkdir(); models.mkdir()
            (models/'chosen.pkl').write_bytes(b'synthetic-unloaded-model')
            run.io.write(models/'complete.json', {'models': [{'key': 'chosen', 'sha256': run.io.sha(models/'chosen.pkl')}]})
            identity = {'source_sha256': run.io.sha(run.study.__file__),
                'models_sha256': run.io.sha(models/'complete.json'),
                'review': {'thinking_cap': 1024, 'answer_cap': 512, 'guided_cap': 128}}
            baselines = {'Repair': {'accuracy': .5}}
            run.io.write(directory/'identity.json', identity)
            run.io.write(directory/'summary.json', {'baselines': baselines, 'selected_C': {'model': 'chosen', 'dev': {'accuracy': .6}}})
            with patch.object(run.inherited, 'completed_bank', return_value=(b'', {'directory': str(directory)})):
                candidate, _ = run.prior_reference(directory/'summary.json', models, 'same-panel', baselines)
                self.assertTrue(candidate['retained_from_previous'])
                with self.assertRaisesRegex(ValueError, 'panel controls'):
                    run.prior_reference(directory/'summary.json', models, 'same-panel', {})
                identity['review']['guided_cap'] = 256; run.io.write(directory/'identity.json', identity)
                with self.assertRaisesRegex(ValueError, 'protocol'):
                    run.prior_reference(directory/'summary.json', models, 'same-panel', baselines)

    def test_completion_seal_binds_search_raw_cache_and_transport_errors(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary); (output/'models').mkdir()
            names = ['registration.json', 'identity.json', 'cache-import.json', 'summary.json', 'search.json',
                'dev-observations.jsonl', 'raw-generations.jsonl', 'private-neural-cache.pkl',
                'private-neural-caches.pkl', 'models/identity.json', 'models/complete.json', 'transport-errors.jsonl']
            for name in names: (output/name).write_bytes(b'synthetic')
            first = run.seal_payload(output)
            self.assertEqual(set(first['files']), set(names))
            (output/'private-neural-caches.pkl').write_bytes(b'changed')
            self.assertNotEqual(run.seal_payload(output), first)

    def test_full_synthetic_replay_seals_all156_and_resume_does_no_work(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary); output, source, data = [root/n for n in ('output', 'source', 'data')]
            for path in (output, source/'models', data): path.mkdir(parents=True)
            run.io.write(output/'registration.json', {'synthetic': True})
            run.io.write(data/'dev-plain-complete.json', {'synthetic': True})
            (source/'raw-generations.jsonl').write_bytes(b'')
            observer = FakeObserver(root/'observer', 435_000_000)
            source_model = SimpleNamespace(mentor=SimpleNamespace(classifier=observer))
            raw_cache = {('solver', 'q0', '', 'a'*64, 'paid'): (np.zeros(4), np.zeros(8), np.zeros(8), 12)}
            (source/'private-neural-cache.pkl').write_bytes(pickle.dumps(raw_cache))
            (source/'private-neural-caches.pkl').write_bytes(pickle.dumps({run.previous.share_key(observer): raw_cache}))
            (source/'models/original.pkl').write_bytes(pickle.dumps(source_model))
            source_manifest = {'models': [{'key': 'original'}]}
            chosen = {'model_sha256': 'same-A', 'candidate': {'settings': {}, 'dev': {'accuracy': .65, 'mean_compute': 10.}}}
            record = {'dev_questions_sha256': 'same-panel', 'source_receipt': {}, 'base_selection': chosen,
                'A_references': [], 'pending_optional_A': [], 'prior_C': [],'plain_feature_augmentation':None}
            args = SimpleNamespace(output=output, source=source, models=root/'fit', data=data, registry=root,
                b_reference=root/'B.json', registration_sha256=run.io.sha(output/'registration.json'), device='cpu', collect=False,plain_features=None)
            b = {'dev': {'accuracy': .7, 'mean_compute': 200.}}
            controls = {name: {'accuracy': .5, 'mean_compute': 10., 'mean_judge_calls': 0.} for name in run.study.BASELINES}
            run.io.write(args.b_reference, {'B': b, 'ordinary_B': b, 'matched_B': b, 'baselines': controls})
            run.io.write(root/'identity.json', {})
            rows = [{'id': f'q{i}', 'subject': 'math' if i < 128 else 'chemistry' if i < 192 else 'physics'} for i in range(256)]
            trees = {r['id']: {'': {}} for r in rows}
            def construct(*unused):
                folder = output/'models'; folder.mkdir(exist_ok=True)
                specs = []
                for i in range(6):
                    path = folder/f'model{i}.pkl'; path.write_bytes(pickle.dumps(source_model))
                    specs.append({'key': path.stem, 'sha256': run.io.sha(path), 'smoothing': 8. if i < 3 else 32.,
                        'metadata_review': i%3 == 2, 'semantic_review': i%3 == 1,
                        'policy_module': 'metadata_review_hybrid' if i%3 == 2 else 'stronger_base_hybrid'})
                result = {'models': specs, 'observer_assets': []}
                run.io.write(folder/'identity.json', {'synthetic': True}); run.io.write(folder/'complete.json', result)
                return result
            calls = []
            def evaluate(model, module, settings, actual_rows, *args):
                calls.append((module, settings)); self.assertEqual(len(actual_rows), 256)
                return [item(True, module == 'metadata_review_hybrid') for row in actual_rows]
            patches = [patch.object(run, 'validate_registration', return_value=record),
                patch.object(run, 'sealed_source', return_value=(b'', {}, source_manifest)),
                patch.object(run.builder, 'construct', side_effect=construct),
                patch.object(run.previous, 'attach'), patch.object(run.previous, 'references', return_value=(chosen, [], [])),
                patch.object(run.inherited, 'validate_b'), patch.object(run.study, 'development_view', return_value=(rows, trees, {})),
                patch.object(run.common, 'validate_nodes'), patch.object(run.common, 'coverage', return_value={}),
                patch.object(run.study, 'statistics', side_effect=statistic),
                patch.object(run.study, 'fixed', return_value={'correct': .5, 'compute': 10., 'judge_calls': 0.}),
                patch.object(run, 'evaluate', side_effect=evaluate)]
            for p in patches: stack.enter_context(p)
            run.run(args)
            self.assertEqual(len(calls), 156)
            result = run.io.read(output/'summary.json')
            self.assertTrue(result['development_gaps_met']); self.assertFalse(result['test_used'])
            self.assertEqual(result['new_physical_generations'], 0)
            self.assertEqual(len(run.io.read(output/'search.json')), 156)
            marker = (output/'expanded-replay-complete.json').read_bytes()
            run.run(args)
            self.assertEqual(len(calls), 156); self.assertEqual((output/'expanded-replay-complete.json').read_bytes(), marker)
            (output/'raw-generations.jsonl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'completed expanded replay'): run.run(args)


if __name__ == '__main__': unittest.main()
