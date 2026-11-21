import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import development_stronger_metadata as driver
import stronger_metadata_review as builder
import metadata_review_hybrid as runtime
from test_metadata_review import make_adapter, review
from test_stronger_base_hybrid import FakeObserver, FakeBase, FakeFusion, parent


class FakeADynamics:
    def values(self, belief, remaining, penalty):
        return np.array([1., 0., 0.])


def plain(observer):
    result = FakeBase(observer)
    result.dynamics = {s: FakeADynamics() for s in ('math', 'chemistry', 'physics')}
    return result


def metadata_parent(observer, query=True):
    original = parent(observer, query=query)
    original.base = plain(observer)
    for dynamics in original.mentor.dynamics.values():
        dynamics.review_costs = np.ones(8) * 100.
    return runtime.build(original, make_adapter(), {'penalty': .08}, {})


def raw_cache():
    return {('solver', 'q', '', 'a' * 64, 'paid'): (np.zeros(4), np.zeros(8), np.zeros(8), 12)}


class StrongerMetadataTest(unittest.TestCase):
    def test_builder_preserves_all_metadata_state_across_saved_reload(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            old = FakeObserver(folder / 'old', 400_000_000)
            new = FakeObserver(folder / 'new', 600_000_000)
            original = metadata_parent(old)
            before = builder.retained_state(original)
            base = plain(new)
            candidate = builder.build(base, 'continuous_repairability',
                {'penalty': .04, 'posterior_weight': .5}, original, {'test': 'synthetic'})
            self.assertIs(candidate.mentor, original.mentor)
            self.assertIs(candidate.metadata_adapter, original.metadata_adapter)
            self.assertIs(candidate.transfer, original.transfer)
            self.assertEqual(candidate.fit_receipt['builder_module'], 'stronger_metadata_review')
            self.assertEqual(candidate.fit_receipt['policy_module'], 'metadata_review_hybrid')
            restored = pickle.loads(pickle.dumps(candidate))
            self.assertEqual(builder.retained_state(restored), before)
            self.assertEqual(builder.retained_state(original), before)
            restored.mentor.dynamics['physics'].review_costs[0] += 1.
            self.assertNotEqual(builder.retained_state(restored), before)

    def test_subject_fusion_bills_each_actual_asset_without_review_neural_pass(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            old = FakeObserver(folder / 'old', 400_000_000)
            option = FakeObserver(folder / 'option', 600_000_000)
            base = SimpleNamespace(models={s: plain(FakeFusion([old, option])) for s in ('math', 'chemistry', 'physics')},
                                   settings={s: {'penalty': .08, 'posterior_weight': .5} for s in ('math', 'chemistry', 'physics')})
            model = builder.build(base, 'science_fusion_pomdp', {}, metadata_parent(old), {})
            row = {'id': 'q', 'question': 'Public?', 'answer': 'B', 'subject': 'physics', 'domain': 'science'}
            node = lambda answer: {'answer': answer, 'text': 'Final: ' + answer, 'input_tokens': 10, 'output_tokens': 5}
            tree = {'': node('A'), 'guided': node('B'), 'guided/reuse': node('B')}
            result = runtime.evaluate(row, tree, {('q', ''): review()}, model, {'penalty': .08}, 'C')
            self.assertEqual(result['verifier_calls'], 3)
            self.assertEqual(result['review_neural_calls'], 0)
            self.assertEqual(result['judge_physical_calls'], 2)
            self.assertEqual(result['metadata_calls'], 1)
            self.assertAlmostEqual(result['compute'], (4 * 45 + 3 * 16 + .4 * 13 * 2 + .6 * 13) / 3.09)
            self.assertTrue(all(v['kind'] == 'solver' for v in result['verifier_components']))

    def test_no_query_uses_selected_continuous_A_settings_and_no_metadata(self):
        with TemporaryDirectory() as temporary:
            observer = FakeObserver(Path(temporary) / 'observer', 400_000_000)
            model = builder.build(plain(observer), 'continuous_repairability',
                {'by_subject': {'physics': {'penalty': .04, 'posterior_weight': .5}}},
                metadata_parent(observer, query=False), {})
            row = {'id': 'q', 'question': 'Public?', 'answer': 'A', 'subject': 'physics', 'domain': 'science'}
            tree = {'': {'answer': 'A', 'text': 'Paid', 'input_tokens': 10, 'output_tokens': 5}}
            def continuous(safe, paid, judges, base, settings, family):
                self.assertEqual(safe['answer'], '')
                self.assertEqual(settings, {'penalty': .12, 'posterior_weight': .5})
                return runtime.mentor_core.outcome(safe, paid, {}, '', [''], [],
                    [{'path': '', 'action': 'stop', 'queried': False}])
            with patch.object(runtime.continuous, 'evaluate', side_effect=continuous):
                with patch.object(model.metadata_adapter, 'predict_paid', side_effect=AssertionError('Unpaid review')):
                    value = runtime.evaluate(row, tree, {}, model, {'penalty': .12}, 'C')
            self.assertEqual(value['metadata_calls'], 0)
            self.assertEqual(value['correct'], 1)

    def test_class_aware_cache_never_relabels_option_outputs_as_raw_logits(self):
        raw = raw_cache(); recipe = 'b' * 64
        old_key = (recipe, 'dynamic_grounded_pomdp', 'Observer')
        option_key = (recipe, 'science_option_observer', 'ScienceOptionObserver')
        option = {key[:4]: value for key, value in raw.items()}
        restored = driver.cache_values(pickle.dumps({old_key: raw, option_key: option}))
        self.assertEqual(len(restored), 2)
        with self.assertRaisesRegex(ValueError, 'semantics'):
            driver.cache_values(pickle.dumps({option_key: raw}))
        mutated = copy.deepcopy(raw); key = next(iter(mutated)); mutated[key][0][0] = 1.
        with self.assertRaisesRegex(ValueError, 'changed an imported'):
            driver.merge_imported({old_key: mutated}, old_key, raw)

    def test_seven_required_A_and_optional_future_winner_are_retained(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i, name in enumerate(driver.REQUIRED_A):
                path = root / name; path.mkdir()
                driver.write(path / 'summary.json', {'accuracy': .5 + i * .01})
            def resolve(path, registry, panel):
                return {'candidate': {'dev': {'accuracy': driver.read(path)['accuracy'], 'mean_compute': 10.}},
                        'model_sha256': path.parent.name}
            with patch.object(driver.selection, 'resolve_a', side_effect=resolve):
                winner, evidence, pending = driver.references(root, 'panel', False)
                self.assertEqual(len(evidence), 7)
                self.assertEqual(winner['model_sha256'], 'development-nonlinear-v1')
                self.assertEqual(len(pending), 1)
                optional = root / driver.OPTIONAL_A[0]; optional.mkdir()
                driver.write(optional / 'summary.json', {'accuracy': .7})
                winner, evidence, pending = driver.references(root, 'panel', False)
                self.assertEqual(winner['model_sha256'], driver.OPTIONAL_A[0])
                self.assertEqual(len(evidence), 8); self.assertEqual(pending, [])
                (root / driver.REQUIRED_A[0] / 'summary.json').unlink()
                with self.assertRaisesRegex(ValueError, 'prerequisite'):
                    driver.references(root, 'panel', False)

    def test_final_selection_does_not_weaken_A_or_B_to_make_gaps(self):
        def candidate(acc, compute=10., calls=1.):
            return {'settings': {}, 'dev': {'accuracy': acc, 'mean_compute': compute, 'mean_judge_calls': calls}}
        a = {'candidate': candidate(.65), 'model_sha256': 'old'}
        future = {'candidate': candidate(.78), 'model_sha256': 'new'}
        b = {'B': candidate(.7, 100.), 'ordinary_B': candidate(.64, 80.), 'matched_B': candidate(.7, 100.)}
        result = driver.final_summary([candidate(.81, 90.)], a, future,
                                      {'stop': {'accuracy': .5}}, b)
        self.assertAlmostEqual(result['C_above_higher_A_B'], .03)
        self.assertFalse(result['C_plus_10_points'])
        self.assertFalse(result['inherited_base_is_strongest_completed_A'])
        self.assertFalse(result['development_gaps_met'])
        none = driver.final_summary([candidate(.99, 101.)], a, a, {'stop': {'accuracy': .5}}, b)
        self.assertIsNone(none['selected_C'])
        self.assertEqual(none['best']['dev']['accuracy'], .99)

    def test_final_seal_detects_class_aware_cache_or_model_manifest_change(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'models').mkdir()
            names = ['registration.json', 'identity.json', 'cache-import.json', 'summary.json', 'search.json',
                'dev-observations.jsonl', 'private-neural-cache.pkl', 'private-neural-caches.pkl',
                'raw-generations.jsonl', 'models/identity.json', 'models/complete.json']
            for name in names:
                (root / name).write_bytes(b'{}\n')
            driver.finalize(root)
            (root / 'private-neural-caches.pkl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'seal'):
                driver.finalize(root)

    def test_full_synthetic_104_policy_grid_and_sealed_resume(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary); data = root / 'data'; (data / 'splits').mkdir(parents=True)
            (data / 'splits/dev-questions.jsonl').write_bytes(b'synthetic-panel\n')
            driver.write(data / 'dev-plain-complete.json', {'synthetic': True})
            source, source_models = root / 'source', root / 'source-models'
            source.mkdir(); source_models.mkdir()
            observer = FakeObserver(root / 'observer', 400_000_000)
            source_model = SimpleNamespace(mentor=SimpleNamespace(classifier=observer))
            source_pickle = source_models / 'source.pkl'; source_pickle.write_bytes(pickle.dumps(source_model))
            manifest = {'models': [{'key': 'source', 'sha256': driver.sha(source_pickle)}]}
            driver.write(source_models / 'complete.json', manifest)
            driver.write(source_models / 'identity.json', {'cohort': 'newsource'})
            recipe = driver.share_key(observer)[0]
            driver.write(source / 'identity.json', {'models_sha256': driver.sha(source_models / 'complete.json'),
                'source_sha256': driver.sha(driver.previous.__file__)})
            driver.write(source / 'metadata-replay-complete.json', {})
            (source / 'raw-generations.jsonl').write_bytes(b'')
            driver.previous.atomic_pickle(source / 'private-neural-cache.pkl', raw_cache())
            for name in ('mixed', 'newsource'):
                (root / name).mkdir(); driver.write(root / name / 'complete.json', {})
            bdir = root / 'B'; bdir.mkdir()
            rows = [{'id': f'q{i}', 'subject': ('math', 'chemistry', 'physics')[i % 3]} for i in range(256)]
            trees = {r['id']: {'': {}} for r in rows}
            def statistics(items):
                return {'n': len(items), 'accuracy': float(np.mean([v['correct'] for v in items])),
                    'mean_compute': float(np.mean([v['compute'] for v in items])),
                    'mean_judge_calls': float(np.mean([v['judge_calls'] for v in items]))}
            base_stats = statistics([{'correct': 0, 'compute': 10., 'judge_calls': 0}] * 256)
            b = {'dev': {'accuracy': .7, 'mean_compute': 100.}}
            driver.write(bdir / 'summary.json', {'B': b, 'ordinary_B': b, 'matched_B': b,
                'baselines': {name: base_stats for name in driver.study.BASELINES}})
            driver.write(bdir / 'identity.json', {})
            selected = {'candidate': {'settings': {}, 'dev': {'accuracy': .6, 'mean_compute': 10.}},
                        'model_sha256': 'synthetic-base'}
            args = SimpleNamespace(data=data, registry=root, mixed_models=root / 'mixed', newsource_models=root / 'newsource',
                source=source, source_models=source_models, b_reference=bdir / 'summary.json',
                output=root / 'output', device='cpu', collect=False)
            def construct(*unused):
                folder = args.output / 'models'; folder.mkdir(exist_ok=True)
                specs = []
                for cohort in ('mixed', 'newsource'):
                    for smoothing in (8., 32.):
                        path = folder / f'{cohort}-{smoothing:g}.pkl'; path.write_bytes(pickle.dumps(source_model))
                        specs.append({'key': path.stem, 'sha256': driver.sha(path), 'cohort': cohort, 'smoothing': smoothing})
                complete = {'models': specs, 'base_A': selected['candidate'], 'observer_assets': []}
                driver.write(folder / 'complete.json', complete)
                driver.write(folder / 'identity.json', {'A_references': [], 'base_selection': selected, 'pending_optional_A': []})
                return complete
            calls = []
            def evaluate(model, settings, actual_rows, actual_trees, judges, acquire, lock):
                calls.append(settings)
                return [{'correct': 1, 'compute': 50., 'judge_calls': 1., 'metadata_calls': 1,
                    'metadata_cpu_seconds': .001, 'judge_physical_calls': 2, 'review_neural_calls': 0} for r in actual_rows]
            patches = [
                patch.object(driver.fitting, 'source_files', return_value={}),
                patch.object(driver, 'references', return_value=(selected, [], [])),
                patch.object(driver, 'construct', side_effect=construct),
                patch.object(driver, 'attach'),
                patch.object(driver.previous, 'wait_completed'),
                patch.object(driver.previous, 'completed_bank', return_value=(b'', {})),
                patch.object(driver.previous, 'model_assets', return_value=(manifest, recipe)),
                patch.object(driver.inherited, 'validate_b'),
                patch.object(driver.study, 'development_view', return_value=(rows, trees, {})),
                patch.object(driver.previous, 'validate_nodes'),
                patch.object(driver.previous, 'evaluate', side_effect=evaluate),
                patch.object(driver.study, 'statistics', side_effect=statistics),
                patch.object(driver.study, 'fixed', return_value={'correct': 0, 'compute': 10., 'judge_calls': 0})]
            for p in patches:
                stack.enter_context(p)
            driver.run(args)
            self.assertEqual(len(calls), 104)
            summary = driver.read(args.output / 'summary.json')
            self.assertTrue(summary['development_gaps_met'])
            self.assertEqual(summary['new_physical_generations'], 0)
            self.assertEqual(len(driver.read(args.output / 'search.json')), 104)
            marker = (args.output / 'stronger-metadata-replay-complete.json').read_bytes()
            driver.run(args)
            self.assertEqual(len(calls), 104)
            self.assertEqual((args.output / 'stronger-metadata-replay-complete.json').read_bytes(), marker)


if __name__ == '__main__':
    unittest.main()
