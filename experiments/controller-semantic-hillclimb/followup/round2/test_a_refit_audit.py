"""Synthetic provenance and deterministic-refit checks; no neural inference."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import audit_a_refit as audit
import repairability_pomdp as core
import factorized_repairability as factorized


class FakeObserver:
    def entry(self, row, path, tree):
        if row['answer']:
            raise AssertionError('Reference answer entered a neural input')
        if set(tree) != {'', path}:
            raise AssertionError('Unpaid solver response entered neural input')
        return {'left': row['question'], 'right': tree[path]['answer'], 'numeric': [float(bool(path))]}


class RefitTests(unittest.TestCase):
    def model(self, factor=False):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(96, 9)); x[:, -4:] = 0
        x[:, -4] = np.arange(96) % 2
        x[np.arange(96), 6 + np.arange(96) % 3] = 1
        y = np.arange(96) % 8
        predictor = factorized.fit_forecaster(x, y) if factor else make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)).fit(x, y)
        receipt = dict(zip(audit.BASE_RECEIPT, [8, ['current_correct', 'repair_correct', 'retry_correct'],
            32, 96, 3, 1., 8., False, 'two calls', 'finite observations']))
        if factor:
            receipt.update(prior_smoothing=16., ipf_marginal_clip=[1e-6, 1 - 1e-6],
                           emission_folds=[], interaction_odds_ratios_preserved=True)
        dynamics = core.Dynamics(np.ones(8) / 8, np.ones((2, 8, 8)) / 8,
                                 np.ones((2, 8, core.N_OBS)) / core.N_OBS, np.ones((2, 2, 8)))
        return core.Planner(None, predictor, {'math': dynamics}, receipt)

    def test_comparison_covers_parameters_dynamics_and_ignores_runtime_cache(self):
        for factor in (False, True):
            a = self.model(factor); b = copy.deepcopy(a)
            if factor:
                b.forecaster._cache[b'irrelevant'] = np.ones(8)
            result = audit.compare_learned(a, b)
            self.assertTrue(result['exact_numeric_equality'])
            self.assertEqual(result['learned_state_sha256'], result['reconstructed_state_sha256'])
            b.dynamics['math'].emission[0, 0, 0] += .001
            with self.assertRaisesRegex(ValueError, 'emission'):
                audit.compare_learned(a, b)
            b = copy.deepcopy(a)
            pipeline = b.forecaster.binary[0] if factor else b.forecaster
            pipeline.steps[-1][1].coef_[0, 0] += .001
            with self.assertRaisesRegex(ValueError, 'coef_'):
                audit.compare_learned(a, b)

    def samples(self):
        rows = [{'id': 'q', 'question': 'question', 'answer': 'secret', 'subject': 'math'}]
        trees = {'q': {p: {'answer': p, 'prompt_sha256': 'prompt' + p} for p in audit.PLAIN}}
        observer = FakeObserver()
        metadata = []
        for p in audit.PATHS:
            entry = observer.entry({**rows[0], 'answer': ''}, p, {q: trees['q'][q] for q in ('', p)})
            metadata.append({'id': 'q', 'path': p, 'input_sha256': audit.fingerprint(entry), 'prompt_sha256': 'prompt' + p})
        outputs = {k: np.zeros((3, width)) for k, width in (('state', 4), ('plain', 8), ('guided', 8))}
        return rows, trees, observer, metadata, outputs

    def test_cache_requires_exact_reference_blind_paid_input_and_prompt(self):
        rows, trees, observer, metadata, outputs = self.samples()
        lookup, receipt = audit.validate_cache(metadata, outputs, rows, trees, observer, {'q'})
        self.assertEqual(len(lookup), 3); self.assertEqual(len(receipt), 3)
        for field in ('input_sha256', 'prompt_sha256'):
            changed = copy.deepcopy(metadata); changed[0][field] = 'changed'
            with self.assertRaises(ValueError):
                audit.validate_cache(changed, outputs, rows, trees, observer, {'q'})
        with self.assertRaises(ValueError):
            audit.validate_cache(metadata[:-1], outputs, rows, trees, observer, {'q'})
        outputs['state'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            audit.validate_cache(metadata, outputs, rows, trees, observer, {'q'})

    def test_input_guard_rejects_held_out_files_and_changed_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); inputs = audit.Inputs()
            for name in ('test-questions.jsonl', 'dev-questions.jsonl', 'test-observations.jsonl'):
                # Refusal must occur before even checking whether the file exists.
                with self.assertRaisesRegex(ValueError, 'held-out'):
                    inputs.read(root / name)
            path = root / 'train.jsonl'; path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'Changed recorded input'):
                inputs.read(path, 'not a digest')
            inputs.read(path); path.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, 'during audit'):
                inputs.read(path)

    def test_neural_guard_and_fusion_cost_recipe(self):
        from dynamic_grounded_pomdp import Observer
        from science_option_observer import ScienceOptionObserver
        from science_fusion_pomdp import FusionObserver
        with audit.forbid_neural_calls():
            for observer in (Observer('unused', 1.), ScienceOptionObserver('unused', 1.), FusionObserver([])):
                with self.assertRaisesRegex(AssertionError, 'forbidden'):
                    observer.load()
                with self.assertRaisesRegex(AssertionError, 'forbidden'):
                    observer.values(None, None, None)
        self.assertEqual(Observer.load.__name__, 'load')

    def test_recursive_subject_audits_shared_child_once(self):
        from subject_science_pomdp import SubjectPlanner
        engine = audit.RefitAudit(SimpleNamespace(), audit.Inputs(), [], {})
        child = self.model()
        # Pre-audited leaf stands in for an expensive fit; wrapper must retain
        # both routed uses while recognizing their shared fitted artifact.
        engine.seen[id(child)] = 'earlier'
        wrapper = SubjectPlanner({s: child for s in ('math', 'chemistry', 'physics')},
                                 {s: {'penalty': 0.1} for s in ('math', 'chemistry', 'physics')}, {})
        layout = engine.reconstruct(wrapper)
        self.assertEqual(set(layout['children']), {'math', 'chemistry', 'physics'})
        self.assertTrue(all(v == {'reuse_component': 'earlier'} for v in layout['children'].values()))

    def test_nonlinear_comparison_covers_tree_bins_and_calibration(self):
        import nonlinear_repairability as nonlinear
        model = self.model()
        rng = np.random.default_rng(8)
        x = rng.normal(size=(96, 5)); y = np.arange(96) % 8
        model.forecaster = nonlinear.Forecaster(nonlinear.raw_fit(x, y, 7), 1.5)
        model.fit_receipt.pop('calibration_C')
        model.fit_receipt.update(fit_module='nonlinear_repairability', forecaster_hyperparameters={'max_leaf_nodes': 7},
            temperature_grid=list(nonlinear.TEMPERATURES), calibration={'outer_folds': []},
            feature_sha256='synthetic', training_question_ids_sha256='synthetic',
            forecaster_and_temperature_exclude_own_question_in_emission_fit=True,
            structural_transitions_and_costs_unchanged=True)
        self.assertTrue(audit.compare_learned(model, copy.deepcopy(model))['exact_numeric_equality'])
        changed = copy.deepcopy(model); changed.forecaster.temperature += 0.1
        with self.assertRaisesRegex(ValueError, 'temperature'):
            audit.compare_learned(model, changed)
        changed = copy.deepcopy(model)
        changed.forecaster.estimator._bin_mapper.bin_thresholds_[0][0] += 0.1
        with self.assertRaisesRegex(ValueError, 'thresholds'):
            audit.compare_learned(model, changed)
        changed = copy.deepcopy(model)
        changed.forecaster.estimator._predictors[0][0].nodes['value'][0] += 0.1
        with self.assertRaisesRegex(ValueError, 'nodes/value'):
            audit.compare_learned(model, changed)

    def test_waiter_only_accepts_completed_development_summary(self):
        from audit_a_refit_wait import completed
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'summary.json'
            self.assertFalse(completed(path))
            path.write_text('{')
            self.assertFalse(completed(path))
            path.write_text(json.dumps({'selected_A': {}, 'test_used': False}))
            self.assertTrue(completed(path))
            path.write_text(json.dumps({'selected_A': {}, 'test_used': True}))
            with self.assertRaises(ValueError):
                completed(path)

    def test_declared_dependency_closure_includes_legacy_scoring_not_unrelated_drivers(self):
        with patch.dict(audit.sys.modules, {'unrelated_live_module': SimpleNamespace(
                __file__=str(audit.HERE / 'development_nonlinear.py'))}):
            paths = audit.project_sources(declare_supported=True)
        self.assertIn(audit.HERE / 'nonlinear_repairability.py', paths)
        self.assertIn(audit.HERE / 'science_fusion_pomdp.py', paths)
        self.assertTrue(any('results/cached-tree-dataset' in str(p) and p.name == 'scoring.py' for p in paths))
        self.assertTrue(any('controller-hillclimb' in str(p) and p.name == 'difficulty_pomdp.py' for p in paths))
        self.assertNotIn(audit.HERE / 'development_nonlinear.py', paths)
        self.assertNotIn(audit.HERE / 'development_stronger_base.py', paths)
        self.assertTrue(all(not any(part.startswith('.') for part in p.relative_to(audit.PROJECT).parts)
                            for p in paths))


if __name__ == '__main__':
    unittest.main()
