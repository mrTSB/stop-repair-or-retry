import copy
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import prospective_freeze as freeze
import prospective_audit as audit


def metrics(correct, compute, reviews=0.):
    return {'n': 256, 'correct': correct, 'accuracy': correct / 256,
            'mean_compute': compute, 'mean_judge_calls': reviews}


def selections():
    baseline = {name: metrics(128 if name == 'Repair twice' else 120, 100.) for name in freeze.BASELINES}
    chosen = {'A': {'dev': metrics(156, 150.)}, 'B': {'dev': metrics(164, 200., 1.)},
              'C': {'dev': metrics(192, 190., .6)}}
    return chosen, baseline


def outcome(path, tree, reviews, decisions, encoders=()):
    solver = sum(n['input_tokens'] + n['output_tokens'] for n in tree.values())
    judge = sum(n['input_tokens'] + n['output_tokens'] for n in reviews.values())
    encoder = sum(r['slots'] * r['parameters_billions'] for r in encoders)
    return {'path': path, 'correct': 999, 'decisions': decisions,
            'solver_tokens': solver, 'judge_tokens': judge, 'solver_calls': len(tree),
            'judge_calls': len(reviews), 'compute': (4 * solver + 3 * judge + encoder) / 3.09,
            'total_parameter_compute': (4 * solver + 35 * judge + encoder) / 3.09,
            'verifier_calls': len(encoders), 'verifier_tokens': sum(r['slots'] for r in encoders),
            'verifier_token_slots': sum(r['slots'] for r in encoders)}


class ProspectiveTests(unittest.TestCase):
    def test_forwarded_A_resolves_original_runtime_and_digest_bound_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original'; original.mkdir()
            forwarded = root / 'fusion'; forwarded.mkdir()
            model = original / 'C1-smooth8.pkl'; model.write_bytes(b'original fitted planner')
            (forwarded / model.name).write_bytes(b'wrong model with the same name')
            candidate = {'model': model.stem}
            identity = {'source_sha256': freeze.sha(freeze.HERE / 'development_continuous.py'),
                        'model_sha256': {model.name: freeze.sha(model)}}
            self.assertEqual(freeze.infer_a_module(candidate, identity, 'science_fusion_pomdp'), 'continuous_repairability')
            resolved = freeze.resolve_a_model(candidate, identity, original / 'summary.json', forwarded, root)
            self.assertEqual(resolved, model.resolve())
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'resolve recorded A'):
                freeze.resolve_a_model(candidate, identity, original / 'summary.json', forwarded, root)

    def test_stronger_base_provenance_binds_original_mentor_and_selected_A(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(folder, name, data):
                folder.mkdir(exist_ok=True)
                path = folder / name
                path.write_text(json.dumps(data))
                return path
            parent = root / 'parent'
            parent_identity = write(parent, 'identity.json', {'source_sha256': freeze.sha(freeze.HERE / 'prepare_mixed_mentor.py'), 'model_files': {}})
            parent_model = parent / 'mentor.pkl'; parent_model.write_bytes(b'original mentor')
            parent_complete = write(parent, 'complete.json', {'identity_sha256': freeze.sha(parent_identity),
                'models': [{'key': 'mentor', 'sha256': freeze.sha(parent_model)}]})
            a = root / 'A'; a.mkdir()
            a_model = a / 'model.pkl'; a_model.write_bytes(b'stronger plain A')
            a_identity = write(a, 'identity.json', {})
            candidate = {'model': 'model', 'settings': {'penalty': .01}}
            a_summary = write(a, 'summary.json', {'selected_A': candidate})
            selection = {'candidate': candidate, 'module': 'continuous_repairability', 'summary': str(a_summary),
                         'summary_sha256': freeze.sha(a_summary), 'identity_sha256': freeze.sha(a_identity),
                         'model_path': str(a_model), 'model_sha256': freeze.sha(a_model)}
            child = root / 'child'
            child_identity = write(child, 'identity.json', {'source_sha256': freeze.sha(freeze.HERE / 'development_stronger_base.py'),
                'hybrid_sha256': freeze.sha(freeze.HERE / 'stronger_base_hybrid.py'), 'parent_mentor_directory': str(parent),
                'stronger_base_parent_complete_sha256': freeze.sha(parent_complete), 'parent_identity_sha256': freeze.sha(parent_identity),
                'base_selection': selection, 'development_or_test_used_for_mentor_refit': False,
                'mentor_refitted': False, 'transfer_refitted': False, 'test_used': False})
            write(child, 'complete.json', {'identity_sha256': freeze.sha(child_identity), 'base_A': candidate,
                'parent_mentor_complete_sha256': freeze.sha(parent_complete),
                'models': [{'key': 'strongbase-mentor', 'parent_sha256': freeze.sha(parent_model)}]})
            _, files = freeze.fitting_chain(child, root)
            self.assertIn(parent_model, files)
            self.assertIn(a_model, files)
            parent_model.write_bytes(b'changed mentor')
            with self.assertRaisesRegex(ValueError, 'parent mentor model changed'):
                freeze.fitting_chain(child, root)

    def test_shared_encoder_modes_charge_one_identical_paid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'recipe.json').write_text('{"parameters":400000000,"max_length":512}')
            class Observer:
                def __init__(self):
                    self.directory = str(folder)
                    self._recipe = {'parameters': 400000000, 'max_length': 512}
                def load(self):
                    pass
                def entry(self, row, path, tree, judge=None):
                    if set(tree) != {''}:
                        raise AssertionError('future response reached encoder')
                    return {'left': row['id'], 'right': tree[path]['answer'] if judge is None else judge['answer']}
                def values(self, row, path, tree, judge=None):
                    self.entry(row, path, tree, judge)
                    return [0.] * 4, [0.] * 8, [0.] * 8, 20
            first, second = Observer(), Observer()
            model = SimpleNamespace(classifier=first, mentor=SimpleNamespace(classifier=second))
            row = {'id': 'q', 'answer': ''}
            tree = {'': {'answer': 'A'}, 'retry': {'answer': 'UNPAID'}}
            with audit.encoder_ledger(model) as ledger:
                first.values(row, '', tree)
                second.values(row, '', tree)
                self.assertEqual(len(ledger), 1)
                second.values(row, '', tree, {'answer': 'B'})
                self.assertEqual(len(ledger), 2)
            self.assertNotIn('values', vars(first))
            self.assertNotIn('values', vars(second))

    def test_registry_requires_completed_matched_B_and_preserves_stronger_A(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / 'data'; (data / 'splits').mkdir(parents=True)
            q = data / 'splits/dev-questions.jsonl'; q.write_bytes(b'opaque development data')
            dev_sha = freeze.sha(q)
            def write_study(name, summary, identity, search=None):
                path = root / name; path.mkdir()
                (path / 'identity.json').write_text(json.dumps({'dev_questions_sha256': dev_sha, 'test_used': False, **identity}))
                if summary is not None:
                    (path / 'summary.json').write_text(json.dumps({'test_used': False, **summary}))
                if search is not None:
                    (path / 'search.json').write_text(json.dumps(search))
                return path / 'summary.json'
            low = write_study('A-low', {'selected_A': {'dev': metrics(156, 150.)}}, {})
            write_study('A-high', {'selected_A': {'dev': metrics(170, 150.)}}, {})
            ordinary = [{'family': 'B', 'late': True, 'settings': {'judge_stop': stop, 'reuse_review': reuse},
                         'dev': metrics(164, 200., 1.)}
                        for stop, reuse in product(('action', 'agreement', .8, .95, .995, 1.), (False, True))]
            opath = write_study('B-ordinary', {'B': ordinary[0]},
                               {'source_sha256': freeze.sha(freeze.HERE / 'development_b.py')}, ordinary)
            with self.assertRaises(ValueError):
                freeze.development_registry(root, dev_sha)
            matched = [{'family': 'B', 'protocol': 'matched-retained-1024-512-128',
                        'settings': {'root_stop': stop, 'second_stop': second, 'second_action': action},
                        'dev': metrics(180, 300., 1.)}
                       for stop, second, action in product(('action', 'agreement', .8, .95, .995, 1.),
                                                           ('agreement', 'never'), ('reuse', 'retry', 'root_action'))]
            retained = {**ordinary[0], 'protocol': 'ordinary-late', 'source_summary_sha256': freeze.sha(opath)}
            mpath = write_study('B-matched', None, {'source_sha256': freeze.sha(freeze.HERE / 'matched_review_b.py'),
                                'ordinary_B_source': str(opath), 'root_stop_rules': list(range(6))}, matched)
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                freeze.development_registry(root, dev_sha)
            mpath.write_text(json.dumps({'test_used': False, 'B': matched[0], 'matched_B': matched[0], 'ordinary_B': retained}))
            registry = freeze.development_registry(root, dev_sha)
            self.assertEqual(freeze.best([r['candidate'] for r in registry['B']])['module'], 'matched_review_b')
            # A C report may quote B for comparison; it is not a new B study.
            cpath = write_study('C', {'B': ordinary[0]}, {})
            args = SimpleNamespace(data=data, a_summary=low, b_summary=opath, c_summary=cpath)
            with self.assertRaisesRegex(ValueError, 'strongest completed same-panel A'):
                freeze.select_inputs(args)

    def test_targets_use_points_and_strict_compute_and_all_five_baselines(self):
        chosen, baselines = selections()
        self.assertTrue(freeze.check_targets(chosen, baselines)['all_required_passed'])
        chosen['A']['dev'] = metrics(150, 150.)
        self.assertFalse(freeze.check_targets(chosen, baselines)['criteria']['A_above_highest_baseline'])
        chosen, baselines = selections()
        chosen['C']['dev']['mean_compute'] = chosen['B']['dev']['mean_compute']
        self.assertFalse(freeze.check_targets(chosen, baselines)['criteria']['C_compute_below_B'])
        baselines.pop('Repair twice')
        with self.assertRaises(ValueError):
            freeze.check_targets(chosen, baselines)

    def test_auxiliary_subject_observers_and_shared_hybrid_assets_are_discovered(self):
        observer = SimpleNamespace(directory='old')
        science = SimpleNamespace(directory='science')
        a = SimpleNamespace(classifier=observer)
        b = SimpleNamespace(classifier=science)
        routed = SimpleNamespace(models={'math': a, 'chemistry': b, 'physics': b})
        self.assertEqual(len(freeze.observer_slots(routed)), 2)
        mentor = SimpleNamespace(classifier=observer, regular=a)
        hybrid = SimpleNamespace(classifier=observer, base=a, mentor=mentor)
        self.assertEqual(len(freeze.observer_slots(hybrid)), 1)

    def test_archive_deduplicates_and_detects_weight_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'source'; weights.mkdir()
            (weights / 'weight.bin').write_bytes(b'synthetic weights')
            recipe = {'weights': {'weight.bin': freeze.sha(weights / 'weight.bin')}, 'parameters': 100, 'max_length': 32}
            (weights / 'recipe.json').write_text(json.dumps(recipe))
            out = root / 'archive'; (out / 'models').mkdir(parents=True)
            assets = {}
            for family in ('A', 'C'):
                model = SimpleNamespace(classifier=SimpleNamespace(directory=str(weights)))
                freeze.archive_model(model, family, out, assets)
            self.assertEqual(len(assets), 1)
            destination = out / next(iter(assets.values()))['directory']
            freeze.verify_weight_folder(destination)
            (destination / 'weight.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                freeze.verify_weight_folder(destination)

    def test_manifest_external_digest_and_runtime_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / 'study'; out.mkdir()
            (root / 'data/splits').mkdir(parents=True)
            # Bytes are hashed; this deliberately is not parseable question JSON.
            test = root / 'data/splits/test-questions.jsonl'; test.write_bytes(b'opaque reserved data')
            (root / 'experiments').mkdir()
            code = root / 'experiments/runtime.py'; code.write_text('pass\n')
            archived = out / 'runtime/experiments/runtime.py'; archived.parent.mkdir(parents=True)
            archived.write_bytes(code.read_bytes())
            chosen, baselines = selections()
            manifest = {'schema': 'round2-prospective-freeze-v1', 'questions': 1024,
                        'selected': chosen, 'development_baselines': baselines,
                        'project_directory': str(root), 'data_directory': str(root / 'data'),
                        'reserved_question_sha256': freeze.sha(test), 'runtime_sources': {'experiments/runtime.py': freeze.sha(code)},
                        'evidence': {}, 'models': {}, 'observer_assets': {}}
            path = out / 'frozen.json'; path.write_text(json.dumps(manifest))
            digest = freeze.sha(path)
            freeze.verify(out, digest)
            with self.assertRaises(ValueError):
                freeze.verify(out, 'wrong')
            code.write_text('raise RuntimeError()\n')
            with self.assertRaises(ValueError):
                freeze.verify(out, digest)

    def test_blind_replay_never_exposes_sibling_or_reference(self):
        node = {'input_tokens': 10, 'output_tokens': 5}
        available = {'': node, 'repair': node, 'retry': {**node, 'secret': 'UNPAID'}}
        def policy(row, tree, reviews):
            self.assertEqual(row['answer'], '')
            self.assertNotIn('retry', tree)
            self.assertEqual(reviews, {})
            if 'repair' not in tree:
                raise audit.MissingObservation('solver', 'repair')
            return outcome('repair', tree, reviews, [{'path': '', 'queried': False}])
        result, ledger = audit.replay_paid_inputs(policy, {'id': 'q', 'answer': 'secret'}, available, {}, 'A')
        self.assertNotIn('correct', result)
        self.assertEqual(ledger, [])
        def fixed_twice(row, paid, reviews):
            # Fixed policies use normal mapping indexing rather than the
            # adaptive controllers' MissingObservation exception.
            for p in ('', 'repair', 'repair/repair'):
                paid[p]
            return outcome('repair/repair', paid, reviews, [])
        result, _ = audit.replay_paid_inputs(fixed_twice, {'id': 'q', 'answer': 'secret'},
            {**available, 'repair/repair': node}, {}, 'baseline')
        self.assertEqual(result['solver_calls'], 3)

    def test_physical_review_phases_and_neural_work_are_charged(self):
        tree = {'': {'input_tokens': 10, 'output_tokens': 5}}
        review = {'input_tokens': 11, 'output_tokens': 9, 'physical_api_calls': 2,
                  'calls': [{'input_tokens': 5, 'output_tokens': 4}, {'input_tokens': 6, 'output_tokens': 5}]}
        reviews = {('q', ''): review}
        neural = [{'slots': 12, 'parameters_billions': .4}]
        item = outcome('', tree, reviews, [{'path': '', 'queried': True}], neural)
        result = audit.audit_costs(item, tree, reviews, neural, 'C')
        self.assertEqual(result['logical_review_calls'], 1)
        self.assertEqual(result['physical_review_calls'], 2)
        underpriced = {**item, 'compute': item['compute'] - 1}
        with self.assertRaises(ValueError):
            audit.audit_costs(underpriced, tree, reviews, neural, 'C')
        malformed = copy.deepcopy(reviews); malformed['q', '']['calls'][0]['output_tokens'] += 1
        with self.assertRaises(ValueError):
            audit.audit_costs(item, tree, malformed, neural, 'C')

    def test_A_cannot_buy_reviews_or_a_third_continuation(self):
        node = {'input_tokens': 1, 'output_tokens': 1}
        def reviewing(row, tree, reviews):
            raise audit.MissingObservation('judge', '')
        with self.assertRaises(ValueError):
            audit.replay_paid_inputs(reviewing, {'id': 'q', 'answer': 'x'}, {'': node}, {}, 'A')
        def too_long(row, tree, reviews):
            latest = max(tree, key=len)
            raise audit.MissingObservation('solver', latest + '/repair' if latest else 'repair')
        with self.assertRaises(ValueError):
            audit.replay_paid_inputs(too_long, {'id': 'q', 'answer': 'x'},
                                    {p: node for p in ('', 'repair', 'repair/repair')}, {}, 'A')


if __name__ == '__main__':
    unittest.main()
