"""Synthetic pure-A role, reference blindness, paid-cost and cache checks."""
from contextlib import ExitStack
import copy
import hashlib
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import development_expanded_plain_a4 as run


def fixture():
    observer = run.policy.Observer('synthetic-unloaded', 1.)
    rows = [{'id': s, 'domain': 'math' if s == 'math' else 'science', 'subject': s, 'answer': '1', 'question': 'Public problem'}
            for s in ('math', 'chemistry', 'physics')]
    trees, cache, oof = {}, {}, {}
    for row in rows:
        tree = {}
        for i, path in enumerate(run.study.PLAIN):
            node = {'answer': '1' if path else '0', 'text': 'Paid answer '+str(i),
                'prompt_sha256': hashlib.sha256((row['id']+path).encode()).hexdigest(), 'confidence': None,
                'input_tokens': 100+i, 'output_tokens': 20+i}
            tree[path] = node
            if path in ('', 'repair', 'retry'):
                cache[observer.key(row, path, node, 'solver')] = (np.zeros(4), np.zeros(8), np.zeros(8), 31+i)
                oof[row['id'], path] = np.full(4, .25)
        trees[row['id']] = tree
    model = run.fitting.core.plain_regular(rows, trees, observer, oof, 72., {'synthetic': True})
    recipe = {'parameters': 435_000_000, 'max_length': 512}
    return rows, trees, model, cache, recipe


class PureA4Tests(unittest.TestCase):
    def test_source_ancestor_context_binds_same_plain_completion(self):
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)
            run.io.write(directory/'identity.json',{'source_sha256':run.io.sha(run.previous.__file__),
                'ordinary_plain_completion_sha256':'original'})
            with patch.object(run.inherited,'completed_bank',return_value=(b'',{})):
                with self.assertRaisesRegex(ValueError,'ancestor context'):
                    run.cache_source(directory,'panel','changed')

    def test_fixed14_settings_only_pure_A_context_zero(self):
        self.assertEqual(len(run.SETTINGS), 14)
        self.assertEqual(len({s['penalty'] for s in run.SETTINGS}), 7)
        self.assertEqual({s['continuous_filter'] for s in run.SETTINGS}, {False, True})
        self.assertTrue(all(s['context_weight'] == 0. for s in run.SETTINGS))

    def test_cache_only_refuses_load_missing_review_terminal_and_reference(self):
        rows, trees, model, cache, recipe = fixture(); row = {**rows[0], 'answer': ''}; tree = trees[row['id']]
        observer = model.classifier
        with patch.object(run.policy, 'load', side_effect=AssertionError('No network load')):
            with run.cache_only(observer, cache, recipe):
                observer.load(); self.assertEqual(observer.values(row, '', tree)[3], 31)
                for badrow, path, judge in ((rows[0], '', None), (row, 'repair/repair', None), (row, '', {})):
                    with self.assertRaisesRegex(ValueError, 'Reference/review/terminal'):
                        observer.values(badrow, path, tree, judge)
                altered = copy.deepcopy(tree); altered['']['text'] += 'changed'
                with self.assertRaisesRegex(ValueError, 'missing observation'): observer.values(row, '', altered)
        self.assertNotIn('load', vars(observer)); self.assertNotIn('values', vars(observer))
        self.assertEqual(observer._cache, {}); self.assertEqual(observer._recipe, {})

    def test_actual_runtime_pays_two_observed_nodes_not_forced_terminal(self):
        rows, trees, model, cache, recipe = fixture(); row = rows[0]; tree = trees[row['id']]
        d = model.dynamics[row['subject']]
        with patch.object(d, 'values', return_value=np.array([0., 1., 0.])):
            with run.cache_only(model.classifier, cache, recipe):
                result = run.replay_one(row, tree, model, {'penalty': 0., 'continuous_filter': True, 'context_weight': 0.})
        self.assertEqual(result['path'], 'repair/repair'); self.assertEqual(result['solver_calls'], 3)
        self.assertEqual(result['verifier_calls'], 2); self.assertEqual(result['judge_calls'], 0)
        slots = cache[model.classifier.key(row, '', tree[''], 'solver')][3] + cache[model.classifier.key(row, 'repair', tree['repair'], 'solver')][3]
        self.assertEqual(result['verifier_token_slots'], slots)
        solver_tokens = sum(tree[p]['input_tokens']+tree[p]['output_tokens'] for p in ('', 'repair', 'repair/repair'))
        self.assertAlmostEqual(result['compute'], (4*solver_tokens+.435*slots)/3.09)
        self.assertEqual(result['correct'], 1)

    def test_reference_changes_grading_only_and_never_reaches_policy(self):
        rows, trees, model, cache, recipe = fixture(); row = rows[0]; tree = trees[row['id']]
        observed = []
        def evaluate(safe, paid, judges, passed, settings, family):
            self.assertEqual(safe['answer'], ''); self.assertEqual(judges, {}); self.assertEqual(family, 'A')
            observed.append(copy.deepcopy(safe))
            return {'path': '', 'judge_calls': 0, 'judge_tokens': 0, 'solver_calls': 1,
                'verifier_calls': 1, 'verifier_token_slots': 31, 'decisions': [{'path': '', 'queried': False}], 'correct': 999}
        with run.cache_only(model.classifier, cache, recipe), patch.object(run.policy, 'evaluate', side_effect=evaluate):
            left = run.replay_one(row, tree, model, run.SETTINGS[0])
            right = run.replay_one({**row, 'answer': '0'}, tree, model, run.SETTINGS[0])
        self.assertEqual(observed[0], observed[1]); self.assertEqual(left['path'], right['path'])
        self.assertEqual((left['correct'], right['correct']), (0, 1))

    def test_alignment_requires_all768_without_correctness_filter(self):
        rows, trees, model, cache, recipe = fixture()
        allrows, alltrees, allcache = [], {}, {}
        for i in range(256):
            original = rows[0 if i < 128 else 1 if i < 192 else 2]
            row = {**original, 'id': f'q{i}'}; allrows.append(row)
            tree = copy.deepcopy(trees[original['id']]); alltrees[row['id']] = tree
            for path in ('', 'repair', 'retry'):
                allcache[model.classifier.key(row, path, tree[path], 'solver')] = (np.zeros(4), np.zeros(8), np.zeros(8), 32)
        with patch.object(run.protocol, 'validate_node') as check:
            selected, receipt = run.aligned_cache(model.classifier, allcache, allrows, alltrees)
            self.assertEqual(len(selected), 768); self.assertEqual(len(receipt), 768); self.assertEqual(check.call_count, 256*7)
            self.assertTrue(all(call.args[0]['answer'] == '' for call in check.call_args_list))
            del allcache[next(iter(allcache))]
            with self.assertRaisesRegex(ValueError, 'no inference'): run.aligned_cache(model.classifier, allcache, allrows, alltrees)
            with self.assertRaisesRegex(ValueError, 'entire256'): run.aligned_cache(model.classifier, {}, allrows[:-1], alltrees)

    def test_stronger_prior_A_retained_with_no_four_to_eight_bridge_claim(self):
        candidate = {'model': 'A4', 'settings': {}, 'dev': {'accuracy': .56, 'mean_compute': 900.}}
        prior = {'candidate': {'model': 'best-old', 'settings': {}, 'dev': {'accuracy': .5625, 'mean_compute': 856.}},
            'module': 'subject_science_pomdp', 'model_path': '/synthetic', 'model_sha256': 'hash',
            'summary':'/synthetic/summary.json','summary_sha256':'summaryhash'}
        result = run.select([candidate], prior, {'Repair': {'accuracy': .5}})
        self.assertTrue(result['retained_previous_A']); self.assertEqual(result['selected_A']['model'], 'best-old')
        self.assertFalse(result['C_bridge_supported']); self.assertFalse(result['A_plus_10_points'])
        self.assertEqual(result['new_encoder_calls'], 0); self.assertFalse(result['test_used'])
        self.assertEqual(result['selected_A']['source_summary'],prior['summary'])

    def test_full_synthetic14_candidate_replay_and_sealed_no_work_resume(self):
        rows3, trees3, model, cache, recipe = fixture()
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary); output=root/'output'; source=root/'source'; output.mkdir(); source.mkdir()
            (root/'data').mkdir();run.io.write(root/'data/dev-plain-complete.json',{})
            run.io.write(output/'registration.json', {'synthetic': True})
            args = SimpleNamespace(output=output, models=root/'models', data=root/'data', source=source, registry=root,
                registration_sha256=run.io.sha(output/'registration.json'),plain_features=None)
            rows = [{**rows3[0 if i<128 else 1 if i<192 else 2], 'id':f'q{i}'} for i in range(256)]
            trees = {r['id']:trees3[r['subject']] for r in rows}
            chosen = {'candidate': {'model':'prior', 'settings':{}, 'dev':{'accuracy':.6,'mean_compute':50.}},
                'model_path':'/synthetic','model_sha256':'hash','module':'continuous_repairability',
                'summary':'/synthetic/summary.json','summary_sha256':'summaryhash'}
            record = {'dev_questions_sha256':'panel', 'cache_asset':['recipe','class','name']}
            spec = {'key':'A4','model_path':'/synthetic','sha256':'hash'}
            statistic = lambda values: {'accuracy':float(np.mean([r['correct'] for r in values])),
                                       'mean_compute':float(np.mean([r['compute'] for r in values]))}
            baselines = {name: {'accuracy':.5, 'mean_compute':10.} for name in run.study.BASELINES}
            run.io.write(source/'summary.json', {'baselines':baselines})
            patches = [patch.object(run,'validate_registration',return_value=record),
                patch.object(run,'fitted_model',return_value=(model,spec,recipe,{})),
                patch.object(run,'cache_source',return_value=({tuple(record['cache_asset']):cache},{})),
                patch.object(run,'plain_data',return_value=(rows,trees)),
                patch.object(run,'aligned_cache',return_value=(cache,[])),
                patch.object(run,'replay_one',return_value={'correct':1,'compute':100.}),
                patch.object(run.previous,'references',return_value=(chosen,[],[])),
                patch.object(run.study,'statistics',side_effect=statistic),
                patch.object(run.study,'fixed',return_value={'correct':.5,'compute':10.})]
            mocks=[stack.enter_context(p) for p in patches]
            run.run(args)
            self.assertEqual(mocks[5].call_count,14*256)
            self.assertEqual(len(run.io.read(output/'search.json')),14)
            marker=(output/'plain-A4-replay-complete.json').read_bytes()
            run.run(args); self.assertEqual(mocks[5].call_count,14*256)
            self.assertEqual((output/'plain-A4-replay-complete.json').read_bytes(),marker)
            (output/'selected-neural-cache.pkl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'completed pure-A4'):run.run(args)


if __name__ == '__main__': unittest.main()
