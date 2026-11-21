import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import word_repairability_pomdp as w
import development_word_repairability as driver


def fixture(n=48):
    rows = [{'id': f'q{i}', 'answer': '1', 'subject': ('math', 'chemistry', 'physics')[i % 3],
             'domain': 'math', 'question': 'Compare Co and CO in the synthetic paid problem'} for i in range(n)]
    random = np.random.default_rng(41); trees, lookup = {}, {}
    for row in rows:
        trees[row['id']] = {path: {'answer': str(random.integers(2)), 'text': 'Check the Co molecule. Final: 1',
            'confidence': None, 'input_tokens': 10, 'output_tokens': 8, 'finish': 'stop'}
            for path in ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')}
        for path in ('', 'repair', 'retry'):
            lookup[row['id'], path] = (random.normal(size=4), random.normal(size=8), random.normal(size=8), 19)
    observer = SimpleNamespace(_recipe={'parameters': 400_000_000, 'max_length': 512})
    data = w.text.table(rows, trees, lookup)
    data['answer_present'] = np.ones(len(data['labels']), dtype=bool)
    return rows, trees, lookup, observer, data


class WordTests(unittest.TestCase):
    def test_exact_parent_dynamics_costs_support_and_finite_backup(self):
        rows, trees, lookup, observer, data = fixture()
        with threadpool_limits(limits=1):
            parent = w.core.fit(rows, trees, observer, lookup, smoothing=8.)
            forecast = w.text.fit_forecaster(data['texts'], data['numeric'], data['labels'], 'word12_C1')
        changed = w.build(parent, forecast, data, np.full((len(data['labels']), 8), .125), 8., {})
        for subject, d in changed.dynamics.items():
            self.assertIs(type(d), w.core.Dynamics)
            for key in ('prior', 'transition', 'cost'):
                np.testing.assert_array_equal(getattr(d, key), getattr(parent.dynamics[subject], key))
            np.testing.assert_allclose(d.emission.sum(-1), 1.)
            self.assertTrue(np.all(d.emission[:, w.core.CURRENT == 1, -1] == 0.))
            belief = np.arange(1., 9.); belief /= belief.sum(); penalty = .04
            expected = [belief @ w.core.CURRENT]
            alpha = np.vstack([w.core.CURRENT, w.core.AFTER[0] - penalty*d.cost[1, 0]/1000,
                               w.core.AFTER[1] - penalty*d.cost[1, 1]/1000])
            for action in (0, 1):
                predicted = belief @ d.transition[action]
                future = sum(max(alpha @ (predicted*d.emission[action, :, obs])) for obs in range(w.core.N_OBS))
                expected.append(future - penalty*(belief @ d.cost[0, action])/1000)
            np.testing.assert_allclose(d.values(belief, 2, penalty), expected)
            with self.assertRaises(ValueError):
                d.values(belief, 3, penalty)

    def test_prediction_reads_only_paid_prefix_and_reference_is_blank(self):
        rows, trees, lookup, _, data = fixture()
        with threadpool_limits(limits=1):
            forecast = w.text.fit_forecaster(data['texts'], data['numeric'], data['labels'], 'word12_C1')
        row = rows[0]; tree = trees[row['id']]; test = self
        class Observer:
            def values(self, safe, path, paid):
                test.assertEqual(safe['answer'], '')
                test.assertEqual(set(paid), {'', path})
                return lookup[row['id'], path]
        model = w.Planner(Observer(), forecast, {}, {})
        first = model.predicted(row, 'repair', tree)
        changed = copy.deepcopy(tree)
        for path in set(tree) - {'', 'repair'}:
            changed[path] = {'answer': 'UNPAID SECRET', 'text': 'MALICIOUS FUTURE'}
        second = model.predicted({**row, 'answer': 'DIFFERENT REFERENCE'}, 'repair', changed)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(model.head_receipts[0]['input_sha256'], model.head_receipts[1]['input_sha256'])
        changed['repair']['answer'] = ''
        missing = model.predicted(row, 'repair', changed)
        self.assertEqual(missing[w.core.CURRENT == 1].sum(), 0.)
        self.assertAlmostEqual(missing.sum(), 1.)

    def test_missing_answer_emission_and_subject_pooled_formula(self):
        _, _, _, _, data = fixture()
        data['answer_present'][1] = False; data['labels'][1] = 0
        p = np.full((len(data['labels']), 8), .125)
        actual = w.emissions(data, p, 8.)
        support = np.ones((2, 8, w.core.N_OBS)); support[:, w.core.CURRENT == 1, -1] = 0
        arrays = {}
        for subject in (None, 'math', 'chemistry', 'physics'):
            values = .1*support.copy()
            for i, path in enumerate(data['paths']):
                if path and (subject is None or subject == data['subjects'][i]):
                    signal = w.core.observation(p[i], bool(data['answer_present'][i]))
                    values[int(path == 'retry'), data['labels'][i], signal] += 1
            arrays[subject] = values
        pooled = arrays[None]/arrays[None].sum(-1, keepdims=True)
        for subject, d in actual.items():
            expected = arrays[subject] + 8*pooled
            np.testing.assert_allclose(d, expected/expected.sum(-1, keepdims=True))
        malformed = p.copy(); malformed[0] *= 2
        with self.assertRaises(ValueError):
            w.emissions(data, malformed, 8.)

    def test_registration_precedes_fit_and_cache_missing_never_runs_inference(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); (root/'data/splits').mkdir(parents=True)
            (root/'data/splits/dev-questions.jsonl').write_text('Only hash is used before training fit')
            (root/'diagnostic').mkdir(); (root/'diagnostic/summary.json').write_text('{}')
            args = SimpleNamespace(output=root/'new', base=root/'base', data=root/'data', diagnostic=root/'diagnostic', reference=[])
            _, _, _, _, data = fixture()
            def stop(*args, **kwargs):
                self.assertTrue((root/'new/fitting-registration.json').exists())
                raise RuntimeError('Before full-fit')
            with patch.object(driver, 'training_inputs', return_value=(data, [], None,
                    {'train_question_sha256': 'x', 'encoder_recipe_sha256': 'y'}, {})), \
                    patch.object(driver.diagnostic, 'source_files', return_value={}), \
                    patch.object(w.text, 'fit_forecaster', side_effect=stop):
                with self.assertRaisesRegex(RuntimeError, 'Before full-fit'):
                    driver.fit(args)
        observer = driver.cached.CachedObserver('unused', 1.)
        observer._recipe = {'parameters': 400000000, 'max_length': 512}
        observer.saved = {}; observer.input_sha256 = {}
        with patch.object(observer, 'entry', return_value={'safe': True}), driver.diagnostic.no_model_calls():
            with self.assertRaisesRegex(ValueError, 'no neural fallback'):
                observer.values({'id': 'q', 'answer': 'SECRET'}, '', {'': {'text': 'Paid'}})


if __name__ == '__main__':
    unittest.main()
