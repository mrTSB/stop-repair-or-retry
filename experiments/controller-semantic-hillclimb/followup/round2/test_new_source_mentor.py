import copy
import inspect
import unittest

import numpy as np

import fit_new_source_mentor as run
import new_source_mentor_core as core


class NewSourceMentorTest(unittest.TestCase):
    def test_derived_fitter_preserves_exact_parent_operations(self):
        expected = inspect.getsource(run.mentor.fit).replace('def fit(', 'def fit_new(')
        expected = expected.replace('len(paired) != 192', 'len(paired) != 128')
        expected = expected.replace('all192matched', 'all128new-source matched')
        expected = expected.replace('all384mentor-reuse', 'all256new-source mentor-reuse')
        expected = expected.replace("'paired_root_questions': 192", "'paired_root_questions': 128")
        expected = expected.replace("'paired_reuse_outcomes': 384", "'paired_reuse_outcomes': 256")
        expected = expected.replace('return Planner(', 'return MentorPlanner(')
        self.assertEqual(inspect.getsource(core.fit_new).strip(), expected.strip())
        expected = inspect.getsource(run.hybrid.fit_transfer).replace('def fit_transfer(', 'def fit_transfer_new(')
        expected = expected.replace('len(paired) != 192', 'len(paired) != 128')
        expected = expected.replace('all192registered mixed', 'all128registered new-source')
        self.assertEqual(inspect.getsource(core.fit_transfer_new).strip(), expected.strip())
        expected = inspect.getsource(run.hybrid.build).replace('def build(', 'def build_new(')
        expected = expected.replace('return Planner(', 'return HybridPlanner(').replace('fit_transfer(rows,', 'fit_transfer_new(rows,')
        self.assertEqual(inspect.getsource(core.build_new).strip(), expected.strip())

    def fixture(self):
        rows = [{'id': f'{subject}-{i}', 'subject': subject, 'answer': 'A'}
                for subject, n in (('math', 256), ('chemistry', 128), ('physics', 128)) for i in range(n)]
        selected = [{'id': f'{subject}-{i}', 'subject': subject, 'bank': 'new'}
                    for subject, n in (('math', 64), ('chemistry', 32), ('physics', 32)) for i in range(n)]
        selected += [{'id': f'old-{i}', 'subject': 'math', 'bank': 'old'} for i in range(64)]
        return rows, selected

    def test_selection_is_registered_source_only(self):
        rows, selection = self.fixture()
        chosen, excluded = run.select_new(rows, selection)
        self.assertEqual((len(chosen), len(excluded)), (128, 64))
        changed = [dict(row, answer='CHANGED') for row in rows]
        self.assertEqual(run.select_new(changed, selection), (chosen, excluded))
        with self.assertRaises(ValueError):
            run.select_new(rows[:-1], selection)
        wrong = copy.deepcopy(selection); wrong[0]['bank'] = 'old'
        with self.assertRaises(ValueError):
            run.select_new(rows, wrong)

    def test_cache_alignment_retains_exact_values_and_rejects_tampering(self):
        metadata = [{'id': qid, 'kind': 'solver', 'path': '', 'fold': 'full',
                     'input_sha256': qid + '-input', 'prompt_sha256': qid + '-prompt', 'label': 1}
                    for qid in ('old', 'a', 'b')]
        outputs = {name: np.arange(3 * width).reshape(3, width)
                   for name, width in (('state', 4), ('plain', 8), ('guided', 8))}
        selected, indexes = run.aligned_subset(metadata, outputs, metadata[1:])
        self.assertEqual(indexes, [1, 2])
        for name in outputs:
            np.testing.assert_array_equal(selected[name], outputs[name][1:])
        for key, changed in (('fold', 0), ('label', 0), ('prompt_sha256', 'bad'), ('input_sha256', 'bad')):
            expected = copy.deepcopy(metadata[1:]); expected[0][key] = changed
            with self.assertRaises(ValueError):
                run.aligned_subset(metadata, outputs, expected)

    def test_transfer_preserves_current_truth_and_probability_support(self):
        rows = [{'id': f'q{i}', 'subject': ('math', 'chemistry', 'physics')[i % 3], 'answer': 'A'} for i in range(128)]
        trees = {row['id']: {path: {'answer': 'A' if (i + j) % 3 == 0 else 'B'}
                             for j, path in enumerate(core.FIXED)} for i, row in enumerate(rows)}
        judges = {(row['id'], ''): {'answer': 'A' if i % 2 else 'B'} for i, row in enumerate(rows)}
        transfer = core.fit_transfer_new(rows, trees, judges, 8.)
        for matrix in transfer.values():
            np.testing.assert_allclose(matrix.sum(1), 1.)
            np.testing.assert_allclose(matrix @ run.mentor.REWARD, core.action_core.CURRENT)
            self.assertTrue(np.all(matrix[np.asarray(core.action_core.CURRENT)[:, None] != run.mentor.REWARD[None]] == 0))
        with self.assertRaises(ValueError):
            core.fit_transfer_new(rows[:-1], trees, judges, 8.)

    def test_neural_guard_fails_before_any_observer_load(self):
        observer = run.mentor.perception.Observer('unused', 1.)
        with run.no_neural_inference():
            with self.assertRaisesRegex(AssertionError, 'CPU-only'):
                observer.load()


if __name__ == '__main__':
    unittest.main()
