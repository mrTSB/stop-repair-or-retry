import copy
import unittest
import numpy as np

import expanded1280_completion_data as dataset
import fit_expanded1280_completion as fit


def block(prefix, count):
    rows = [{'id': prefix+str(i), 'subject': 'chemistry' if i < count//2 else 'physics'} for i in range(count)]
    records = [{'id': r['id'], 'path': p} for r in rows for p in ('', 'repair', 'retry')]
    data = {'records': records, 'features': {f: np.zeros((count*3, width)) for f, width in zip(dataset.nested.FAMILIES, (130, 386))},
            'labels': np.zeros(count*3, dtype=int), 'valid': np.zeros(count*3, dtype=bool)}
    return rows, {r['id']: {} for r in rows}, data


class Expanded1280Tests(unittest.TestCase):
    def test_exact_disjoint_block_join_preserves_common768_order(self):
        old = block('old', 768); new = block('new', 512)
        rows, trees, data = dataset.merge(*old, *new)
        self.assertEqual(len(rows), 1280); self.assertEqual(len(trees), 1280)
        self.assertEqual(data['records'][:2304], old[2]['records'])
        changed = copy.deepcopy(new); changed[0][0]['id'] = old[0][0]['id']
        with self.assertRaises(ValueError): dataset.merge(*old, *changed)
        changed = copy.deepcopy(new); changed[2]['records'][0]['path'] = 'unpaid'
        with self.assertRaises(ValueError): dataset.merge(*old, *changed)

    def test_identical_common_question_predictions_have_zero_deltas(self):
        old = block('old', 768); p = np.ones((2304, 16))/16
        report = fit.comparison(old[2]['labels'], p, p, old[2]['records'])
        for value in report.values():
            self.assertEqual(value['delta'], 0.); self.assertEqual(value['paired_question_CI95'], [0., 0.])


if __name__ == '__main__': unittest.main()
