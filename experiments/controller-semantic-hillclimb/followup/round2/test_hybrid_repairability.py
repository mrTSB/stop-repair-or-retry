import unittest
import numpy as np
import hybrid_repairability as hybrid
from repairability_pomdp import CURRENT
from mentor_pomdp import REWARD


class HybridTransferTest(unittest.TestCase):
    def test_query_transfer_cannot_change_current_answer_truth(self):
        rows, trees, judges = [], {}, {}
        def node(bit):
            return {'answer': '1' if bit else '0'}
        for i in range(192):
            qid = f'synthetic{i}'
            c, r, t = (i >> 2) & 1, (i >> 1) & 1, i & 1
            rows.append({'id': qid, 'answer': '1', 'subject': ('math', 'chemistry', 'physics')[i % 3]})
            trees[qid] = {'': node(c), 'repair': node(r), 'retry': node(t),
                          'repair/repair': node((i // 8) % 2), 'retry/retry': node((i // 16) % 2)}
            judges[qid, ''] = node((i // 8) % 2)
        transfer = hybrid.fit_transfer(rows, trees, judges)
        for matrix in transfer.values():
            np.testing.assert_allclose(matrix.sum(1), 1.)
            np.testing.assert_allclose(matrix @ REWARD, CURRENT, atol=1e-12)
            b = np.arange(1., 9.) / 36
            self.assertAlmostEqual(float((b @ matrix) @ REWARD), float(b @ CURRENT))


if __name__ == '__main__':
    unittest.main()
