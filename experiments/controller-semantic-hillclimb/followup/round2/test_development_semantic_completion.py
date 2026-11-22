import unittest
import numpy as np
import development_semantic_completion as dev


class CachedSensorTests(unittest.TestCase):
    def sensor(self):
        sensor = dev.CachedSemantic('unused', 1.)
        sensor._recipe = {'sealed': True}
        sensor.entry = lambda row, path, tree: {'answer': tree[path]['answer']}
        sensor.expected = {('q', ''): dev.fingerprint({'answer': 'B'})}
        sensor.saved = {('q', ''): (np.zeros(4), np.zeros(8), np.zeros(8), 42)}
        return sensor

    def test_exact_paid_input_returns_charged_slots(self):
        result = self.sensor().values({'id': 'q', 'answer': ''}, '', {'': {'answer': 'B'}})
        self.assertEqual(result[3], 42)

    def test_references_unpaid_branches_and_changed_inputs_rejected(self):
        sensor = self.sensor()
        for row, tree in [({'id': 'q', 'answer': 'SECRET'}, {'': {'answer': 'B'}}),
                          ({'id': 'q', 'answer': ''}, {'': {'answer': 'B'}, 'repair': {'answer': 'C'}}),
                          ({'id': 'q', 'answer': ''}, {'': {'answer': 'C'}})]:
            with self.assertRaises(ValueError): sensor.values(row, '', tree)


if __name__ == '__main__': unittest.main()
