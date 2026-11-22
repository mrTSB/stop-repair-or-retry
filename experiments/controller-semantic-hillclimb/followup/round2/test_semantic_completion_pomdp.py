from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import semantic_completion_pomdp as runtime


class SemanticPOMDPTests(unittest.TestCase):
    def test_invalid_answer_skips_additional_neural_pass(self):
        model = runtime.Planner(None, None, {}, {}, 'raw20_plus_project256',
            SimpleNamespace(values=lambda *a: self.fail('Invalid answer bought a sensor')), None)
        base = np.r_[np.ones(8)/8, np.zeros(8)]
        with patch.object(runtime.previous.Planner, 'predicted', return_value=base):
            result = model.predicted({'subject': 'physics', 'answer': 'SECRET'}, '', {'': {'answer': ''}})
        np.testing.assert_array_equal(result, base)

    def test_valid_answer_sensor_receives_only_paid_blank_reference_prefix(self):
        seen = []
        def values(row, path, tree):
            seen.append((row, path, tree)); return np.zeros(4), np.zeros(8), np.zeros(8), 30
        model = runtime.Planner(None, None, {}, {}, 'raw20_plus_project256',
            SimpleNamespace(values=values, temperature=1.), SimpleNamespace(predict_proba=lambda x: np.array([[.1, .9]])))
        with patch.object(runtime.previous.Planner, 'predicted', return_value=np.ones(16)/16):
            p = model.predicted({'subject': 'physics', 'answer': 'SECRET'}, 'repair',
                {'': {'answer': 'A'}, 'repair': {'answer': 'B'}, 'retry': {'answer': 'C'}})
        self.assertAlmostEqual(float(p @ runtime.core.CURRENT), .9)
        self.assertEqual(len(seen), 1); self.assertEqual(seen[0][0]['answer'], '')
        self.assertEqual(set(seen[0][2]), {'', 'repair'})

    def test_subject_route_preserves_math_and_rejects_llm_family(self):
        with patch.object(runtime.routed, 'evaluate', return_value={'same_math': True}) as old:
            self.assertEqual(runtime.evaluate({'subject': 'math'}, {}, {}, None, {}, 'A'), {'same_math': True})
        self.assertEqual(old.call_count, 1)
        with self.assertRaises(ValueError): runtime.evaluate({'subject': 'physics'}, {}, {}, None, {}, 'C')


if __name__ == '__main__': unittest.main()
