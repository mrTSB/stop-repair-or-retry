import unittest
from unittest.mock import patch
import numpy as np
import math_specialization_diagnostic as study


class MathSpecializationTests(unittest.TestCase):
    def test_outer_math_labels_excluded_from_both_fit_and_selection(self):
        rng = np.random.default_rng(238); n = 180
        rows = [{'id': str(i//3), 'subject': 'math' if i < 90 else 'physics', 'fold': i//3 % 3} for i in range(n)]
        data = {'records': rows, 'x': rng.normal(size=(n, 6)), 'y': np.arange(n) % 8,
                'valid': np.ones(n, dtype=bool)}
        with patch.object(study, 'SETTINGS', (study.SETTINGS[0],)):
            held, old, new, receipt = study.outer(data, 1)
            changed = {**data, 'y': data['y'].copy()}; changed['y'][held] = 7-changed['y'][held]
            held2, old2, new2, receipt2 = study.outer(changed, 1)
        np.testing.assert_array_equal(held, held2); np.testing.assert_array_equal(old, old2)
        np.testing.assert_array_equal(new, new2); self.assertEqual(receipt, receipt2)
        self.assertTrue(all(r['subject'] == 'math' for r, flag in zip(rows, held) if flag))

    def test_invalid_answers_have_no_correct_state_mass(self):
        x = np.arange(40).reshape(20, 2); y = np.arange(20) % 8
        with patch.object(study, 'TREES', 3): model = study.fit(x, y, {'kind': 'forest', 'leaf': 8})
        p = study.predict(model, x, np.zeros(20, dtype=bool))
        np.testing.assert_array_equal(p[:, 4:], 0.); np.testing.assert_allclose(p.sum(1), 1.)


if __name__ == '__main__': unittest.main()
