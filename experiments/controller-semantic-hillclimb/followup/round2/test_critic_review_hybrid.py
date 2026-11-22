from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import critic_review_hybrid as runtime


class CriticHybridTests(unittest.TestCase):
    def test_quantiles_preserve_action_agreement_and_invalid_signals(self):
        edges = np.array([.1, .2, .3, .4]); node = {'answer': 'A'}
        judge = {'answer': 'A', 'parse_valid': True, 'action': 'repair'}
        self.assertEqual(runtime.signal(judge, node, .25, edges), (5+2)*2+1)
        self.assertEqual(runtime.signal({**judge, 'answer': ''}, node, .25, edges), 31)
        self.assertEqual(runtime.signal({**judge, 'parse_valid': False}, node, .25, edges), 30)

    def test_math_routes_unchanged(self):
        with patch.object(runtime.original, 'evaluate', return_value={'original': True}) as call:
            result = runtime.evaluate({'subject': 'math'}, {}, {}, None, {}, 'C')
        self.assertEqual(result, {'original': True}); self.assertEqual(call.call_count, 1)

    def test_combined_bills_both_observers_option_only_bills_one(self):
        values = lambda *a: (np.array([0., 1., 0., 1.]), None, None, 100)
        old, new = [SimpleNamespace(values=values, temperature=1.) for _ in range(2)]
        model = SimpleNamespace(prepare_observer=lambda _: None, mentor=SimpleNamespace(classifier=old))
        calibration = SimpleNamespace(predict_proba=lambda x: np.array([[.6, .4]]))
        row = {'id': 'q', 'answer': ''}; tree = {'': {'answer': 'A'}}
        judge = {'answer': 'B', 'action': 'repair', 'parse_valid': True}
        for combined, expected in ((False, [new]), (True, [new, old])):
            observed = []
            with patch.object(runtime.original, 'record_observation', side_effect=lambda ledger, observer, *a: observed.append(observer)):
                runtime.ReviewCritic(new, calibration, np.array([.1, .2, .3, .5]), combined).observe(model, row, tree, judge, {})
            self.assertEqual([id(o) for o in observed], [id(o) for o in expected])


if __name__ == '__main__': unittest.main()
