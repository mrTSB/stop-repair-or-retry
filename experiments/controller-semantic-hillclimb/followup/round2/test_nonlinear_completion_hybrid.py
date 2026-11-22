import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import nonlinear_completion_hybrid_v2 as hybrid
import test_completion_mentor as fixtures


class NonlinearHybridTests(unittest.TestCase):
    def test_explicit_regularized_schema_preserves_forest_and_truth(self):
        rows, complete, strong, judges = fixtures.fixture()
        records, _, receipt = fixtures.transfer.join(rows, complete, rows, strong, judges)
        matrices, receipt = fixtures.transfer.fit(records, receipt)
        observer = fixtures.FakeObserver(); forecaster = object()
        fit = {'schema': 'regularized-science-Complete16-v1', 'state_count': 16,
               'original_dynamics_formulas_preserved': True, 'maximum_additional_solver_calls': 2,
               'complete_is_terminal': True}
        science = hybrid.regularized.Planner(observer, forecaster,
            {s: SimpleNamespace() for s in hybrid.SCIENCE}, fit, 'raw20_plus_project256')
        mentor = hybrid.mentor_core.Planner(observer, {}, {'state_count': 8, 'maximum_logical_reviews': 1,
            'paired_root_questions': 320, 'fit_module': 'expanded_mentor_core'},
            SimpleNamespace(classifier=observer), review_observer=observer)
        parent = fixtures.hybrid.retained.Planner(SimpleNamespace(classifier=observer), mentor, {}, {})
        with patch.object(hybrid.assets, 'asset_id', return_value='same-recipe'):
            model = hybrid.build(science, {}, parent, 'hybrid_repairability', {}, matrices, receipt, {})
            self.assertIs(model.science.forecaster, forecaster); self.assertIs(model.mentor, mentor)
            self.assertEqual(model.fit_receipt['states_per_mode'], [16, 8])
            for matrix in model.transfer.values():
                np.testing.assert_allclose(matrix @ hybrid.mentor_core.REWARD, hybrid.plain.CURRENT)
            changed = copy.deepcopy(science); changed.fit_receipt['schema'] = 'unknown-state-meaning'
            with self.assertRaises(ValueError): hybrid.build(changed, {}, parent, 'hybrid_repairability', {}, matrices, receipt, {})


if __name__ == '__main__': unittest.main()
