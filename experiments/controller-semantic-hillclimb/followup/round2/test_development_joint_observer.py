import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import development_joint_observer as run
import test_joint_observer_pomdp as fixtures


class DevelopmentJointTests(unittest.TestCase):
    def test_training_gate_runs_before_development_loading(self):
        with patch.object(run, 'verified_fit', side_effect=ValueError('screen failed')), \
                patch.object(run.previous, 'inputs', side_effect=AssertionError('DEV read')) as loader:
            with self.assertRaisesRegex(ValueError, 'screen failed'):
                run.inputs(SimpleNamespace(models=Path('synthetic')))
            loader.assert_not_called()

    def test_cached_observation_binds_root_and_current_and_charges_no_encoder(self):
        with TemporaryDirectory() as tmp:
            observer, recipe = fixtures.JointRuntimeTests().setup_observer(Path(tmp))
            cached = run.CachedObserver(observer.directory, observer.input_builder, observer.recipe_sha256)
            row = {'id': 'q', 'question': 'question', 'answer': ''}
            paid = {'': {'text': 'root', 'answer': 'A'}, 'repair': {'text': 'child', 'answer': ''}}
            p = np.r_[np.ones(8)/8, np.zeros(8)]
            entry = cached.entry(row, 'repair', paid)
            cached.saved = {('q', 'repair'): {'entry_sha256': run.fixed.fingerprint(entry),
                'valid': False, 'probability': p.tolist(), 'token_slots': 9}}
            with patch.object(run.runtime.network, 'load', side_effect=AssertionError('Extra encoder pass')):
                values, slots = cached.values(row, 'repair', paid)
                np.testing.assert_array_equal(values, p); self.assertEqual(slots, 9)
                changed = copy.deepcopy(paid); changed['']['text'] = 'longer root'
                with self.assertRaises(ValueError): cached.values(row, 'repair', changed)
                changed = copy.deepcopy(paid); changed['repair']['answer'] = 'A'
                with self.assertRaises(ValueError): cached.values(row, 'repair', changed)
                with self.assertRaises(ValueError): cached.values(row, '', {'': paid['']})


if __name__ == '__main__': unittest.main()
