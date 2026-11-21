"""Cached forward passes preserve dtype and are bound to the frozen model recipe."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from semantic_study_common import models, sha


class NeuralReceipts(unittest.TestCase):
    def test_restored_logits_are_float32_and_recipe_bound(self):
        with TemporaryDirectory() as name:
            root = Path(name)
            (root / 'observer').mkdir()
            recipe = root / 'observer/recipe.json'
            recipe.write_text('{}')
            observer = SimpleNamespace(_cache={}, device='cpu')
            planner = SimpleNamespace(classifier=observer)
            frozen = {'models': {'A': {'observer_directory': 'observer'}}}
            key = ['solver', 'qid', '', 'prompt-sha', 'visible text']
            item = {'family': 'A', 'key': key, 'recipe_sha256': sha(recipe),
                    'state': [.1] * 4, 'plain': [.2] * 8, 'guided': [.3] * 8, 'tokens': 17}
            cache = root / 'test-neural-cache.jsonl'
            cache.write_text(json.dumps(item) + '\n')
            with patch('semantic_study_common.original_models', return_value={'A': planner}):
                actual = models(frozen, root)['A'].classifier._cache[tuple(key)]
                self.assertTrue(all(v.dtype == np.float32 for v in actual[:3]))
                self.assertEqual(actual[3], 17)
                recipe.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'Wrong cached observation'):
                    models(frozen, root)


if __name__ == '__main__':
    unittest.main()
