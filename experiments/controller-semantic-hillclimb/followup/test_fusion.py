"""Check physical ensemble accounting and the aligned fusion representation."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_pomdp import FusionNetwork, Observer


class Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, **kw):
        n = len(kw['input_ids'])
        state = (torch.arange(4) + self.value).float().repeat(n, 1)
        plain = torch.full((n, 8), float(self.value))
        return state, plain, 2 * plain


class FusionTests(unittest.TestCase):
    def test_scaled_component_logits_remain_aligned(self):
        model = FusionNetwork([Constant(10), Constant(20)], [2., 4.])
        state, plain, guided = model(input_ids=torch.ones((2, 4), dtype=torch.long))
        expected = torch.tensor([5., 5.5, 6., 6.5, 5., 5.25, 5.5, 5.75]).repeat(2, 1)
        torch.testing.assert_close(state, expected)
        torch.testing.assert_close(plain, torch.full((2, 8), 15.))
        torch.testing.assert_close(guided, torch.full((2, 8), 30.))

    def test_both_physical_forwards_are_charged_exactly_once(self):
        component_parameters = (.184, .434)
        total = sum(component_parameters)
        observer = Observer('unused', 1., _model=nn.Identity(),
                            _recipe={'parameters': round(total * 1e9), 'component_count': 2})
        row = {'id': 'fixture'}
        node = {'prompt_sha256': 'prompt', 'text': 'response'}
        observer._cache[observer.key(row, '', node, 'solver')] = (np.zeros(8), np.zeros(8), np.zeros(8), 13)
        tokens, calls, slots, average_price = observer.charges(
            row, {'': node}, {}, [{'path': '', 'queried': False}], include_reviews=False)
        self.assertEqual((tokens, calls, slots), (26, 2, 26))
        self.assertAlmostEqual(average_price * slots, sum(p * 13 for p in component_parameters))


if __name__ == '__main__':
    unittest.main()
