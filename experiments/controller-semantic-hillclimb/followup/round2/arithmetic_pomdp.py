"""Same finite-horizon POMDP with additional observable arithmetic consistency."""

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks  # Initialize the archived module search paths.
import dynamic_grounded_pomdp as core
from difficulty_pomdp import visible
from numeric_consistency import observations


@dataclass
class Observer(core.Observer):
    auxiliary: dict = field(default_factory=dict, repr=False)
    auxiliary_mean: object = None
    auxiliary_std: object = None

    def __getstate__(self):
        state = super().__getstate__()
        state['auxiliary'] = {}
        return state

    def values(self, row, path, tree, judge=None):
        result = super().values(row, path, tree, judge)
        if judge is None:
            self.auxiliary[row['id'], path] = [*visible(row, path, tree), *observations(tree[path]['text'])]
        return result

    def calibration_features(self, row, path, logits):
        original = super().calibration_features(row, path, logits)
        values = np.asarray(self.auxiliary[row['id'], path], dtype=float)
        normalized = np.clip((values - self.auxiliary_mean) / self.auxiliary_std, -8., 8.)
        return [*original, *normalized]


def fit(rows, trees, judges, directory, metadata, outputs, **kwargs):
    auxiliary = {}
    for item in metadata:
        if item['kind'] == 'solver':
            text = item['right'].partition('\nSolution:\n')[2]
            auxiliary[item['id'], item['path']] = [*item['numeric'], *observations(text)]
    matrix = np.asarray(list(auxiliary.values()), dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite observable feature')
    mean, std = matrix.mean(0), matrix.std(0).clip(min=.05)
    previous = core.Observer
    def create(directory, temperature):
        return Observer(directory, temperature, auxiliary=auxiliary,
                        auxiliary_mean=mean, auxiliary_std=std)
    try:
        # The original fitter uses the supplied observer for both calibration
        # and training observation distributions; neither step is skipped.
        core.Observer = create
        model = core.fit(rows, trees, judges, directory, metadata, outputs, **kwargs)
    finally:
        core.Observer = previous
    model.fit_receipt['arithmetic_consistency_observations'] = True
    model.fit_receipt['extra_observations'] = 'Visible response features and safely interpreted numeric equalities; no references or extra LLM calls.'
    return model


evaluate = core.evaluate
