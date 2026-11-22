"""One-forward, equal-weight joint-belief ensemble with exact POMDP dynamics."""
from dataclasses import dataclass

import numpy as np

import regularized_completion_pomdp as previous


def combine(forecasters, features, valid):
    if len(forecasters) != 2: raise ValueError('Exactly the two frozen forecasters are required')
    # Condition each component before mixing, matching its held-question TRAIN
    # forecasts even when the paid response has no valid answer.
    return np.mean([previous.core.mask_invalid(previous.core.probabilities(f, features), valid)
                    for f in forecasters], axis=0)


@dataclass
class Planner(previous.Planner):
    def predicted(self, row, path, tree):
        if row['subject'] not in ('chemistry', 'physics') or path not in ('', 'repair', 'retry'):
            raise ValueError('Only paid nonterminal science prefixes may be observed')
        if self.feature_family != 'raw20_plus_project256': raise ValueError('Fixed dense feature family required')
        safe = {**row, 'answer': ''}; paid = {p: tree[p] for p in ('', path)}
        features = self.classifier.features(safe, path, paid)
        return combine(self.forecaster, features, bool(paid[path]['answer']))
