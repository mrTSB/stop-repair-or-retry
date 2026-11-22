"""Route a retained math POMDP and registered expanded science variants."""
import copy
from dataclasses import dataclass

import continuous_completion_pomdp as continuous
import quantile_completion_pomdp as finite
import regularized_completion_pomdp as original


@dataclass
class FixedKernel:
    def signal(self, probability, valid):
        return original.core.observation(probability, valid)


def wrap(model):
    if hasattr(model, 'kernel'):
        return copy.deepcopy(model)
    return finite.Planner(copy.deepcopy(model.classifier), copy.deepcopy(model.forecaster),
        copy.deepcopy(model.dynamics), copy.deepcopy(model.fit_receipt), model.feature_family, FixedKernel())


def evaluate_science(row, tree, judges, model, settings, family):
    local = settings.get('by_subject', {}).get(row['subject'], settings)
    if local['posterior_weight'] == 0. and isinstance(model.kernel, FixedKernel):
        # Retain the exact established arithmetic for the original-partition control.
        return original.evaluate(row, tree, judges, model, local, family)
    return continuous.evaluate_science(row, tree, judges, model, local, family)


SubjectPlanner = continuous.SubjectPlanner


def evaluate(row, tree, judges, model, settings, family):
    if family != 'A': raise ValueError('Expanded A never queries an LLM reviewer')
    if row['subject'] == 'math':
        return continuous.evaluate(row, tree, judges, model, settings, family)
    return evaluate_science(row, tree, {}, model.science, model.science_settings, family)
