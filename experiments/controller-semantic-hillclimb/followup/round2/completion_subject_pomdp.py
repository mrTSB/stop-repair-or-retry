"""Observed-subject routing between retained math A and science Complete A."""
from dataclasses import dataclass

import completion_pomdp
import continuous_repairability
import repairability_pomdp


@dataclass
class Planner:
    math: object
    science: object
    math_settings: dict
    science_settings: dict
    fit_receipt: dict


def evaluate(row, tree, judges, model, settings, family):
    if family != 'A':
        raise ValueError('This subject-routed POMDP acquires no LLM review')
    if row['subject'] == 'math':
        function = (continuous_repairability.evaluate if 'posterior_weight' in model.math_settings
                    else repairability_pomdp.evaluate)
        return function(row, tree, {}, model.math, model.math_settings, 'A')
    if row['subject'] in ('chemistry', 'physics'):
        return completion_pomdp.evaluate(row, tree, {}, model.science, model.science_settings, 'A')
    raise ValueError('Unsupported registered subject')
