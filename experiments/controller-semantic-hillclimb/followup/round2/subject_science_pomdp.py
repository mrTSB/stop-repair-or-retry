"""Route to one fitted POMDP/observer by the observed problem subject."""

from dataclasses import dataclass

import continuous_repairability as policy


@dataclass
class SubjectPlanner:
    models: dict
    settings: dict
    fit_receipt: dict


def evaluate(row, tree, judges, model, settings, family):
    if family != 'A':
        raise ValueError('This subject-observed POMDP acquires no LLM review')
    subject = row['subject']
    # The selected policy charges its one observer; no hidden second pass or
    # accuracy-dependent selection is performed by this subject dispatch.
    return policy.evaluate(row, tree, {}, model.models[subject], model.settings[subject], 'A')
