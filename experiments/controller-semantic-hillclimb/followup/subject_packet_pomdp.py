"""One-brief-review POMDP with penalties selected by the observed task subject."""

from packet_pomdp import fit, view, training_observations, evaluate as original_evaluate


def evaluate(row, tree, judges, model, settings, family):
    return original_evaluate(row, tree, judges, model, settings['by_subject'][row['subject']], family)
