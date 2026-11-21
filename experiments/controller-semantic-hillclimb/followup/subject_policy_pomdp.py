"""The same finite POMDP with development-selected penalties by observed subject.

Subject is available before inference and already indexes the learned dynamics.
No question-specific setting or test outcome is used to select these penalties.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamic_grounded_pomdp import fit, evaluate as original_evaluate


def evaluate(row, tree, judges, model, settings, family):
    return original_evaluate(row, tree, judges, model, settings['by_subject'][row['subject']], family)
