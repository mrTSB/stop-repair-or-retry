"""Observed-subject composition of genuine frozen POMDP+LLM experts.

Only the subject chooses an expert. That expert performs its original belief
updates and Bellman decisions, with its original solver/reviewer/encoder bills.
Unselected experts cannot produce observations or contribute free evidence.
"""
from dataclasses import dataclass

import development_expanded_mentor as runtime
import matched_review_b as llm

SUBJECTS = ('math','chemistry','physics')
MODULES = {'stronger_base_hybrid','metadata_review_hybrid'}


@dataclass
class Planner:
    models: dict
    experts: dict
    fit_receipt: dict


def evaluate(row, tree, judges, model, settings, family):
    if family != 'C' or row['subject'] not in SUBJECTS or settings:
        raise ValueError('Require the frozen subject-routed hybrid configuration')
    expert = model.experts[row['subject']]
    if expert['policy_module'] not in MODULES:
        raise ValueError('Each expert must be an existing POMDP+LLM controller')
    item = runtime.dispatch(model.models[expert['model']],expert['policy_module'],row,tree,judges,expert['settings'])
    return {**item,'subject_expert_model':expert['model'],'subject_expert_candidate':expert['original_candidate_index']}


def evaluate_B(row, tree, judges, settings):
    if row['subject'] not in SUBJECTS or set(settings.get('by_subject',{})) != set(SUBJECTS):
        raise ValueError('Require all three frozen LLM-only subject rules')
    return llm.policy(row,tree,judges,settings['by_subject'][row['subject']])
