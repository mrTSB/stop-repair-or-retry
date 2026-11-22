"""Explicit Complete16/mentor8 hybrid using the independently trained forest A.

Reuses the audited finite-horizon evaluation and full observation ledger.
Only the accepted science class changes from the original preparation; the
sixteen-state source meaning and truth-preserving transfer are unchanged.
"""
import numpy as np
import regularized_completion_pomdp as regularized
from completion_mentor_hybrid import (
    Planner, evaluate, SCIENCE, SOURCE_BITS, TARGET_STATES, MATH_MODULES,
    plain, mentor_core, assets, validate_transfer, parent_semantics)

def build(science, science_settings, parent_C, parent_module, parent_settings,
          transfer, transfer_receipt, provenance):
    if parent_module not in MATH_MODULES or type(parent_C) is not MATH_MODULES[parent_module].Planner:
        raise ValueError('Only explicitly supported retained semantic/reported C parents are accepted')
    if (type(science) is not regularized.Planner or science.fit_receipt.get('state_count') != 16
            or science.fit_receipt.get('state_bits') != SOURCE_BITS
            or science.fit_receipt.get('complete_is_terminal') is not True
            or set(science.dynamics) != set(SCIENCE)):
        raise ValueError('Require the independently fitted science Complete-A16')
    mentor = parent_C.mentor
    if (type(mentor) is not mentor_core.Planner or mentor.fit_receipt.get('state_count') != 8
            or mentor.fit_receipt.get('maximum_logical_reviews') != 1
            or (mentor.review_observer is not None and mentor.review_observer is not mentor.classifier)
            or mentor.regular.classifier is not mentor.classifier):
        raise ValueError('Require unchanged H/current/mentor state meaning and ordinary review observer')
    if (transfer_receipt.get('smoothing') != 32. or transfer_receipt.get('paired_questions') != 256
            or transfer_receipt.get('source_bits') != SOURCE_BITS
            or transfer_receipt.get('target_states') != TARGET_STATES
            or transfer_receipt.get('hardness', {}).get('fixed_paths') != list(mentor_core.core.FIXED)):
        raise ValueError('Require the registered256-science transfer and exact parent hardness definition')
    validate_transfer(transfer)
    semantics = parent_semantics(mentor, transfer_receipt)
    observers = [*assets.components(science.classifier), *assets.components(mentor.classifier)]
    receipt = {'schema': 'nonlinear-complete-retained-mentor-hybrid-v1',
        'observed_modes': ['unqueried_complete_A16', 'queried_retained_mentor8'],
        'states_per_mode': [16, 8], 'state_count': 24, 'source_bits': SOURCE_BITS,
        'target_states': TARGET_STATES, 'maximum_reviews': 1, 'query_availability': 'Root only',
        'complete_availability': 'Unqueried root/ordinary repair/ordinary retry; always terminal',
        'maximum_additional_solver_calls': 2, 'mentor_dynamics_unchanged': True,
        'math_route_unchanged': True, 'math_policy_module': parent_module,
        'observer_recipe_sha256': sorted({assets.asset_id(o) for o in observers}),
        'science_A': science.fit_receipt, 'mentor': mentor.fit_receipt, 'parent_semantics':semantics,
        'transfer': transfer_receipt, 'provenance': provenance,
        'new_matched_B_required_before_development_claim': True,
        'prospective_support': 'Unsupported by old freeze/collector; explicit extension required'}
    return Planner(parent_C, parent_module, parent_settings, science, science_settings,
                   mentor, transfer, receipt)

