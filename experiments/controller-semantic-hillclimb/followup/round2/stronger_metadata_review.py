"""Replace only metadata C's unqueried A, retaining the fitted review channel.

The runtime remains explicitly metadata_review_hybrid. This constructor does
not refit any dynamics, metadata coefficients, calibration, or transfer matrix.
"""
from types import SimpleNamespace

import fit_metadata_review as fitting
import metadata_review_hybrid as runtime
import stronger_base_hybrid as inherited
from paid_review_metadata import digest


def retained_state(model):
    """Value digest stable across pickle reloads and independent of runtime caches."""
    protected = fitting.preserved_digest(SimpleNamespace(
        base=None, mentor=model.mentor, transfer=model.transfer))
    review = {subject: {name: getattr(dynamics, name).tolist()
                       for name in ('review', 'reviewed_transition', 'review_costs')}
              for subject, dynamics in sorted(model.mentor.dynamics.items())}
    return digest({'protected': protected, 'review': review,
                   'mentor_receipt': model.mentor.fit_receipt,
                   'metadata_state': model.metadata_adapter.state(),
                   'metadata_receipt': model.metadata_adapter.receipt})


def build(base, base_module, base_settings, parent, provenance):
    if (not isinstance(parent, runtime.Planner)
            or parent.fit_receipt.get('policy_module') != 'metadata_review_hybrid'
            or parent.review_observer is not None or parent.mentor.review_observer is not None):
        raise ValueError('Require an explicit fitted metadata-review parent without review NN')
    parent.metadata_adapter.validate()
    before = retained_state(parent)
    wrapped = inherited.build(base, base_module, base_settings, parent, provenance)
    receipt = {**parent.fit_receipt, **wrapped.fit_receipt,
               'builder_module': __name__, 'policy_module': 'metadata_review_hybrid',
               'metadata_adapter': parent.metadata_adapter.receipt,
               'include_reviews': False, 'parent_retained_state_sha256': before,
               'mentor_refitted': False, 'transfer_refitted': False,
               'metadata_adapter_refitted': False}
    result = runtime.Planner(wrapped.base, wrapped.mentor, wrapped.transfer,
        wrapped.base_module, wrapped.base_settings, receipt,
        metadata_adapter=parent.metadata_adapter)
    if retained_state(result) != before or retained_state(parent) != before:
        raise ValueError('Construction changed the retained metadata/mentor components')
    return result
