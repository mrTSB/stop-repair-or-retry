"""POMDP + retained LLM review, with an explicit CPU metadata observation.

The regular solver/child neural observer is unchanged. No review neural pass
is made. This policy must never be dispatched through the semantic evaluator.
"""
from dataclasses import dataclass

import numpy as np

import stronger_base_hybrid as inherited
from stronger_base_hybrid import (action_core, continuous, mentor_core, resolve_base,
                                  prefix, record_observation, bill)
from paid_review_metadata import Adapter, signal as metadata_signal


@dataclass
class Planner(inherited.Planner):
    metadata_adapter: Adapter = None

    @property
    def review_observer(self):
        # Neural accounting explicitly excludes reviews. Metadata has its own
        # receipt and is not a neural observer with a fake directory/recipe.
        return None


def build(parent, adapter, base_settings, provenance):
    wrapped = inherited.build(parent.base, 'repairability_pomdp', base_settings, parent, provenance)
    receipt = {**wrapped.fit_receipt, 'policy_module': __name__,
               'review_observation': 'paid-metadata-logistic-v1',
               'metadata_adapter': adapter.receipt, 'include_reviews': False,
               'cpu_cost_accounting': 'Measured metadata wall time separately; unchanged parameter-token proxy excludes CPU planning/linear algebra'}
    parent.mentor.review_observer = None
    return Planner(wrapped.base, wrapped.mentor, wrapped.transfer, wrapped.base_module,
                   wrapped.base_settings, receipt, metadata_adapter=adapter)


def finish(item, ledger, judges, row, metadata_ledger):
    item = bill(item, ledger, judges, row)
    item['judge_physical_calls'] = sum(
        int(judges[row['id'], decision['path']].get('physical_api_calls')
            or len(judges[row['id'], decision['path']].get('calls', [])) or 1)
        for decision in item['decisions'] if decision['queried'])
    item.update(metadata_observations=metadata_ledger,
                metadata_calls=len(metadata_ledger),
                metadata_cpu_seconds=sum(value['cpu_seconds'] for value in metadata_ledger),
                review_neural_calls=0, include_reviews=False)
    return item


def evaluate(row, tree, judges, model, settings, family):
    if not isinstance(model, Planner) or model.fit_receipt.get('policy_module') != __name__:
        raise ValueError('Metadata hybrids require explicit metadata_review_hybrid dispatch')
    metadata_ledger = []
    if family != 'C':
        raise ValueError('This controller is a POMDP plus an optional paid LLM review')
    settings = settings.get('by_subject', {}).get(row['subject'], settings)
    penalty = settings['penalty']; safe = {**row, 'answer': ''}
    base, original = resolve_base(model.base, model.base_module, model.base_settings, row['subject'])
    model.prepare_observer(base.classifier)
    root = base.predicted(safe, '', prefix(tree, ''))
    ledger = {}
    record_observation(ledger, base.classifier, safe, '', tree)
    belief = action_core.normalize(root @ model.transfer[row['subject']])
    d = model.mentor.dynamics[row['subject']]
    information = float(d.information_value(belief, penalty) - base.dynamics[row['subject']].values(root, 2, penalty).max())
    if information <= settings.get('threshold', 0.) + 1e-9:
        fixed = {**original, 'penalty': penalty}
        function = continuous.evaluate if 'posterior_weight' in fixed else action_core.evaluate
        item = function(safe, tree, {}, base, fixed, 'A')
        item['correct'] = int(action_core.correct(tree[item['path']], row['answer']))
        item['decisions'][0]['information_value'] = information
        for decision in item['decisions']:
            record_observation(ledger, base.classifier, safe, decision['path'], tree)
        return finish(item, ledger, {}, row, metadata_ledger)
    if (row['id'], '') not in judges:
        raise action_core.MissingObservation('judge', '')
    judge = judges[row['id'], '']
    probability, receipt = model.metadata_adapter.predict_paid(safe, tree[''], judge)
    metadata_ledger.append(receipt)
    signal = metadata_signal(judge, tree[''], probability)
    belief = action_core.normalize(belief * d.review[:, signal])
    action = int(d.values(belief, signal, penalty).argmax())
    decisions = [{'path': '', 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                  'queried': True, 'information_value': information, 'tracked_mentor': True}]
    path, visited = '', ['']
    if action:
        first = action - 1; path = 'guided' if first == 0 else 'retry'
        if path not in tree:
            raise action_core.MissingObservation('solver', path)
        visited.append(path)
        model.prepare_observer(model.mentor.classifier)
        obs = model.mentor.regular.observed(safe, path, prefix(tree, path))
        record_observation(ledger, model.mentor.classifier, safe, path, tree)
        comparison = mentor_core.comparison(judge, tree[path])
        likelihood = d.emission[first, :, obs * 3 + comparison].copy()
        if not tree[path]['answer']:
            likelihood[mentor_core.REWARD == 1] = 0.
        belief = action_core.normalize((belief @ d.reviewed_transition[signal, first]) * likelihood)
        action = int((d.alpha(first, penalty) @ belief).argmax())
        decisions.append({'path': path, 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                          'queried': False, 'information_value': None, 'tracked_mentor': True})
        if action:
            path += '/' + ('reuse' if action == 1 else 'retry')
            if path not in tree:
                raise action_core.MissingObservation('solver', path)
            visited.append(path)
    item = mentor_core.outcome(row, tree, judges, path, visited, [''], decisions)
    return finish(item, ledger, judges, row, metadata_ledger)
