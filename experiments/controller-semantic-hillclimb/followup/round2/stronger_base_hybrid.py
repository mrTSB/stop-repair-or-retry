"""Retained-review hybrid with a separately selected, potentially richer A.

Both unqueried and queried modes retain the original eight-state meanings.
Observer assets may differ. Every distinct model/input pass is billed, and
weights or caches are shared only for matching recipe hashes and input classes.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

import repairability_pomdp as action_core
import continuous_repairability as continuous
import mentor_pomdp as mentor_core


def components(observer):
    children = getattr(observer, 'components', None)
    return [leaf for child in children for leaf in components(child)] if children is not None else [observer]


def asset_id(observer):
    return hashlib.sha256((Path(observer.directory) / 'recipe.json').read_bytes()).hexdigest()


def prefix(tree, path):
    paths = ['']
    if path:
        parts = path.split('/')
        paths += ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
    return {p: tree[p] for p in paths}


def resolve_base(base, module, settings, subject):
    if hasattr(base, 'models'):
        if module not in ('subject_science_pomdp', 'science_fusion_pomdp'):
            raise ValueError('Subject routing requires its declared POMDP module')
        child, chosen = base.models[subject], base.settings[subject]
    else:
        if module not in ('repairability_pomdp', 'continuous_repairability'):
            raise ValueError('Unsupported plain POMDP module')
        child, chosen = base, settings.get('by_subject', {}).get(subject, settings)
    if child.fit_receipt.get('state_count') != 8 or child.fit_receipt.get('state_bits') != [
            'current_correct', 'repair_correct', 'retry_correct']:
        raise ValueError('Replacement A must retain the C/repair/retry latent-state meaning')
    if subject not in child.dynamics or getattr(child, 'review_observer', None) is not None:
        raise ValueError('Replacement A must be a plain subject-compatible POMDP')
    return child, chosen


@dataclass
class Planner:
    base: object
    mentor: object
    transfer: dict
    base_module: str
    base_settings: dict
    fit_receipt: dict
    _asset_pool: dict = field(default_factory=dict, repr=False)

    def __getstate__(self):
        return {**vars(self), '_asset_pool': {}}

    @property
    def review_observer(self):
        return self.mentor.review_observer

    def prepare_observer(self, observer):
        for leaf in components(observer):
            key = asset_id(leaf)
            if key not in self.fit_receipt['observer_recipe_sha256']:
                raise ValueError('Changed hybrid observer assets')
            # A class-specific cache may contain transformed raw logits. Match
            # the input/output adapter as well as the complete weight recipe.
            share_key = key, type(leaf).__module__, type(leaf).__name__
            source = self._asset_pool.get(share_key)
            if source is not None and source is not leaf:
                source.load()
                for name in ('_model', '_tokenizer', '_recipe', '_cache', '_retriever', '_backgrounds'):
                    if hasattr(source, name):
                        setattr(leaf, name, getattr(source, name))
                leaf.device = source.device
            else:
                leaf.load()
                self._asset_pool[share_key] = leaf
        observer.load()


def build(base, base_module, base_settings, parent, provenance):
    observers = []
    for subject in ('math', 'chemistry', 'physics'):
        child, _ = resolve_base(base, base_module, base_settings, subject)
        observers += components(child.classifier)
        transfer = parent.transfer[subject]
        np.testing.assert_allclose(transfer @ mentor_core.REWARD, action_core.CURRENT, atol=1e-12)
    observers += components(parent.mentor.classifier)
    receipt = {'state_count': 16, 'states_per_observed_mode': 8, 'maximum_reviews': 1,
               'observed_modes': ['unqueried_action_fixability', 'queried_retained_mentor'],
               'base_module': base_module, 'selected_base_settings': base_settings,
               'mentor': parent.mentor.fit_receipt, 'parent_transfer_unchanged': True,
               'observer_recipe_sha256': sorted({asset_id(o) for o in observers}),
               'solver_budget': 'At most two additional compact solver calls',
               'encoder_billing': 'Every distinct recipe/input pass; heterogeneous component parameters charged separately',
               'provenance': provenance}
    return Planner(base, parent.mentor, parent.transfer, base_module, base_settings, receipt)


def record_observation(ledger, observer, row, path, tree, judge=None):
    safe, paid = {**row, 'answer': ''}, prefix(tree, path)
    for leaf in components(observer):
        leaf.load()
        entry = leaf.entry(safe, path, paid, judge)
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        recipe = asset_id(leaf); kind = 'review' if judge is not None else 'solver'
        key = recipe, row['id'], kind, path, digest
        values = leaf.values(safe, path, paid, judge)
        slots, parameters = int(values[3]), leaf._recipe['parameters'] / 1e9
        charge = {'recipe_sha256': recipe, 'kind': kind, 'path': path, 'input_sha256': digest,
                  'slots': slots, 'parameters_billions': parameters,
                  'compute': parameters * slots / 3.09}
        if key in ledger and ledger[key] != charge:
            raise ValueError('Repeated observer input changed its measured cost')
        ledger[key] = charge


def bill(item, ledger, judges, row):
    charges = list(ledger.values())
    slots = sum(r['slots'] for r in charges)
    work = sum(r['parameters_billions'] * r['slots'] for r in charges)
    item.update(verifier_components=charges, verifier_tokens=slots,
                verifier_token_slots=slots, verifier_calls=len(charges),
                verifier_parameters_billions=work / slots if slots else 0.,
                verifier_parameter_accounting='Slot-weighted aggregate; exact components in verifier_components')
    item['compute'] = (4 * item['solver_tokens'] + 3 * item['judge_tokens'] + work) / 3.09
    item['total_parameter_compute'] = (4 * item['solver_tokens'] + 35 * item['judge_tokens'] + work) / 3.09
    item['judge_physical_calls'] = sum(judges[row['id'], d['path']].get('physical_api_calls', 1)
                                       for d in item['decisions'] if d['queried'])
    return item


def evaluate(row, tree, judges, model, settings, family):
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
        return bill(item, ledger, {}, row)
    if (row['id'], '') not in judges:
        raise action_core.MissingObservation('judge', '')
    judge = judges[row['id'], '']
    probability = None
    if model.review_observer is not None:
        model.prepare_observer(model.mentor.classifier)
        probability = mentor_core.perception.softmax(model.mentor.classifier.values(safe, '', prefix(tree, ''), judge)[0],
                                                      model.mentor.classifier.temperature)
        record_observation(ledger, model.mentor.classifier, safe, '', tree, judge)
    signal = mentor_core.signal(judge, tree[''], probability)
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
    return bill(item, ledger, judges, row)
