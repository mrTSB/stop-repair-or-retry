"""Science Complete-A16 plus the unchanged retained-mentor eight-state mode.

Root Query is optional. It changes the observed mode using a separately fitted
truth-preserving16x8 kernel. Complete is available only in the unqueried plain
mode and is terminal; no hidden bit or old eight-state transfer is repurposed.
This preparation module has no collector, development search or test runner.
"""
from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

import completion_pomdp as plain
import hybrid_repairability as retained
import stronger_base_hybrid as assets
import mentor_pomdp as mentor_core
import new_source_mentor_core as newsource_core
import expanded_mentor_core as expanded_core

SCIENCE = ('chemistry', 'physics')
SOURCE_BITS = ['current_correct', 'repair_correct', 'retry_correct', 'complete_correct']
TARGET_STATES = [f'H{h}/current{c}/mentor{m}' for h in (0, 1) for c in (0, 1) for m in (0, 1)]
MATH_MODULES = {'hybrid_repairability': retained, 'stronger_base_hybrid': assets}
PARENT_FITS = {'mentor_pomdp.fit': mentor_core.fit,
               'new_source_mentor_core.fit_new': newsource_core.fit_new,
               'expanded_mentor_core.fit_mentor': expanded_core.fit_mentor}


def parent_hardness_sources():
    return {name: {'function_sha256': hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
                   'source_sha256': hashlib.sha256(Path(inspect.getfile(function)).read_bytes()).hexdigest()}
            for name, function in PARENT_FITS.items()}


def parent_semantics(mentor, transfer_receipt):
    receipt = mentor.fit_receipt
    count = receipt.get('paired_root_questions')
    if receipt.get('fit_module') == 'expanded_mentor_core' and count == 320:
        function = 'expanded_mentor_core.fit_mentor'
    elif count in (128, 192) and receipt.get('hidden_states') == TARGET_STATES:
        function = 'new_source_mentor_core.fit_new' if count == 128 else 'mentor_pomdp.fit'
    else:
        raise ValueError('Unrecognized retained mentor hardness/state provenance')
    if transfer_receipt.get('hardness', {}).get('supported_parent_functions') != parent_hardness_sources():
        raise ValueError('Parent hardness implementation changed after transfer preparation')
    return {'parent_training_function':function, **parent_hardness_sources()[function],
            'parent_fit_receipt':receipt, 'target_states':TARGET_STATES,
            'hardness_uses_original_fixed_five_only':True}


def validate_transfer(transfer):
    if set(transfer) != set(SCIENCE):
        raise ValueError('Require both complete science transfer tables')
    support = plain.CURRENT[:, None] == mentor_core.REWARD[None, :]
    for matrix in transfer.values():
        matrix = np.asarray(matrix)
        if (matrix.shape != (16, 8) or not np.isfinite(matrix).all()
                or np.any(matrix < 0) or np.any(matrix[~support] != 0)
                or not np.allclose(matrix.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError('Transfer must preserve current correctness with exact structural zeros')
        np.testing.assert_allclose(matrix @ mentor_core.REWARD, plain.CURRENT, atol=1e-12, rtol=0)


@dataclass
class Planner:
    math_C: object
    math_policy_module: str
    math_settings: dict
    science: object
    science_settings: dict
    mentor: object
    transfer: dict
    fit_receipt: dict
    _asset_pool: dict = field(default_factory=dict, repr=False)

    def __getstate__(self):
        return {**vars(self), '_asset_pool': {}}

    @property
    def review_observer(self):
        return self.mentor.review_observer

    def prepare_observer(self, observer):
        # Shares neural assets only when recipe AND input adapter class agree.
        assets.Planner.prepare_observer(self, observer)


def build(science, science_settings, parent_C, parent_module, parent_settings,
          transfer, transfer_receipt, provenance):
    if parent_module not in MATH_MODULES or type(parent_C) is not MATH_MODULES[parent_module].Planner:
        raise ValueError('Only explicitly supported retained semantic/reported C parents are accepted')
    if (type(science) is not plain.Planner or science.fit_receipt.get('state_count') != 16
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
    receipt = {'schema': 'science-complete-retained-mentor-hybrid-v1',
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


def record_observation(ledger, observer, row, path, tree, judge=None):
    """Charge each actual class/recipe/input forward, including heterogeneous assets."""
    safe, paid = {**row, 'answer': ''}, assets.prefix(tree, path)
    for leaf in assets.components(observer):
        leaf.load()
        entry = leaf.entry(safe, path, paid, judge)
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        recipe = assets.asset_id(leaf)
        adapter = type(leaf).__module__ + '.' + type(leaf).__name__
        kind = 'review' if judge is not None else 'solver'
        key = adapter, recipe, row['id'], kind, path, digest
        values = leaf.values(safe, path, paid, judge)
        slots = int(values[3]); parameters = leaf._recipe['parameters'] / 1e9
        if slots <= 0 or not np.isfinite(parameters) or parameters <= 0:
            raise ValueError('Invalid physical encoder charge')
        charge = {'observer_class': adapter, 'recipe_sha256': recipe, 'kind': kind, 'path': path,
                  'input_sha256': digest, 'slots': slots, 'parameters_billions': parameters,
                  'compute': parameters * slots / 3.09}
        if key in ledger and ledger[key] != charge:
            raise ValueError('Repeated identical observation changed physical cost')
        ledger[key] = charge


def physical_calls(judge):
    calls = judge.get('calls')
    count = int(judge.get('physical_api_calls', len(calls) if calls is not None else 1))
    if count <= 0 or (calls is not None and count != len(calls)):
        raise ValueError('Invalid physical review phase count')
    if calls is not None:
        for field in ('input_tokens', 'output_tokens'):
            if any(int(call[field]) < 0 for call in calls) or sum(call[field] for call in calls) != judge[field]:
                raise ValueError('Review aggregate does not charge every physical phase')
    return count


def bill(item, ledger, judges, row):
    result = assets.bill(item, ledger, judges, row)
    result['judge_physical_calls'] = sum(physical_calls(judges[row['id'], d['path']])
        for d in item['decisions'] if d['queried'])
    result.update(completion_calls=int(item['path'].endswith('complete')),
                  completion_is_new_action=True, observed_mode_state_counts=[16, 8])
    return result


def evaluate(row, tree, judges, model, settings, family):
    if family != 'C':
        raise ValueError('This is the explicitly expanded POMDP+LLM hybrid')
    safe = {**row, 'answer': ''}
    if row['subject'] == 'math':
        module = MATH_MODULES.get(model.math_policy_module)
        if module is None: raise ValueError('Unsupported retained math C module')
        result = module.evaluate(safe, tree, judges, model.math_C, model.math_settings, 'C')
        result['correct'] = int(plain.plain.correct(tree[result['path']], row['answer']))
        result['judge_physical_calls'] = sum(physical_calls(judges[row['id'], d['path']])
            for d in result['decisions'] if d['queried'])
        result.update(completion_calls=0, math_C_route_retained=True)
        return result
    if row['subject'] not in SCIENCE: raise ValueError('Unknown subject')
    chosen = settings.get('by_subject', {}).get(row['subject'], settings)
    penalty, threshold = float(chosen['penalty']), float(chosen.get('threshold', 0.))
    if not np.isfinite([penalty, threshold]).all() or penalty < 0 or threshold < 0:
        raise ValueError('Require finite nonnegative utility and query threshold')
    model.prepare_observer(model.science.classifier)
    b_plain = model.science.predicted(safe, '', assets.prefix(tree, ''))
    ledger = {}
    record_observation(ledger, model.science.classifier, safe, '', tree)
    belief = plain.normalize(b_plain @ model.transfer[row['subject']])
    dynamics = model.mentor.dynamics[row['subject']]
    information = float(dynamics.information_value(belief, penalty)
                        - model.science.dynamics[row['subject']].values(b_plain, 2, penalty).max())
    if information <= threshold + 1e-9:
        # This delegates the actual Complete-A Bellman policy unchanged. Root
        # recomputation uses its same exact raw-input cache and is charged once.
        item = plain.evaluate(safe, tree, {}, model.science, {'penalty': penalty}, 'A')
        item['correct'] = int(plain.plain.correct(tree[item['path']], row['answer']))
        item['decisions'][0]['information_value'] = information
        for decision in item['decisions']:
            record_observation(ledger, model.science.classifier, safe, decision['path'], tree)
        return bill(item, ledger, {}, row)
    if (row['id'], '') not in judges: raise plain.plain.MissingObservation('judge', '')
    judge = judges[row['id'], '']
    physical_calls(judge)
    probability = None
    if model.review_observer is not None:
        model.prepare_observer(model.mentor.classifier)
        probability = mentor_core.perception.softmax(model.mentor.classifier.values(
            safe, '', assets.prefix(tree, ''), judge)[0], model.mentor.classifier.temperature)
        record_observation(ledger, model.mentor.classifier, safe, '', tree, judge)
    signal = mentor_core.signal(judge, tree[''], probability)
    belief = plain.normalize(belief * dynamics.review[:, signal])
    action = int(dynamics.values(belief, signal, penalty).argmax())
    decisions = [{'path': '', 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                  'queried': True, 'information_value': information, 'tracked_mentor': True}]
    path, visited = '', ['']
    if action:
        first = action - 1; path = 'guided' if first == 0 else 'retry'
        if path not in tree: raise plain.plain.MissingObservation('solver', path)
        visited.append(path)
        model.prepare_observer(model.mentor.classifier)
        observation = model.mentor.regular.observed(safe, path, assets.prefix(tree, path))
        record_observation(ledger, model.mentor.classifier, safe, path, tree)
        comparison = mentor_core.comparison(judge, tree[path])
        likelihood = dynamics.emission[first, :, observation * 3 + comparison].copy()
        if not tree[path]['answer']: likelihood[mentor_core.REWARD == 1] = 0.
        belief = plain.normalize((belief @ dynamics.reviewed_transition[signal, first]) * likelihood)
        action = int((dynamics.alpha(first, penalty) @ belief).argmax())
        decisions.append({'path': path, 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                          'queried': False, 'information_value': None, 'tracked_mentor': True})
        if action:
            path += '/' + ('reuse' if action == 1 else 'retry')
            if path not in tree: raise plain.plain.MissingObservation('solver', path)
            visited.append(path)
    item = mentor_core.outcome(row, tree, judges, path, visited, [''], decisions)
    return bill(item, ledger, judges, row)
