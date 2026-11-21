"""Hybrid that extends the repairability POMDP with a paid retained-review mode.

The observed query mode selects one of two eight-state belief spaces. A learned
truth-preserving transfer kernel maps A's hidden action outcomes into the
mentor POMDP's difficulty/current/mentor state before buying its observation.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks
from difficulty_pomdp import FIXED, MissingObservation
from scoring import correct
from context_pomdp import outcome
import mentor_pomdp as mentor_core
import repairability_pomdp as action_core


@dataclass
class Planner:
    base: object
    mentor: object
    transfer: dict
    fit_receipt: dict

    @property
    def classifier(self):
        return self.base.classifier

    @property
    def review_observer(self):
        return self.mentor.review_observer

    def prepare(self):
        a, m = self.base.classifier, self.mentor.classifier
        if a._model is not None and m._model is a._model:
            return
        recipe_a = (Path(a.directory) / 'recipe.json').read_bytes()
        if hashlib.sha256(recipe_a).hexdigest() != self.fit_receipt['observer_recipe_sha256']:
            raise ValueError('Changed hybrid observer assets')
        # Both fitting recipes were checked in build(). Relocation after a
        # freeze can now point both observers at the same archived checkpoint.
        m.directory = a.directory
        a.load()
        for attr in ('_model', '_tokenizer', '_recipe', '_cache', 'device', '_retriever', '_backgrounds'):
            setattr(m, attr, getattr(a, attr))


def fit_transfer(rows, trees, judges, smoothing=16.):
    paired = [r for r in rows if (r['id'], '') in judges]
    if len(paired) != 192:
        raise ValueError('Use all192registered mixed mentor-training questions')
    support = np.asarray([[float(action_core.CURRENT[s] == mentor_core.REWARD[t])
                           for t in range(8)] for s in range(8)])
    records = []
    for row in paired:
        qid = row['id']; tree = trees[qid]
        c = int(correct(tree[''], row['answer']))
        source = action_core.state(c, correct(tree['repair'], row['answer']), correct(tree['retry'], row['answer']))
        hard = int(np.mean([correct(tree[p], row['answer']) for p in FIXED]) < .6)
        target = mentor_core.state(2 * hard + c, correct(judges[qid, ''], row['answer']))
        records.append((row['subject'], source, target))
    global_counts = .1 * support.copy()
    for _, source, target in records:
        global_counts[source, target] += 1
    pooled = global_counts / global_counts.sum(1, keepdims=True)
    result = {}
    for subject in sorted({r['subject'] for r in rows}):
        counts = .1 * support.copy()
        for sub, source, target in records:
            if sub == subject:
                counts[source, target] += 1
        counts += smoothing * pooled
        result[subject] = counts / counts.sum(1, keepdims=True)
        np.testing.assert_allclose(result[subject] @ mentor_core.REWARD, action_core.CURRENT, atol=1e-12)
    return result


def build(base, mentor, rows, trees, judges, smoothing=16.):
    recipe_a = hashlib.sha256((Path(base.classifier.directory) / 'recipe.json').read_bytes()).hexdigest()
    recipe_m = hashlib.sha256((Path(mentor.classifier.directory) / 'recipe.json').read_bytes()).hexdigest()
    if recipe_a != recipe_m:
        raise ValueError('The two hybrid modes must use exactly the same neural weights and input profile')
    return Planner(base, mentor, fit_transfer(rows, trees, judges, smoothing),
                   {'state_count': 16, 'observed_modes': ['unqueried_action_fixability', 'queried_retained_mentor'],
                    'states_per_observed_mode': 8, 'base_A': base.fit_receipt, 'mentor': mentor.fit_receipt,
                    'transfer_smoothing': smoothing, 'maximum_reviews': 1,
                    'observer_recipe_sha256': recipe_a,
                    'solver_budget': 'At most two additional compact calls',
                    'shared_neural_passes_require_identical_assets': True})


def evaluate(row, tree, judges, model, settings, family):
    if family != 'C':
        raise ValueError('This is the POMDP+LLM hybrid')
    settings = settings.get('by_subject', {}).get(row['subject'], settings)
    model.prepare()
    penalty = settings['penalty']
    b_action = model.base.predicted(row, '', tree)
    belief = action_core.normalize(b_action @ model.transfer[row['subject']])
    d = model.mentor.dynamics[row['subject']]
    plain = model.base.dynamics[row['subject']].values(b_action, 2, penalty)
    information = float(d.information_value(belief, penalty) - plain.max())
    if information <= settings.get('threshold', 0.) + 1e-9:
        item = action_core.evaluate(row, tree, {}, model.base, {'penalty': penalty}, 'A')
        item['decisions'][0]['information_value'] = information
        return item
    if (row['id'], '') not in judges:
        raise MissingObservation('judge', '')
    judge = judges[row['id'], '']
    p = None
    if model.review_observer is not None:
        p = mentor_core.perception.softmax(model.mentor.classifier.values(row, '', tree, judge)[0],
                                            model.mentor.classifier.temperature)
    signal = mentor_core.signal(judge, tree[''], p)
    belief = action_core.normalize(belief * d.review[:, signal])
    action = int(d.values(belief, signal, penalty).argmax())
    decisions = [{'path': '', 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                  'queried': True, 'information_value': information, 'tracked_mentor': True}]
    path, visited = '', ['']
    if action:
        first = action - 1
        path = 'guided' if first == 0 else 'retry'
        if path not in tree:
            raise MissingObservation('solver', path)
        visited.append(path)
        obs = model.mentor.regular.observed(row, path, tree)
        equal = mentor_core.comparison(judge, tree[path])
        likelihood = d.emission[first, :, obs * 3 + equal].copy()
        if not tree[path]['answer']:
            likelihood[mentor_core.REWARD == 1] = 0.
        belief = action_core.normalize((belief @ d.reviewed_transition[signal, first]) * likelihood)
        action = int((d.alpha(first, penalty) @ belief).argmax())
        decisions.append({'path': path, 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                          'queried': False, 'information_value': None, 'tracked_mentor': True})
        if action:
            path += '/' + ('reuse' if action == 1 else 'retry')
            if path not in tree:
                raise MissingObservation('solver', path)
            visited.append(path)
    item = outcome(row, tree, judges, path, visited, [''], decisions)
    tokens, calls, slots, parameters = model.classifier.charges(
        row, tree, judges, decisions, include_reviews=model.review_observer is not None)
    extra = parameters * slots / 3.09
    item.update(verifier_tokens=tokens, verifier_calls=calls, verifier_token_slots=slots,
                verifier_parameters_billions=parameters)
    item['compute'] += extra
    item['total_parameter_compute'] += extra
    return item
