"""A finite POMDP with a separately paid science reviewer-correctness sensor.

The post-guidance solver observer and math controller remain unchanged.
All additional science critic forwards enter the explicit inference ledger.
"""
from dataclasses import dataclass

import numpy as np

import stronger_base_hybrid as original
import science_review_critic as critic_training

action_core, mentor_core = original.action_core, original.mentor_core


def signal(judge, current, probability, edges):
    if not judge['answer'] or not judge['parse_valid']:
        return mentor_core.signal(judge, current)
    bucket = int(np.searchsorted(edges, probability))
    representative = (.1, .35, .65, .9, .99)[bucket]
    return mentor_core.signal(judge, current, np.array([1-representative, representative, 0., 0.]))


@dataclass
class ReviewCritic:
    observer: object
    calibrator: object
    edges: np.ndarray
    combined: bool

    def observe(self, model, row, tree, judge, ledger):
        model.prepare_observer(self.observer)
        new = self.observer.values(row, '', original.prefix(tree, ''), judge)[0]
        logits = [float(critic_training.margin(new)/self.observer.temperature)]
        original.record_observation(ledger, self.observer, row, '', tree, judge)
        if self.combined:
            old = model.mentor.classifier; model.prepare_observer(old)
            value = old.values(row, '', original.prefix(tree, ''), judge)[0]
            logits.insert(0, float(critic_training.margin(value/old.temperature)))
            original.record_observation(ledger, old, row, '', tree, judge)
        probability = float(self.calibrator.predict_proba(np.array([logits]))[0, 1])
        return signal(judge, tree[''], probability, self.edges)


@dataclass
class Planner(original.Planner):
    critic: object = None


def evaluate(row, tree, judges, model, settings, family):
    if family != 'C': raise ValueError('Paid LLM observations require hybrid C')
    if row['subject'] == 'math': return original.evaluate(row, tree, judges, model, settings, family)
    settings = settings.get('by_subject', {}).get(row['subject'], settings)
    penalty = settings['penalty']; safe = {**row, 'answer': ''}
    base, original_settings = original.resolve_base(model.base, model.base_module, model.base_settings, row['subject'])
    model.prepare_observer(base.classifier)
    root = base.predicted(safe, '', original.prefix(tree, '')); ledger = {}
    original.record_observation(ledger, base.classifier, safe, '', tree)
    belief = action_core.normalize(root @ model.transfer[row['subject']]); d = model.mentor.dynamics[row['subject']]
    information = float(d.information_value(belief, penalty)-base.dynamics[row['subject']].values(root, 2, penalty).max())
    if information <= settings.get('threshold', 0.)+1e-9:
        fixed = {**original_settings, 'penalty': penalty}
        function = original.continuous.evaluate if 'posterior_weight' in fixed else action_core.evaluate
        item = function(safe, tree, {}, base, fixed, 'A')
        item['correct'] = int(action_core.correct(tree[item['path']], row['answer']))
        item['decisions'][0]['information_value'] = information
        for decision in item['decisions']:
            original.record_observation(ledger, base.classifier, safe, decision['path'], tree)
        return original.bill(item, ledger, {}, row)
    if (row['id'], '') not in judges: raise action_core.MissingObservation('judge', '')
    judge = judges[row['id'], '']; observed = model.critic.observe(model, safe, tree, judge, ledger)
    belief = action_core.normalize(belief*d.review[:, observed]); action = int(d.values(belief, observed, penalty).argmax())
    decisions = [{'path': '', 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                  'queried': True, 'information_value': information, 'tracked_mentor': True}]
    path, visited = '', ['']
    if action:
        first = action-1; path = 'guided' if first == 0 else 'retry'
        if path not in tree: raise action_core.MissingObservation('solver', path)
        visited.append(path); model.prepare_observer(model.mentor.classifier)
        obs = model.mentor.regular.observed(safe, path, original.prefix(tree, path))
        original.record_observation(ledger, model.mentor.classifier, safe, path, tree)
        comparison = mentor_core.comparison(judge, tree[path]); likelihood = d.emission[first, :, obs*3+comparison].copy()
        if not tree[path]['answer']: likelihood[mentor_core.REWARD == 1] = 0.
        belief = action_core.normalize((belief @ d.reviewed_transition[observed, first])*likelihood)
        action = int((d.alpha(first, penalty) @ belief).argmax())
        decisions.append({'path': path, 'belief': belief.tolist(), 'action': ('stop', 'repair', 'retry')[action],
                          'queried': False, 'information_value': None, 'tracked_mentor': True})
        if action:
            path += '/'+('reuse' if action == 1 else 'retry')
            if path not in tree: raise action_core.MissingObservation('solver', path)
            visited.append(path)
    return original.bill(mentor_core.outcome(row, tree, judges, path, visited, [''], decisions), ledger, judges, row)
