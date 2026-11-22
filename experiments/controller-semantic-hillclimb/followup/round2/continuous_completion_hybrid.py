"""Hybrid with TRAIN-clustered A16 and its paid-history posterior update.

Root Query, the truth-preserving16-to8 transfer, and all retained mentor-mode
updates are unchanged. The no-query branch delegates to continuous Complete A.
Every observed encoder forward and both physical reviewer phases stay billed.
"""
import numpy as np
import continuous_completion_pomdp as continuous
from completion_mentor_hybrid import (
    plain, assets, mentor_core, MATH_MODULES, SCIENCE, record_observation,
    physical_calls, bill, Planner)

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
        science_settings = model.science_settings.get('by_subject', {}).get(row['subject'], model.science_settings)
        item = continuous.evaluate_science(safe, tree, {}, model.science,
            {'penalty': penalty, 'posterior_weight': science_settings['posterior_weight']}, 'A')
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
