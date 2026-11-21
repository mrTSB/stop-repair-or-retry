"""Repairability POMDP with a discriminative belief over the paid history.

Future values retain the fitted finite-observation Bellman backup. At an actual
child observation, a fixed development-selected blend combines that finite
filter with the calibrated eight-state posterior. This is an approximate
belief estimator, not an exact continuous-likelihood Bayesian filter: the
forecaster already observes features from the paid root and current answer.
"""

import numpy as np
import repairability_pomdp as core


def update(model, row, path, tree, prior, action, weight):
    if not 0. <= weight <= 1.:
        raise ValueError('Posterior blend must be in [0,1]')
    d = model.dynamics[row['subject']]
    posterior = model.predicted(row, path, tree)
    obs = core.observation(posterior, bool(tree[path]['answer']))
    finite = core.normalize((prior @ d.transition[action]) * d.emission[action, :, obs])
    belief = core.normalize((1. - weight) * finite + weight * posterior)
    if not tree[path]['answer']:
        belief[core.CURRENT == 1] = 0.
        belief = core.normalize(belief)
    return belief


def evaluate(row, tree, judges, model, settings, family):
    if family != 'A':
        raise ValueError('This controller uses no LLM reviews')
    settings = settings.get('by_subject', {}).get(row['subject'], settings)
    d = model.dynamics[row['subject']]
    penalty, weight = settings['penalty'], settings['posterior_weight']
    path, visited, decisions = '', [''], []
    belief = model.predicted(row, '', tree)
    for remaining in (2, 1):
        action = int(d.values(belief, remaining, penalty).argmax())
        decisions.append({'path': path, 'belief': belief.tolist(),
                          'action': ('stop', 'repair', 'retry')[action],
                          'queried': False, 'information_value': 0.})
        if not action:
            break
        edge = ('repair', 'retry')[action - 1]
        path = (path + '/' if path else '') + edge
        if path not in tree:
            raise core.MissingObservation('solver', path)
        visited.append(path)
        if remaining == 2:
            belief = update(model, row, path, tree, belief, action - 1, weight)
    item = core.outcome(row, tree, {}, path, visited, [], decisions)
    tokens, calls, slots, parameters = model.classifier.charges(row, tree, {}, decisions, include_reviews=False)
    extra = parameters * slots / 3.09
    item.update(verifier_tokens=tokens, verifier_calls=calls, verifier_token_slots=slots,
                verifier_parameters_billions=parameters)
    item['compute'] += extra
    item['total_parameter_compute'] += extra
    return item
