"""Thirty-two hidden states: persistent difficulty plus four correctness bits.

Difficulty means fewer than three of the ten TRAIN solver endpoints are correct.
It is a supervised latent variable, never an observed inference-time feature.
The frozen observer, action protocol and two-continuation horizon are unchanged.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import completion_pomdp as previous

STATES = np.arange(32)
DIFFICULT = STATES >> 4
CURRENT = (STATES >> 3)&1
AFTER = np.asarray([(STATES >> 2)&1, (STATES >> 1)&1, STATES&1])
ACTIONS, N_OBS, normalize = previous.ACTIONS, previous.N_OBS, previous.normalize
PATHS = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry',
         'complete', 'repair/complete', 'retry/complete')


def labels(row, tree):
    truth = {p: previous.plain.correct(tree[p], row['answer']) for p in PATHS}
    difficult = int(sum(truth.values()) < 3)
    return {p: 16*difficult+previous.state(truth[p], *[truth[(p+'/' if p else '')+a] for a in ACTIONS[1:]])
        for p in ('', 'repair', 'retry')}


def mask(p, valid):
    p = np.asarray(p, dtype=float).copy()
    if p.shape != (32,) or not np.isfinite(p).all() or np.any(p < 0): raise ValueError('Thirty-two finite nonnegative state masses required')
    if not valid:
        p[CURRENT == 1] = 0.
        if not p.sum(): p[CURRENT == 0] = 1.
    return normalize(p)


def signal(p, valid):
    return previous.observation(mask(p, valid).reshape(2, 16).sum(0), valid)


@dataclass
class Forecaster:
    forest: object
    prior: np.ndarray

    def predict(self, x, valid):
        p = np.tile(.05*self.prior, (len(x), 1))
        p[:, self.forest.classes_.astype(int)] += .95*self.forest.predict_proba(x)
        return np.asarray([mask(v, ok) for v, ok in zip(p, valid)])


def fit_forecaster(x, y):
    counts = np.bincount(y, minlength=32)+1.
    forest = ExtraTreesClassifier(n_estimators=96, max_depth=8, min_samples_leaf=24,
        max_features=.7, random_state=2381124, n_jobs=1).fit(x, y)
    return Forecaster(forest, counts/counts.sum())


@dataclass
class Dynamics:
    prior: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    cost: np.ndarray

    def values(self, belief, remaining, penalty):
        if remaining not in (1, 2) or penalty < 0: raise ValueError('Two-call horizon and nonnegative cost required')
        belief = normalize(belief); result = [float(belief@CURRENT)]
        alpha = np.vstack([CURRENT, *[AFTER[a]-penalty*self.cost[1, a]/1000 for a in range(3)]])
        for a in range(3):
            price = penalty*float(belief@self.cost[2-remaining, a])/1000
            if remaining == 1 or a == 2: future = float(belief@AFTER[a])
            else:
                predicted = belief@self.transition[a]
                future = float((alpha@(predicted[:, None]*self.emission[a])).max(0).sum())
            result.append(future-price)
        return np.asarray(result)


@dataclass
class Planner:
    classifier: object
    forecaster: object
    dynamics: dict
    family: str
    fit_receipt: dict

    def predicted(self, row, path, tree):
        if row.get('answer') != '' or row['subject'] not in ('chemistry', 'physics') or path not in ('', 'repair', 'retry'):
            raise ValueError('Only blank-reference science decision prefixes')
        paid = {p: tree[p] for p in ('', path)}
        if self.family == 'raw20_control':
            x = previous.plain.observables(row, path, paid, self.classifier.values(row, path, paid))
        elif self.family == 'raw20_plus_project256': x = self.classifier.features(row, path, paid)
        else: raise ValueError('Unknown feature family')
        return self.forecaster.predict(np.asarray([x]), [bool(paid[path]['answer'])])[0]


def fit_dynamics(rows, trees, observer, records, oof, forecaster, family, receipt):
    keys = [(r['id'], p) for r in rows for p in ('', 'repair', 'retry')]
    if keys != [(r['id'], r['path']) for r in records] or len(set(keys)) != len(keys): raise ValueError('Aligned unique TRAIN inputs required')
    if np.asarray(oof).shape != (len(keys), 32) or not np.isfinite(oof).all(): raise ValueError('All excluded-question forecasts required')
    np.testing.assert_allclose(oof.sum(1), 1., atol=1e-10)
    states = {r['id']: labels(r, trees[r['id']]) for r in rows}
    signals = {key: signal(p, bool(trees[key[0]][key[1]]['answer'])) for key, p in zip(keys, oof)}
    support = np.asarray([[[float(CURRENT[t] == AFTER[a, s] and DIFFICULT[t] == DIFFICULT[s])
        for t in STATES] for s in STATES] for a in (0, 1)])
    emission_support = np.ones((2, 32, N_OBS)); emission_support[:, CURRENT == 1, 24:] = 0.
    encoder_cost = observer._recipe['parameters']/1e9*observer._recipe['max_length']/3.09

    def counts(group):
        prior = np.ones(32); transition = .1*support.copy(); emission = .1*emission_support.copy()
        prices, number = np.zeros((2, 3, 32)), np.zeros((2, 3, 32))
        for row in group:
            qid = row['id']; root = states[qid]['']; prior[root] += 1
            for a, path in enumerate(('repair', 'retry')):
                nxt = states[qid][path]
                if not support[a, root, nxt]: raise ValueError('Difficulty or current truth changed across a paired transition')
                transition[a, root, nxt] += 1; emission[a, nxt, signals[qid, path]] += 1
            for stage, prefixes in ((0, ('',)), (1, ('repair', 'retry'))):
                for path in prefixes:
                    state = states[qid][path]
                    for a, action in enumerate(ACTIONS[1:]):
                        node = trees[qid][(path+'/' if path else '')+action]
                        price = 4*(node['input_tokens']+node['output_tokens'])/3.09
                        if stage == 0 and action != 'complete': price += encoder_cost
                        prices[stage, a, state] += price; number[stage, a, state] += 1
        return prior, transition, emission, prices, number

    row_prob = lambda x: x/x.sum(-1, keepdims=True)
    pooled = counts(rows); dynamics = {}
    for subject in sorted({r['subject'] for r in rows}):
        local = counts([r for r in rows if r['subject'] == subject])
        prior = normalize(local[0]+32*normalize(pooled[0]))
        transition = row_prob(local[1]+32*row_prob(pooled[1]))
        emission = row_prob(local[2]+32*row_prob(pooled[2]))
        pooled_cost = np.divide(pooled[3], pooled[4], out=np.full_like(pooled[3], 650.), where=pooled[4] > 0)
        cost = (local[3]+4*pooled_cost)/(local[4]+4)
        dynamics[subject] = Dynamics(prior, transition, emission, cost)
    return Planner(observer, forecaster, dynamics, family, receipt)


def evaluate(row, tree, model, penalty=.02, weight=.5):
    if row['subject'] not in ('chemistry', 'physics') or weight not in (0., .5, 1.): raise ValueError('Registered science POMDP only')
    public = {**row, 'answer': ''}; d = model.dynamics[row['subject']]
    path, visited, decisions = '', [''], []; belief = model.predicted(public, '', tree)
    for remaining in (2, 1):
        a = int(d.values(belief, remaining, penalty).argmax())
        decisions.append({'path': path, 'belief': belief.tolist(), 'action': ACTIONS[a], 'queried': False, 'information_value': 0.})
        if a == 0: break
        path = (path+'/' if path else '')+ACTIONS[a]; visited.append(path)
        if path not in tree: raise previous.plain.MissingObservation('solver', path)
        if a == 3 or remaining == 1: break
        posterior = model.predicted(public, path, tree); valid = bool(tree[path]['answer'])
        finite = mask((belief@d.transition[a-1])*d.emission[a-1, :, signal(posterior, valid)], valid)
        belief = mask((1-weight)*finite+weight*posterior, valid)
    item = previous.plain.outcome(row, tree, {}, path, visited, [], decisions)
    paid = {v['path']: tree[v['path']] for v in decisions}
    tokens, calls, slots, parameters = model.classifier.charges(public, paid, {}, decisions, include_reviews=False)
    item.update(verifier_tokens=tokens, verifier_calls=calls, verifier_token_slots=slots,
        verifier_parameters_billions=parameters, completion_calls=int(path.endswith('complete')), completion_is_new_action=True)
    item['compute'] += parameters*slots/3.09; item['total_parameter_compute'] += parameters*slots/3.09
    return item
