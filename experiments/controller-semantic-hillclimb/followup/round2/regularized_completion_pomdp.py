"""Complete-action POMDP with nested-fit forecasts and optional dense observations.

All transition, support, cost and finite Bellman formulas match Complete16.
The new input procedure is explicit; archived older policies are unchanged.
"""
from dataclasses import dataclass, field
import numpy as np

import completion_pomdp as core
import dense_completion_probe as dense
import dynamic_grounded_pomdp as perception


@dataclass
class DenseObserver(perception.Observer):
    _dense_cache: dict = field(default_factory=dict, repr=False)

    def __getstate__(self):
        return {**super().__getstate__(), '_dense_cache': {}}

    def values(self, row, path, tree, judge=None):
        if judge is not None:
            raise ValueError('This A observer never reads a reviewer')
        self.load()
        public = {**row, 'answer': ''}
        paid = {p: tree[p] for p in ('', path)}
        key = self.key(public, path, paid[path], 'solver')
        if key not in self._dense_cache:
            tensors = perception.trim_tensors(perception.encode(self._tokenizer,
                [self.entry(public, path, paid)], self._recipe['max_length']))
            tensors = {k: v.to(self.device) for k, v in tensors.items()}
            result = dense.capture(self._model, tensors)
            self._cache[key] = (*[result[n][0] for n in ('state', 'plain', 'guided')], int(result['token_slots'][0]))
            self._dense_cache[key] = result['joint256'][0]
        return self._cache[key]

    def features(self, row, path, tree):
        public = {**row, 'answer': ''}; paid = {p: tree[p] for p in ('', path)}
        values = self.values(public, path, paid)
        key = self.key(public, path, paid[path], 'solver')
        return np.r_[core.plain.observables(public, path, paid, values), self._dense_cache[key]]


@dataclass
class Planner(core.Planner):
    feature_family: str = 'raw20_control'

    def predicted(self, row, path, tree):
        if row['subject'] not in ('chemistry', 'physics') or path not in ('', 'repair', 'retry'):
            raise ValueError('Only nonterminal science prefixes are observable')
        public = {**row, 'answer': ''}; paid = {p: tree[p] for p in ('', path)}
        if self.feature_family == 'raw20_plus_project256':
            features = self.classifier.features(public, path, paid)
        elif self.feature_family == 'raw20_control':
            features = core.plain.observables(public, path, paid, self.classifier.values(public, path, paid))
        else:
            raise ValueError('Unknown registered feature family')
        return core.mask_invalid(core.probabilities(self.forecaster, features), bool(paid[path]['answer']))


def fit_dynamics(rows, trees, observer, records, oof, forecaster, family, provenance, smoothing=32.):
    """Build exact finite dynamics from independently excluded TRAIN forecasts."""
    if family not in dense.VARIANTS or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Invalid feature family or duplicate training question')
    if any(r['subject'] not in ('chemistry', 'physics') for r in rows):
        raise ValueError('Science training only')
    keys = [(r['id'], p) for r in rows for p in ('', 'repair', 'retry')]
    if [(r['id'], r['path']) for r in records] != keys:
        raise ValueError('OOF forecasts must preserve exact question/prefix ordering')
    oof = np.asarray(oof, dtype=float)
    if oof.shape != (len(keys), 16) or not np.isfinite(oof).all() or np.any(oof < 0):
        raise ValueError('Require all finite, nonnegative sixteen-state forecasts')
    np.testing.assert_allclose(oof.sum(1), 1., atol=1e-10)
    labels, signals = {}, {}
    for i, (row, path) in enumerate((r, p) for r in rows for p in ('', 'repair', 'retry')):
        qid = row['id']; prefix = path + '/' if path else ''
        children = [prefix + a for a in core.ACTIONS[1:]]
        labels[qid, path] = core.state(core.plain.correct(trees[qid][path], row['answer']),
            *[core.plain.correct(trees[qid][p], row['answer']) for p in children])
        signals[qid, path] = core.observation(oof[i], bool(trees[qid][path]['answer']))
    support = np.asarray([[[float(core.CURRENT[t] == core.AFTER[a, s]) for t in core.STATES]
                          for s in core.STATES] for a in (0, 1)])
    emission_support = np.ones((2, 16, core.N_OBS)); emission_support[:, core.CURRENT == 1, 24:] = 0.
    encoder_cost = observer._recipe['parameters'] / 1e9 * observer._recipe['max_length'] / 3.09

    def counts(group):
        prior = np.ones(16); transition = .1 * support.copy(); emission = .1 * emission_support.copy()
        prices, number = np.zeros((2, 3, 16)), np.zeros((2, 3, 16))
        for row in group:
            qid = row['id']; root = labels[qid, '']; prior[root] += 1
            for action, child in enumerate(('repair', 'retry')):
                nxt = labels[qid, child]
                if core.CURRENT[nxt] != core.AFTER[action, root]:
                    raise ValueError('Paired current-answer truth changed')
                transition[action, root, nxt] += 1
                emission[action, nxt, signals[qid, child]] += 1
            for stage, paths in ((0, ('',)), (1, ('repair', 'retry'))):
                for path in paths:
                    s = labels[qid, path]
                    for action, edge in enumerate(core.ACTIONS[1:]):
                        node = trees[qid][(path + '/' if path else '') + edge]
                        price = 4 * (node['input_tokens'] + node['output_tokens']) / 3.09
                        if stage == 0 and edge != 'complete': price += encoder_cost
                        prices[stage, action, s] += price; number[stage, action, s] += 1
        return prior, transition, emission, prices, number

    row_prob = lambda x: x / x.sum(-1, keepdims=True)
    pooled = counts(rows); dynamics = {}
    for subject in sorted({r['subject'] for r in rows}):
        local = counts([r for r in rows if r['subject'] == subject])
        prior = core.normalize(local[0] + smoothing * core.normalize(pooled[0]))
        transition = row_prob(local[1] + smoothing * row_prob(pooled[1]))
        emission = row_prob(local[2] + smoothing * row_prob(pooled[2]))
        pooled_cost = np.divide(pooled[3], pooled[4], out=np.full_like(pooled[3], 650.), where=pooled[4] > 0)
        cost = (local[3] + 4 * pooled_cost) / (local[4] + 4)
        dynamics[subject] = core.Dynamics(prior, transition, emission, cost)
    return Planner(observer, forecaster, dynamics, {
        'schema': 'regularized-science-Complete16-v1', 'state_count': 16, 'paired_questions': len(rows),
        'decision_nodes': len(keys), 'feature_family': family, 'smoothing': smoothing,
        'maximum_additional_solver_calls': 2, 'maximum_reviews': 0, 'complete_is_terminal': True,
        'future_observations': 'Exact finite-observation two-step Bellman backup',
        'observation_forecasts': 'Nested question-excluded training predictions',
        'original_dynamics_formulas_preserved': True, 'provenance': provenance}, family)


evaluate = core.evaluate
