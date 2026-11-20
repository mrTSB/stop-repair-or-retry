"""Contextual two-state POMDP with explicit filtering and selective LLM observations."""

from dataclasses import dataclass, field
import hashlib
import numpy as np

from extended_acquire import CHECKPOINTS, JUDGE_ACTIVE, JUDGE_TOTAL, PATHS
from belief import ACTIONS, GRID, OBSERVATIONS, category
from features import visible_features
from gate import Logistic, basis
from scoring import canonical_answer, correct

REVIEW_BINS = np.array([.2, .8, .97, .995])
REVIEW_WIDTH = 3 * (len(REVIEW_BINS) + 1) * 2 + 1
BELIEF_CATEGORIES = np.array([category(b) for b in GRID])


def equal(a, b):
    return bool(a and b and canonical_answer(a).upper() == canonical_answer(b).upper())


def features(row, path, tree):
    node = tree[path]
    parent = path.rsplit('/', 1)[0] if '/' in path else ''
    return visible_features({**row, 'compact': node})[0] + [
        bool(path), path.endswith(('repair', 'guided')), path.endswith('retry'),
        equal(node['answer'], tree[parent]['answer']) if path else False,
        equal(node['answer'], tree['']['answer']) if path else False,
        'guided' in path, min(len(node['text']) / 1000, 2)]


def encode(judge, current):
    if not judge['parse_valid']:
        return REVIEW_WIDTH - 1
    bucket = int(np.searchsorted(REVIEW_BINS, judge['p_correct']))
    agreement = int(equal(judge['answer'], current['answer']))
    return (ACTIONS.index(judge['action']) * (len(REVIEW_BINS) + 1) + bucket) * 2 + agreement


def mentor_comparison(judge, current):
    if not judge['parse_valid'] or not judge['answer'] or not current['answer']:
        return 0
    return 2 if equal(judge['answer'], current['answer']) else 1


def classifier_fit(examples, strength, richness):
    x = basis(np.array([features(r, p, t) for r, p, t in examples]), richness)
    y = np.array([correct(t[p], r['answer']) for r, p, t in examples])
    folds = np.array([int(hashlib.sha256(r['id'].encode()).hexdigest()[:8], 16) % 5 for r, _, _ in examples])
    scores = np.empty(len(y))
    for fold in range(5):
        inside = folds != fold
        scores[~inside] = Logistic.fit(x[inside], y[inside], strength).predict(x[~inside])
    return Logistic.fit(x, y, strength), scores


@dataclass
class Dynamics:
    prior: float
    initial_obs: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    costs: np.ndarray
    review_emission: np.ndarray
    review_transition: np.ndarray
    review_solver_emission: np.ndarray
    review_costs: np.ndarray
    judge_costs: np.ndarray
    predictors: list
    calibration: Logistic
    agreement_emission: np.ndarray | None = None
    _plans: dict = field(default_factory=dict, repr=False)
    _reviews: dict = field(default_factory=dict, repr=False)
    _information: dict = field(default_factory=dict, repr=False)
    _transitions: dict = field(default_factory=dict, repr=False)
    _current_values: dict = field(default_factory=dict, repr=False)

    def initial(self, score, observation, mode):
        if observation == 0:
            return 0.
        if mode == 'calibrated':
            p = np.clip(score, 1e-6, 1 - 1e-6)
            return float(self.calibration.predict(np.array([[np.log(p / (1-p))]]))[0])
        mass = np.array([1-self.prior, self.prior]) * self.initial_obs[:, observation]
        return float(mass[1] / mass.sum())

    def predict_transition(self, x, stage, contextual):
        if not contextual:
            return self.transition[stage]
        key = x.tobytes()
        if key not in self._transitions:
            self._transitions[key] = np.array([[self.predictors[a][s].predict(x)[0] for s in (0, 1)] for a in (0, 1)])
        return self._transitions[key]

    def posterior(self, belief, transition, action, observation, stage, reviewed=False, comparison=None):
        prediction = (1-belief)*transition[action, 0] + belief*transition[action, 1]
        emission = self.review_solver_emission if reviewed else self.emission
        mass = np.array([1-prediction, prediction])*emission[stage, action, :, observation]
        if reviewed and stage == 0 and comparison is not None and self.agreement_emission is not None:
            mass *= self.agreement_emission[observation, :, comparison]
        return float(mass[1]/max(mass.sum(), 1e-15))

    def review_update(self, belief, observation, condition=None):
        likelihood = self.review_emission[category(belief) if condition is None else condition, :, observation]
        mass = np.array([1-belief, belief])*likelihood
        return float(mass[1]/max(mass.sum(), 1e-15))

    def action_values(self, beliefs, remaining, transition, costs, future, penalty, reviewed=False, reuse=False):
        b = np.atleast_1d(beliefs)
        values = np.zeros((len(b), 3))
        values[:, 0] = b
        stage = 2-remaining
        emission = self.review_solver_emission if reviewed else self.emission
        for a in (0, 1):
            p = (1-b)*transition[a, 0]+b*transition[a, 1]
            for o in range(OBSERVATIONS):
                mass = np.stack((1-p, p))*emission[stage, a, :, o, None]
                outcomes = range(3) if reviewed and reuse and stage == 0 and self.agreement_emission is not None else (None,)
                for comparison in outcomes:
                    joint = mass if comparison is None else mass*self.agreement_emission[o, :, comparison, None]
                    probability = joint.sum(axis=0)
                    posterior = joint[1]/probability.clip(min=1e-15)
                    values[:, a+1] += probability*np.interp(posterior, GRID, future)
            price = (1-b)*costs[a, 0]+b*costs[a, 1]
            values[:, a+1] -= penalty*price/1000
        return values[0] if np.ndim(beliefs) == 0 else values

    def bellman(self, penalty):
        if penalty not in self._plans:
            planned = np.zeros((3, 3, len(GRID)))
            planned[0, 0] = GRID
            for remaining in (1, 2):
                stage = 2-remaining
                planned[remaining] = self.action_values(
                    GRID, remaining, self.transition[stage], self.costs[stage],
                    planned[remaining-1].max(axis=0), penalty).T
            self._plans[penalty] = planned
        return self._plans[penalty]

    def current_values(self, belief, remaining, transition, planned, penalty):
        key = penalty, remaining, float(belief), transition.tobytes()
        if key not in self._current_values:
            self._current_values[key] = self.action_values(
                belief, remaining, transition, self.costs[2-remaining],
                planned[remaining-1].max(axis=0), penalty)
        return self._current_values[key]

    def review_values(self, belief, remaining, signal, planned, penalty, reuse=False):
        stage = 2-remaining
        key = penalty, remaining, signal, reuse
        if key not in self._reviews:
            self._reviews[key] = self.action_values(GRID, remaining, self.review_transition[stage, signal],
                                                    self.review_costs[stage], planned[remaining-1].max(axis=0), penalty, True, reuse)
        return np.array([np.interp(belief, GRID, self._reviews[key][:, a]) for a in range(3)])

    def information_value(self, belief, remaining, current, planned, penalty, condition=None, reuse=False):
        key = penalty, remaining, condition, reuse
        if key not in self._information:
            value = np.zeros(len(GRID))
            for signal in range(REVIEW_WIDTH):
                categories = BELIEF_CATEGORIES if condition is None else np.full(len(GRID), condition)
                weights = np.column_stack((1-GRID, GRID))*self.review_emission[categories, :, signal]
                probability = weights.sum(axis=1)
                posterior = weights[:, 1]/probability.clip(min=1e-15)
                self.review_values(.5, remaining, signal, planned, penalty, reuse)
                curves = self._reviews[penalty, remaining, signal, reuse]
                after = np.max(np.array([np.interp(posterior, GRID, curves[:, a]) for a in range(3)]), axis=0)
                value += probability*after
            self._information[key] = value-penalty*self.judge_costs[2-remaining]/1000
        return float(np.interp(belief, GRID, self._information[key])-current.max())

    def as_dict(self):
        result = {k: v.tolist() if isinstance(v, np.ndarray) else v
                  for k, v in vars(self).items() if not k.startswith('_')}
        result['predictors'] = [[v.as_dict() for v in group] for group in self.predictors]
        result['calibration'] = self.calibration.as_dict()
        return result

    @classmethod
    def restore(cls, payload):
        fields = dict(payload)
        for key in ('initial_obs', 'transition', 'emission', 'costs', 'review_emission',
                    'review_transition', 'review_solver_emission', 'review_costs', 'judge_costs'):
            fields[key] = np.array(fields[key])
        fields['predictors'] = [[restore_logistic(v) for v in group] for group in fields['predictors']]
        fields['calibration'] = restore_logistic(fields['calibration'])
        if fields.get('agreement_emission') is not None:
            fields['agreement_emission'] = np.array(fields['agreement_emission'])
        return cls(**fields)


def restore_logistic(record):
    return Logistic(**{k: np.array(v) for k, v in record.items()})


@dataclass
class Planner:
    classifier: Logistic
    dynamics: dict
    richness: str
    _scores: dict = field(default_factory=dict, repr=False)

    def observed(self, row, path, tree):
        node = tree[path]
        key = row['id'], path, node['prompt_sha256'], node['text']
        if key not in self._scores:
            x = basis(np.array([features(row, path, tree)]), self.richness)
            score = float(self.classifier.predict(x)[0])
            self._scores[key] = score, category(score, bool(node['answer'])), x
        return self._scores[key]

    def as_dict(self):
        return {'classifier': self.classifier.as_dict(), 'richness': self.richness,
                'dynamics': {k: v.as_dict() for k, v in self.dynamics.items()}}

    @classmethod
    def restore(cls, record):
        return cls(restore_logistic(record['classifier']),
                   {k: Dynamics.restore(v) for k, v in record['dynamics'].items()}, record['richness'])


def fit(rows, trees, judges, archive, strength=64., smoothing=12., richness='linear'):
    ids = {r['id'] for r in rows}
    examples = [(r, '', {'': r['compact']}) for r in archive if r['id'] not in ids]
    examples += [(r, p, trees[r['id']]) for r in rows for p in PATHS]
    classifier, scores = classifier_fit(examples, strength, richness)
    observed = {(r['id'], p): category(v, bool(t[p]['answer']))
                for (r, p, t), v in zip(examples, scores)}
    probabilities = {(r['id'], p): v for (r, p, _), v in zip(examples, scores)}
    domains = {}
    for domain in ('math', 'science'):
        domain_rows = [r for r in rows if r['domain'] == domain]
        initial = np.full((2, OBSERVATIONS), .5)
        initial[1, 0] = 0
        roots, odds = [], []
        for row in domain_rows:
            state = int(correct(trees[row['id']][''], row['answer']))
            roots.append(state)
            p = np.clip(probabilities[row['id'], ''], 1e-6, 1-1e-6)
            odds.append([np.log(p/(1-p))])
            initial[state, observed[row['id'], '']] += 1
        calibration = Logistic.fit(np.array(odds), np.array(roots), 16.)
        initial /= initial.sum(axis=1, keepdims=True)
        prior = (sum(roots)+1)/(len(roots)+2)
        transitions = np.full((2, 2, 2, 2), 1.)
        emissions = np.full((2, 2, 2, OBSERVATIONS), .5)
        prices = [[[[ ] for _ in (0,1)] for _ in (0,1)] for _ in (0,1)]
        training_x = [[[] for _ in (0,1)] for _ in (0,1)]
        training_y = [[[] for _ in (0,1)] for _ in (0,1)]
        for row in domain_rows:
            tree = trees[row['id']]
            for path in CHECKPOINTS:
                stage = int(bool(path))
                state = int(correct(tree[path], row['answer']))
                for a, action in enumerate(('repair', 'retry')):
                    child = f'{path}/{action}' if path else action
                    node = tree[child]
                    nxt = int(correct(node, row['answer']))
                    transitions[stage, a, state, nxt] += 1
                    emissions[stage, a, nxt, observed[row['id'], child]] += 1
                    prices[stage][a][state].append((node['input_tokens']+node['output_tokens'])*4/3.09)
                    training_x[a][state].append(features(row, path, tree))
                    training_y[a][state].append(nxt)
        transition = transitions[..., 1]/transitions.sum(axis=-1)
        emissions[:, :, 1, 0] = 0
        emissions /= emissions.sum(axis=-1, keepdims=True)
        costs = np.array([[[np.mean(values) if values else 500 for values in actions] for actions in stage] for stage in prices])
        predictors = [[Logistic.fit(basis(np.array(training_x[a][s]), richness),
                                   np.array(training_y[a][s]), strength) for s in (0,1)] for a in (0,1)]
        pooled_emission = np.ones((2, REVIEW_WIDTH))
        conditioned = np.zeros((OBSERVATIONS, 2, REVIEW_WIDTH))
        reviewed = np.zeros((2, REVIEW_WIDTH, 2, 2, 2))
        reviewed_pool = np.ones((2, 2, 2, 2))
        review_solver_emission = np.full_like(emissions, .5)
        review_prices = [[[[ ] for _ in (0,1)] for _ in (0,1)] for _ in (0,1)]
        agreement_pooled = np.ones((2, 3))
        agreement_conditioned = np.zeros((OBSERVATIONS, 2, 3))
        judge_prices = [[], []]
        for row in domain_rows:
            tree = trees[row['id']]
            for path in CHECKPOINTS:
                stage = int(bool(path))
                node = judges[row['id'], path]
                signal = encode(node, tree[path])
                state = int(correct(tree[path], row['answer']))
                pooled_emission[state, signal] += 1
                conditioned[observed[row['id'], path], state, signal] += 1
                judge_prices[stage].append((node['input_tokens']+node['output_tokens'])*JUDGE_ACTIVE/3.09)
                for a, action in enumerate(('repair', 'retry')):
                    child = ('guided' if action == 'repair' else 'retry') if not path else f'{path}/{action}'
                    target = tree[child]
                    nxt = int(correct(target, row['answer']))
                    reviewed[stage, signal, a, state, nxt] += 1
                    reviewed_pool[stage, a, state, nxt] += 1
                    review_solver_emission[stage, a, nxt, observed[row['id'], child]] += 1
                    review_prices[stage][a][state].append((target['input_tokens']+target['output_tokens'])*4/3.09)
                    if stage == 0:
                        comparison = mentor_comparison(node, target)
                        agreement_pooled[nxt, comparison] += 1
                        agreement_conditioned[observed[row['id'], child], nxt, comparison] += 1
        pooled_emission /= pooled_emission.sum(axis=-1, keepdims=True)
        conditioned += smoothing*pooled_emission[None]
        conditioned /= conditioned.sum(axis=-1, keepdims=True)
        pool = reviewed_pool/reviewed_pool.sum(axis=-1, keepdims=True)
        reviewed += smoothing*pool[:, None]
        reviewed = reviewed[..., 1]/reviewed.sum(axis=-1)
        review_solver_emission[:, :, 1, 0] = 0
        review_solver_emission /= review_solver_emission.sum(axis=-1, keepdims=True)
        review_costs = np.array([[[np.mean(v) if v else 600 for v in actions] for actions in stage] for stage in review_prices])
        agreement_pooled /= agreement_pooled.sum(axis=-1, keepdims=True)
        agreement_conditioned += smoothing*agreement_pooled[None]
        agreement_conditioned /= agreement_conditioned.sum(axis=-1, keepdims=True)
        domains[domain] = Dynamics(prior, initial, transition, emissions, costs, conditioned, reviewed, review_solver_emission,
                                   review_costs, np.array([np.mean(v) for v in judge_prices]), predictors, calibration,
                                   agreement_conditioned)
    return Planner(classifier, domains, richness)


def evaluate_llm(row, tree, judges, settings):
    path, visited, acquired, decisions = '', [''], [], []
    previous_review = None
    for _ in (2, 1):
        if path and settings.get('reuse_review', False) and mentor_comparison(previous_review, tree[path]) == 2:
            decisions.append({'path': path, 'action': 'stop', 'queried': False})
            break
        judge = judges[row['id'], path]
        previous_review = judge
        acquired.append(path)
        threshold = settings.get('judge_stop', 'action')
        action = judge['action']
        if threshold == 'agreement':
            action = 'stop' if equal(judge['answer'], tree[path]['answer']) else 'repair'
        elif threshold != 'action':
            action = 'stop' if judge['p_correct'] >= threshold else (action if action != 'stop' else 'repair')
        decisions.append({'path': path, 'action': action, 'queried': True})
        if action == 'stop':
            break
        edge = 'guided' if action == 'repair' and not path else action
        path = f'{path}/{edge}' if path else edge
        visited.append(path)
    return outcome(row, tree, judges, path, visited, acquired, decisions)


def outcome(row, tree, judges, path, visited, acquired, decisions):
    solver = sum(tree[p]['input_tokens']+tree[p]['output_tokens'] for p in visited)
    review = sum(judges[row['id'], p]['input_tokens']+judges[row['id'], p]['output_tokens'] for p in acquired)
    return {'id': row['id'], 'domain': row['domain'], 'subject': row['subject'],
            'correct': int(correct(tree[path], row['answer'])), 'path': path,
            'solver_tokens': solver, 'judge_tokens': review, 'compute': (4*solver+JUDGE_ACTIVE*review)/3.09,
            'total_parameter_compute': (4*solver+JUDGE_TOTAL*review)/3.09,
            'solver_calls': len(visited), 'judge_calls': len(acquired), 'decisions': decisions}


def evaluate(row, tree, judges, model, settings, method):
    if method == 'B':
        return evaluate_llm(row, tree, judges, settings)
    dynamics = model.dynamics[row['domain']]
    penalty = settings['penalty']*settings.get('science_weight', 1.) if row['domain'] == 'science' else settings['penalty']
    planned = dynamics.bellman(penalty)
    score, observation, _ = model.observed(row, '', tree)
    belief = dynamics.initial(score, observation, settings.get('initial_mode', 'calibrated'))
    path, visited, acquired, decisions = '', [''], [], []
    for remaining in (2, 1):
        stage = 2-remaining
        _, condition, x = model.observed(row, path, tree)
        transition = dynamics.predict_transition(x, stage, settings.get('contextual', True))
        values = dynamics.current_values(belief, remaining, transition, planned, penalty)
        information = None
        query = method == 'always'
        if method == 'C':
            information = dynamics.information_value(belief, remaining, values, planned, penalty, condition, settings.get('reuse_review', False))
            query = information > settings['threshold']
        if method == 'random':
            rank = int(hashlib.sha256(f'random:{row["id"]}:{path}'.encode()).hexdigest()[:8], 16)/2**32
            query = rank < settings['rate']
        if query:
            judge = judges[row['id'], path]
            acquired.append(path)
            signal = encode(judge, tree[path])
            belief = dynamics.review_update(belief, signal, condition)
            transition = dynamics.review_transition[stage, signal]
            values = dynamics.review_values(belief, remaining, signal, planned, penalty, settings.get('reuse_review', False))
            action = ACTIONS[int(values.argmax())]
        else:
            action = ACTIONS[int(values.argmax())]
        decisions.append({'path': path, 'belief': belief, 'action': action, 'queried': query, 'information_value': information})
        if action == 'stop':
            break
        edge = 'guided' if action == 'repair' and query and not path else action
        path = f'{path}/{edge}' if path else edge
        visited.append(path)
        _, observed, _ = model.observed(row, path, tree)
        comparison = mentor_comparison(judge, tree[path]) if query and settings.get('reuse_review', False) else None
        belief = dynamics.posterior(belief, transition, ('repair', 'retry').index(action), observed, stage, query, comparison)
    return outcome(row, tree, judges, path, visited, acquired, decisions)
