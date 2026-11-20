"""Two-state Bayesian filtering and finite-horizon Bellman planning."""

from dataclasses import dataclass, field
import hashlib

import numpy as np

from acquire import CHECKPOINTS, PATHS, paired_archive
from features import visible_features
from gate import Logistic, basis
from scoring import canonical_answer, correct

ACTIONS = ('stop', 'repair', 'retry')
GRID = np.linspace(0, 1, 501)
BINS = np.array([.25, .5, .7, .85])
OBSERVATIONS = 6
JUDGE_BINS = np.array([.2,.8,.97,.995])
JUDGE_OBSERVATIONS = len(ACTIONS)*(len(JUDGE_BINS)+1)


def category(probability, valid=True):
    return 0 if not valid else 1 + int(np.searchsorted(BINS, probability))


def features(row, path, tree):
    node = tree[path]
    values = visible_features({**row, 'compact': node})[0]
    parent_path = path.rsplit('/', 1)[0] if '/' in path else ''
    parent = tree[parent_path] if path else node
    equal = bool(path and node['answer'] and canonical_answer(node['answer']) == canonical_answer(parent['answer']))
    return values + [bool(path), path.endswith('repair'), path.endswith('retry'), equal, bool(parent['answer'])]


def encode(node):
    bucket = int(np.searchsorted(JUDGE_BINS,node['p_correct']))
    return (len(JUDGE_BINS)+1)*ACTIONS.index(node['action'])+bucket


def fitted_classifier(examples, strength, richness):
    x = basis(np.array([features(row, path, tree) for row, path, tree in examples]),richness)
    y = np.array([correct(tree[path], row['answer']) for row, path, tree in examples])
    groups = np.array([int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16) % 5 for row, _, _ in examples])
    predicted = np.empty(len(y))
    for fold in range(5):
        inside = groups != fold
        model = Logistic.fit(x[inside], y[inside], strength)
        predicted[~inside] = model.predict(x[~inside])
    return Logistic.fit(x, y, strength), predicted


@dataclass
class Dynamics:
    prior: float
    initial_obs: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    costs: np.ndarray
    judge_emission: np.ndarray
    judge_cost: float
    judge_transition: np.ndarray | None = None
    judge_stage_cost: np.ndarray | None = None
    _plans: dict = field(default_factory=dict, repr=False)
    _context_plans: dict = field(default_factory=dict, repr=False)
    _information: dict = field(default_factory=dict, repr=False)

    def initial(self, observation):
        weighted = np.array([1-self.prior, self.prior]) * self.initial_obs[:, observation]
        return float(weighted[1] / weighted.sum())

    def update(self, belief, action, observation, judge_observation=None):
        transition = self.transition if judge_observation is None or self.judge_transition is None else self.judge_transition[judge_observation]
        predicted = (1-belief)*transition[action, 0] + belief*transition[action, 1]
        weighted = np.array([1-predicted, predicted]) * self.emission[action, :, observation]
        return float(weighted[1] / max(weighted.sum(), 1e-15))

    def judge_update(self, belief, observation):
        c = category(belief)
        weighted = np.array([1-belief, belief]) * self.judge_emission[c, :, observation]
        return float(weighted[1] / max(weighted.sum(), 1e-15))

    def bellman(self, penalty):
        if penalty in self._plans:
            return self._plans[penalty]
        values = np.zeros((3, 3, len(GRID)))
        values[:, 0] = GRID
        for remaining in (1, 2):
            for action in (0, 1):
                prediction = (1-GRID)*self.transition[action, 0] + GRID*self.transition[action, 1]
                best_future = values[remaining-1].max(axis=0)
                outcome = np.zeros(len(GRID))
                for observation in range(OBSERVATIONS):
                    weights = np.stack((1-prediction, prediction)) * self.emission[action, :, observation, None]
                    probability = weights.sum(axis=0)
                    posterior = weights[1] / probability.clip(min=1e-15)
                    outcome += probability * np.interp(posterior, GRID, best_future)
                cost = (1-GRID)*self.costs[action, 0] + GRID*self.costs[action, 1]
                values[remaining, action+1] = outcome - penalty*cost/1000
        self._plans[penalty] = values
        return values

    def values(self, belief, remaining, planned):
        return np.array([np.interp(belief, GRID, vector) for vector in planned[remaining]])

    def judge_values(self, belief, remaining, observation, planned, penalty):
        if self.judge_transition is None:
            return self.values(belief, remaining, planned)
        key = penalty, remaining, observation
        if key not in self._context_plans:
            values = np.zeros((3,len(GRID)))
            values[0] = GRID
            future = planned[remaining-1].max(axis=0)
            transition = self.judge_transition[observation]
            for action in (0,1):
                prediction = (1-GRID)*transition[action,0]+GRID*transition[action,1]
                for nxt in range(OBSERVATIONS):
                    weights = np.stack((1-prediction,prediction))*self.emission[action,:,nxt,None]
                    probability = weights.sum(axis=0)
                    posterior = weights[1]/probability.clip(min=1e-15)
                    values[action+1] += probability*np.interp(posterior,GRID,future)
                cost = (1-GRID)*self.costs[action,0]+GRID*self.costs[action,1]
                values[action+1] -= penalty*cost/1000
            self._context_plans[key] = values
        return np.array([np.interp(belief,GRID,vector) for vector in self._context_plans[key]])

    def information_value(self, belief, remaining, planned, penalty):
        key = belief,remaining,penalty
        if key in self._information:
            return self._information[key]
        c = category(belief)
        weights = np.array([1-belief, belief])[:, None] * self.judge_emission[c]
        probabilities = weights.sum(axis=0)
        posterior = weights[1] / probabilities.clip(min=1e-15)
        future = np.array([self.judge_values(p,remaining,signal,planned,penalty).max() for signal,p in enumerate(posterior)])
        current = self.values(belief, remaining, planned).max()
        cost = self.judge_cost if self.judge_stage_cost is None else self.judge_stage_cost[2-remaining]
        value = float(probabilities @ future-current-penalty*cost/1000)
        self._information[key] = value
        return value

    def as_dict(self):
        return {k: value.tolist() if isinstance(value, np.ndarray) else value for k, value in vars(self).items() if not k.startswith('_')}

    @classmethod
    def restore(cls, record):
        fields = dict(record)
        for name in ('initial_obs', 'transition', 'emission', 'costs', 'judge_emission'):
            fields[name] = np.array(fields[name])
        if fields.get('judge_transition') is not None:
            fields['judge_transition'] = np.array(fields['judge_transition'])
        if fields.get('judge_stage_cost') is not None:
            fields['judge_stage_cost'] = np.array(fields['judge_stage_cost'])
        return cls(**fields)


@dataclass
class BeliefPlanner:
    classifier: Logistic
    dynamics: dict
    richness: str = 'linear'
    initial_mode: str = 'binned'
    initial_calibration: dict | None = None
    _observations: dict = field(default_factory=dict, repr=False)

    def observation(self, row, path, tree):
        node = tree[path]
        key = row['id'], path, node['prompt_sha256'], node['text']
        if key not in self._observations:
            raw = np.array([features(row,path,tree)])
            probability = self.classifier.predict(basis(raw,self.richness))[0]
            self._observations[key] = probability,category(probability,bool(node['answer']))
        return self._observations[key][1]

    def initial(self,row,tree):
        observed = self.observation(row,'',tree)
        if self.initial_mode=='binned':
            return self.dynamics[row['domain']].initial(observed)
        node = tree['']
        if not node['answer']:
            return 0.
        key = row['id'],'',node['prompt_sha256'],node['text']
        probability = np.clip(self._observations[key][0],1e-6,1-1e-6)
        odds = np.array([[np.log(probability/(1-probability))]])
        return float(self.initial_calibration[row['domain']].predict(odds)[0])

    def as_dict(self):
        return {'classifier': self.classifier.as_dict(), 'dynamics': {k:v.as_dict() for k,v in self.dynamics.items()},
                'richness':self.richness,'initial_mode':self.initial_mode,
                'initial_calibration':None if self.initial_calibration is None else {k:v.as_dict() for k,v in self.initial_calibration.items()}}

    @classmethod
    def restore(cls, payload):
        classifier = Logistic(**{k:np.array(v) for k,v in payload['classifier'].items()})
        saved = payload.get('initial_calibration')
        calibrators = None if saved is None else {name:Logistic(**{k:np.array(v) for k,v in item.items()}) for name,item in saved.items()}
        return cls(classifier, {k:Dynamics.restore(v) for k,v in payload['dynamics'].items()},
                   payload.get('richness','linear'),payload.get('initial_mode','binned'),calibrators)


def fit(rows, trees, judges, strength, smoothing, richness='linear',initial_mode='binned'):
    archive = paired_archive()
    current_ids = {row['id'] for row in rows}
    examples = [(row, '', {'':row['compact']}) for row in archive if row['id'] not in current_ids]
    for row in rows:
        examples.extend((row, path, trees[row['id']]) for path in PATHS)
    classifier, probabilities = fitted_classifier(examples, strength,richness)
    observations = {(row['id'], path): category(p, bool(tree[path]['answer']))
                    for (row,path,tree), p in zip(examples, probabilities)}
    root_probability = {(row['id'],path):p for (row,path,_),p in zip(examples,probabilities)}
    by_domain = {}
    calibrators = {}
    for domain in ('math', 'science'):
        initial = np.full((2, OBSERVATIONS), .5)
        transition = np.full((2, 2, 2), smoothing)
        emission = np.full((2, 2, OBSERVATIONS), .5)
        costs = [[[], []], [[], []]]
        root_states = []
        root_odds = []
        for row, path, tree in examples:
            if row['domain'] != domain or path:
                continue
            state = int(correct(tree[''], row['answer']))
            initial[state, observations[row['id'], '']] += 1
            root_states.append(state)
            probability = np.clip(root_probability[row['id'],''],1e-6,1-1e-6)
            root_odds.append([np.log(probability/(1-probability))])
        calibrators[domain] = Logistic.fit(np.array(root_odds),np.array(root_states),8.)
        for row in rows:
            if row['domain'] != domain:
                continue
            tree = trees[row['id']]
            for parent in CHECKPOINTS:
                state = int(correct(tree[parent], row['answer']))
                for action, name in enumerate(('repair', 'retry')):
                    path = f'{parent}/{name}' if parent else name
                    target = tree[path]
                    nxt = int(correct(target, row['answer']))
                    transition[action, state, nxt] += 1
                    emission[action, nxt, observations[row['id'], path]] += 1
                    costs[action][state].append(target['input_tokens'] + target['output_tokens'])
        emission[:, 1, 0] = 0
        initial[1, 0] = 0
        initial /= initial.sum(axis=1, keepdims=True)
        emission /= emission.sum(axis=2, keepdims=True)
        trans = transition[:, :, 1] / transition.sum(axis=2)
        average = np.array([[np.mean(costs[a][s]) if costs[a][s] else 300 for s in (0,1)] for a in (0,1)])*4/3.09
        prior = (sum(root_states) + 1) / (len(root_states) + 2)
        baseline_judge = np.full((2,JUDGE_OBSERVATIONS),1.)
        conditional = np.zeros((OBSERVATIONS,2,JUDGE_OBSERVATIONS))
        contextual_transition = np.zeros((JUDGE_OBSERVATIONS,2,2,2))
        judge_costs = []
        judge_stage_costs = [[],[]]
        provisional = Dynamics(prior,initial,trans,emission,average,
                               np.full((OBSERVATIONS,2,JUDGE_OBSERVATIONS),1/JUDGE_OBSERVATIONS),0)
        for row in rows:
            if row['domain'] != domain:
                continue
            tree = trees[row['id']]
            first = provisional.initial(observations[row['id'], ''])
            if initial_mode=='calibrated' and tree['']['answer']:
                p = np.clip(root_probability[row['id'],''],1e-6,1-1e-6)
                first = float(calibrators[domain].predict(np.array([[np.log(p/(1-p))]]))[0])
            for path in CHECKPOINTS:
                belief = first
                if path:
                    belief = provisional.update(first, ('repair','retry').index(path), observations[row['id'],path])
                node = judges[row['id'],path]
                label = int(correct(tree[path], row['answer']))
                signal = encode(node)
                baseline_judge[label,signal] += 1
                conditional[category(belief),label,signal] += 1
                for action,name in enumerate(('repair','retry')):
                    child = f'{path}/{name}' if path else name
                    after = int(correct(tree[child],row['answer']))
                    contextual_transition[signal,action,label,after] += 1
                price = (node['input_tokens']+node['output_tokens'])*9/4
                judge_costs.append(price)
                judge_stage_costs[int(bool(path))].append(price)
        pooled = baseline_judge / baseline_judge.sum(axis=1, keepdims=True)
        conditional += smoothing*pooled[None,:,:]
        conditional /= conditional.sum(axis=2, keepdims=True)
        contextual_transition += smoothing*np.stack((1-trans,trans),axis=-1)[None,:,:,:]
        contextual_transition = contextual_transition[:,:,:,1]/contextual_transition.sum(axis=3)
        stage_cost = np.array([np.mean(values) for values in judge_stage_costs])*4/3.09
        by_domain[domain] = Dynamics(prior, initial, trans, emission, average, conditional, float(np.mean(judge_costs))*4/3.09, contextual_transition,stage_cost)
    return BeliefPlanner(classifier, by_domain,richness,initial_mode,calibrators)


def evaluate(row, tree, judges, model, settings, method):
    dynamics = model.dynamics[row['domain']]
    penalty = settings['penalty'] * (settings.get('science_weight',1) if row['domain']=='science' else 1)
    planned = dynamics.bellman(penalty)
    path, visited, acquired = '', [''], []
    belief = model.initial(row,tree)
    decisions = []
    for remaining in (2,1):
        judge_observation = None
        values = dynamics.values(belief, remaining, planned)
        query = method in ('B','always')
        info = None
        if method == 'C':
            info = dynamics.information_value(belief,remaining,planned,penalty)
            query = info > settings['threshold']
        if method == 'random':
            rank = int(hashlib.sha256(f'random:{row["id"]}:{path}'.encode()).hexdigest()[:8],16) / 2**32
            query = rank < settings['rate']
        if query:
            judge = judges[row['id'],path]
            acquired.append(path)
            if method == 'B':
                threshold = settings.get('judge_stop', 'action')
                action = judge['action']
                if threshold != 'action':
                    action = 'stop' if judge['p_correct'] >= threshold else action
                    if action == 'stop' and judge['p_correct'] < threshold:
                        action = 'repair'
            else:
                judge_observation = encode(judge)
                belief = dynamics.judge_update(belief, judge_observation)
                action = ACTIONS[int(dynamics.judge_values(belief,remaining,judge_observation,planned,penalty).argmax())]
        else:
            action = ACTIONS[int(values.argmax())]
        decisions.append({'path':path,'belief':belief,'action':action,'queried':query,'information_value':info})
        if action == 'stop':
            break
        path = f'{path}/{action}' if path else action
        visited.append(path)
        obs = model.observation(row, path, tree)
        belief = dynamics.update(belief, ('repair','retry').index(action), obs, judge_observation)
    solver = sum(tree[key]['input_tokens']+tree[key]['output_tokens'] for key in visited)
    judge_tokens = sum(judges[row['id'],key]['input_tokens']+judges[row['id'],key]['output_tokens'] for key in acquired)
    return {'id':row['id'], 'domain':row['domain'], 'subject':row['subject'], 'correct':int(correct(tree[path],row['answer'])),
            'path':path,'solver_tokens':solver,'judge_tokens':judge_tokens,
            'compute':(solver+judge_tokens*9/4)*4/3.09, 'judge_calls':len(acquired),
            'solver_calls':len(visited), 'decisions':decisions}
