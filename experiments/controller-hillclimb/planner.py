"""Two-checkpoint action-value models trained on paired solver branches."""

from dataclasses import dataclass

import numpy as np

from collect import MODEL_SIZES
from observations import basic, critic_features, trace, visited


SOLVER_PARAMETERS_B = 3.09
CHECKPOINTS = ('', 'repair', 'retry')
FIRST_ACTION = np.array([0, 1, 2, 1, 1, 2, 2])
PLANS = ('', 'repair', 'retry', 'repair/repair', 'repair/retry', 'retry/repair', 'retry/retry')
FIXED = dict(zip(('stop', 'repair_once', 'retry_once', 'repair_twice', 'retry_then_repair',
                  'repair_then_retry', 'retry_twice'), (0, 1, 2, 3, 5, 4, 6)))
PATH_TOKENS = np.array([[i in visited(plan) for i in PLANS] for plan in PLANS], dtype=int)


@dataclass
class Data:
    ids: tuple
    domain: np.ndarray
    basic: np.ndarray
    trace: np.ndarray
    correct: np.ndarray
    tokens: np.ndarray
    extra: dict
    critic_tokens: dict

    def take(self, subset):
        return Data(tuple(key for key, use in zip(self.ids, subset) if use), self.domain[subset],
                    self.basic[subset], self.trace[subset], self.correct[subset], self.tokens[subset],
                    {key: value[subset] for key, value in self.extra.items()},
                    {key: value[subset] for key, value in self.critic_tokens.items()})


def prepare(rows, signals):
    from observations import observed

    ids = tuple(row['id'] for row in rows)
    contexts = [[observed(row, checkpoint) for checkpoint in CHECKPOINTS] for row in rows]
    extra = {}
    critic_tokens = {}
    for model, by_id in signals.items():
        extra[model] = np.array([
            [critic_features(context, by_id[key]) for context in history]
            for key, history in zip(ids, contexts)
        ], dtype=float)
        critic_tokens[model] = np.array([
            (by_id[key]['input_tokens'], by_id[key]['output_tokens']) for key in ids
        ])
    return Data(
        ids, np.array([row['domain'] for row in rows]),
        np.array([[basic(context) for context in history] for history in contexts], dtype=float),
        np.array([[trace(context) for context in history] for history in contexts], dtype=float),
        np.array([[row['tree'][path]['correct'] for path in PLANS] for row in rows], dtype=int),
        np.array([[(row['tree'][path]['input_tokens'], row['tree'][path]['output_tokens'])
                   for path in PLANS] for row in rows]), extra, critic_tokens,
    )


def features(data, richness, critic):
    blocks = [data.basic]
    if richness != 'confidence':
        blocks.append(data.trace)
    if critic:
        blocks.append(data.extra[critic])
    values = np.concatenate(blocks, axis=-1)
    return values


@dataclass
class Ridge:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls, x, y, strength):
        mean, intercept = x.mean(axis=0), y.mean(axis=0)
        # Scaling with a floor keeps rare features from dominating a small training set.
        scale = x.std(axis=0).clip(min=0.25)
        x = (x - mean) / scale
        gram = x.T @ x + strength * np.eye(x.shape[1])
        weights = np.linalg.solve(gram, x.T @ (y - intercept))
        return cls(mean, scale, weights, intercept)

    def predict(self, x):
        return (x - self.mean) / self.scale @ self.weights + self.intercept

    def as_dict(self):
        return {key: value.tolist() for key, value in vars(self).items()}


@dataclass
class Planner:
    root: Ridge
    followup: Ridge
    richness: str
    critic: str

    @classmethod
    def fit(cls, data, richness, strength, critic=''):
        x = features(data, richness, critic)
        y = data.correct
        costs = data.tokens.sum(axis=2) @ PATH_TOKENS.T
        root = np.concatenate([y[:, 1:] - y[:, [0]], (costs[:, 1:] - costs[:, [0]]) / 1000], axis=1)
        # Share the follow-up model across the two previous actions. The previous
        # action and how the returned answer changed are present in the features.
        followup = np.concatenate([
            y[:, 3:].reshape(-1, 2) - y[:, 1:3].reshape(-1, 1),
            (costs[:, 3:].reshape(-1, 2) - costs[:, 1:3].reshape(-1, 1)) / 1000,
        ], axis=1)
        return cls(Ridge.fit(x[:, 0], root, strength),
                   Ridge.fit(x[:, 1:].reshape(-1, x.shape[-1]), followup, strength), richness, critic)

    def predict(self, data):
        x = features(data, self.richness, self.critic)
        return self.root.predict(x[:, 0]), self.followup.predict(x[:, 1:])

    def as_dict(self):
        return {'root': self.root.as_dict(), 'followup': self.followup.as_dict(),
                'richness': self.richness, 'critic': self.critic}


def utility(predictions, penalty):
    gain, cost = np.split(predictions, 2, axis=-1)
    zeros = np.zeros((*gain.shape[:-1], 1))
    return np.concatenate([zeros, gain - penalty.reshape((-1,) + (1,) * (gain.ndim - 1)) * cost.clip(min=0)], axis=-1)


def actions(root, followup, penalty):
    root_values = utility(root, penalty)
    root_values = np.stack([root_values[:, 0], root_values[:, [1, 3, 4]].max(axis=1),
                            root_values[:, [2, 5, 6]].max(axis=1)], axis=1)
    return root_values, utility(followup, penalty)


def critic_gate(data, domain, margin, values):
    ordered = np.sort(values, axis=-1)
    allowed = (data.domain == domain) if domain != 'all' else np.ones(len(data.ids), dtype=bool)
    return allowed & ((ordered[:, -1] - ordered[:, -2]) <= margin)


def decide(data, config, base, reviewed=None):
    math = data.domain == 'math'
    penalty = config['penalty'] * np.where(math, 1.0, config.get('science_weight', 1.0))
    root, followup = actions(*base, penalty)
    queried = np.zeros(len(data.ids), dtype=bool)
    if reviewed is not None:
        checked_root, checked_followup = actions(*reviewed, penalty)
        queried = critic_gate(data, config['scope'], config['margin'], root)
        root = np.where(queried[:, None], checked_root, root)
        followup = np.where(queried[:, None, None], checked_followup, followup)
        if config.get('accept_agreement'):
            agrees = data.extra[config['critic']][:, :, 1].astype(bool)
            verified = (queried & math)[:, None] & agrees
            root[verified[:, 0], 1:] = -np.inf
            followup[:, :, 1:] = np.where(verified[:, 1:, None], -np.inf, followup[:, :, 1:])
    action1 = root.argmax(axis=-1)
    action2 = followup[np.arange(len(data.ids)), np.maximum(action1 - 1, 0)].argmax(axis=-1)
    paths = np.where((action1 == 0) | (action2 == 0), action1, 3 + 2 * (action1 - 1) + action2 - 1)
    return paths, queried


def outcomes(data, paths, queried=None, model=''):
    solver = np.einsum('ij,ijk->ik', PATH_TOKENS[paths], data.tokens)
    extra = np.zeros_like(solver) if not model else data.critic_tokens[model] * queried[:, None]
    work = solver.sum(axis=1) + extra.sum(axis=1) * (MODEL_SIZES.get(model, 0.0) / SOLVER_PARAMETERS_B)
    return {'correct': data.correct[np.arange(len(paths)), paths], 'solver': solver,
            'critic': extra, 'work': work, 'paths': paths}
