"""Controllers trained on training data and selected only on development data."""
from collections import defaultdict
from itertools import product

from .core import ACTIONS, confidence, correct, node_cost, walk


def features(tree, path):
    node = tree[path]
    parts = path.split('/') if path else []
    parent = tree['/'.join(parts[:-1])] if parts else None
    changed = int(parent is not None and node.get('answer') != parent.get('answer'))
    return (confidence(node), len(parts), parts[-1] if parts else 'initial', changed)


class TabularQ:
    """Smoothed direct one-step action-value regression on observable features.

    Predicts terminal correctness and cost from paired branch labels. It deliberately
    does not use latent correctness, ground truth, or future outcomes at inference.
    """
    def __init__(self, rows, smoothing=5):
        self.stats = defaultdict(lambda: [0, 0, 0])
        self.global_stats = defaultdict(lambda: [0, 0, 0])
        self.smoothing = smoothing
        for row in rows:
            tree, gold = row['tree'], row['answer']
            for path, node in walk(tree):
                if len(path.split('/')) >= 2 and path:
                    continue
                f = features(tree, path)
                choices = {'stop': node}
                for action in ACTIONS:
                    child = path + ('/' if path else '') + action
                    if child in tree:
                        choices[action] = tree[child]
                for action, target in choices.items():
                    for stats in (self.stats[(f, action)], self.global_stats[action]):
                        stats[0] += 1
                        stats[1] += int(correct(target, gold))
                        stats[2] += node_cost(target) if action != 'stop' else 0

    def values(self, tree, path, penalty):
        f = features(tree, path)
        values = {}
        for action in ('stop', *ACTIONS):
            n, successes, cost = self.stats[(f, action)]
            gn, gs, gc = self.global_stats[action]
            prior_success = gs / gn if gn else 0.5
            prior_cost = gc / gn if gn else 0
            values[action] = (successes + self.smoothing * prior_success) / (n + self.smoothing) - penalty * (cost + self.smoothing * prior_cost) / (n + self.smoothing) / 1000
        return values


def best(values):
    # Stable tie-breaking favors stopping, then repair.
    return max(values, key=values.get)


def choose(policy, tree, path, belief, remaining, model, qmodel, penalty, setting=None):
    node = tree[path]
    if policy == 'stop': return 'stop', None
    if policy in ('repair', 'retry'): return policy, None
    if policy == 'schedule':
        depth = len(path.split('/')) if path else 0
        return setting[depth], None
    if policy == 'threshold':
        threshold, action = setting
        return ('stop' if ('low', 'medium', 'high').index(confidence(node)) >= threshold else action), None
    if policy == 'supervised': return best(qmodel.values(tree, path, penalty)), None
    if policy == 'one_step': return best(model.values(belief, 1, penalty)), None
    values = model.values(belief, remaining, penalty)
    if policy == 'pomdp': return best(values), None
    if policy in ('judge_full', 'judge_compact'):
        mode = policy.removeprefix('judge_')
        return node['judges'][mode]['action'], node['judges'][mode]
    if policy == 'hybrid':
        ordered = sorted(values.values(), reverse=True)
        if ordered[0] - ordered[1] < setting:
            judge = node['judges']['full']
            return judge['action'], judge
        return best(values), None
    raise ValueError(f'Unknown policy: {policy}')


def simulate(row, policy, model, qmodel, penalty, setting=None):
    tree, gold = row['tree'], row['answer']
    path = ''
    node = tree[path]
    belief = model.initial_belief(confidence(node))
    total_input, total_output = node.get('input_tokens', 0), node.get('output_tokens', 0)
    solver_latency = node.get('latency_seconds', 0)
    judge_latency = 0
    judge_calls = 0
    history = []
    for remaining in (2, 1):
        action, judge = choose(policy, tree, path, belief, remaining, model, qmodel, penalty, setting)
        if judge:
            total_input += judge.get('input_tokens', 0)
            total_output += judge.get('output_tokens', 0)
            judge_latency += judge.get('latency_seconds', 0)
            judge_calls += 1
        if action not in ('stop', *ACTIONS):
            raise ValueError(f'Invalid action {action!r}')
        history.append(action)
        if action == 'stop': break
        next_path = path + ('/' if path else '') + action
        if next_path not in tree:
            raise ValueError(f'Missing trajectory {next_path} for {row.get("id")}')
        node = tree[next_path]
        belief, _ = model.update(belief, action, confidence(node))
        path = next_path
        total_input += node.get('input_tokens', 0)
        total_output += node.get('output_tokens', 0)
        solver_latency += node.get('latency_seconds', 0)
    return {'id': row['id'], 'correct': int(correct(node, gold)), 'initial_correct': int(correct(tree[''], gold)), 'answer': node.get('answer', ''), 'input_tokens': total_input, 'output_tokens': total_output, 'tokens': total_input + total_output, 'solver_latency_seconds': solver_latency, 'judge_latency_seconds': judge_latency, 'judge_calls': judge_calls, 'actions': history, 'path': path}


def tune(rows, policy, model, qmodel, penalty):
    if policy == 'schedule':
        candidates = list(product(('stop', *ACTIONS), repeat=2))
    elif policy == 'threshold':
        # 0 stops on all bins; 3 stops on none.
        candidates = list(product(range(4), ACTIONS))
    elif policy == 'hybrid':
        candidates = [0, .005, .01, .02, .05, .1, .2, 1]
    else:
        return None
    def utility(setting):
        results = [simulate(r, policy, model, qmodel, penalty, setting) for r in rows]
        return sum(x['correct'] - penalty * x['tokens'] / 1000 for x in results) / len(results)
    if not rows: raise ValueError('Development data is empty')
    return max(candidates, key=utility)


def majority(row):
    """Three independent initial/retry/retry samples; deterministic first-sample tie."""
    tree = row['tree']
    nodes = [tree[''], tree['retry'], tree['retry/retry']]
    from .core import canonical_answer
    answers = [canonical_answer(n.get('answer', '')) if n.get('valid', True) else '' for n in nodes]
    counts = {a: answers.count(a) for a in answers if a}
    selected = max(counts, key=lambda a: (counts[a], -answers.index(a))) if counts else ''
    valid = bool(selected)
    tokens_in = sum(n.get('input_tokens', 0) for n in nodes)
    tokens_out = sum(n.get('output_tokens', 0) for n in nodes)
    return {'id': row['id'], 'correct': int(correct({'answer': selected, 'valid': valid}, row['answer'])), 'initial_correct': int(correct(nodes[0], row['answer'])), 'answer': selected, 'input_tokens': tokens_in, 'output_tokens': tokens_out, 'tokens': tokens_in + tokens_out, 'solver_latency_seconds': sum(n.get('latency_seconds', 0) for n in nodes), 'judge_latency_seconds': 0, 'judge_calls': 0, 'actions': ['retry', 'retry', 'majority'], 'path': 'retry/retry'}
