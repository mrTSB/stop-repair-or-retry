"""Data schema, scoring, and fitted two-state model. No external dependencies."""
from __future__ import annotations

import json
import math
import re
import ast
from fractions import Fraction
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ACTIONS = ("repair", "retry")
OBS = ("low", "medium", "high")


def normalize_answer(value):
    """Conservative exact matching, not a symbolic math equivalence checker."""
    value = str(value).strip().lower().replace(",", "")
    value = re.sub(r"\s+", " ", value).strip().strip('$').strip()
    if value.endswith('.') and value != '.':
        value = value[:-1].rstrip()
    return value


def canonical_answer(value):
    """Normalize exact numbers without pulling a number out of explanatory text."""
    text = normalize_answer(value).replace('$', '').strip()
    # GSM8K gold labels express a requested percentage as a number, e.g. 52.
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%', text):
        text = text[:-1].strip()
    # GSM8K references are numeric, but models often append a simple unit.
    # Do not extract embedded numbers or interpret compound expressions.
    unit = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?(?:\d+(?:\.\d*)?|\.\d+))?)\s+[a-z]+(?:\s+[a-z]+){0,3}', text)
    if unit:
        text = unit.group(1)
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?(?:\d+(?:\.\d*)?|\.\d+))?', text):
        try:
            return str(Fraction(text))
        except (ValueError, ZeroDivisionError):
            pass
    # Accept plain arithmetic expressions, and a final expression after '='.
    # Do not execute generated code, accept names, or extract numbers from prose.
    expression = text.rsplit('=', 1)[-1].strip()
    if len(expression) <= 256:
        try:
            def evaluate(node):
                if isinstance(node, ast.Expression): return evaluate(node.body)
                if isinstance(node, ast.Constant) and type(node.value) in (int, float): return Fraction(str(node.value))
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    n = evaluate(node.operand)
                    return n if isinstance(node.op, ast.UAdd) else -n
                if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    a, b = evaluate(node.left), evaluate(node.right)
                    if isinstance(node.op, ast.Add): return a + b
                    if isinstance(node.op, ast.Sub): return a - b
                    if isinstance(node.op, ast.Mult): return a * b
                    return a / b
                raise ValueError('Unsupported expression')
            return str(evaluate(ast.parse(expression, mode='eval')))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
            pass
    return text


def correct(node, gold):
    return bool(node.get("valid", True)) and canonical_answer(node.get("answer", "")) == canonical_answer(gold) and bool(canonical_answer(gold))


def confidence(node):
    c = str(node.get("confidence", "low")).lower()
    return c if c in OBS else "low"


def node_cost(node):
    return int(node.get("input_tokens", 0)) + int(node.get("output_tokens", 0))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def walk(tree, path=""):
    yield path, tree[path]
    for action in ACTIONS:
        child = path + ("/" if path else "") + action
        if child in tree:
            yield from walk(tree, child)


@dataclass
class Model:
    prior: float
    trans: dict
    obs: dict
    costs: dict
    initial_obs: dict

    @classmethod
    def fit(cls, rows, alpha=1.0):
        # Conditional observation model P(confidence | next correctness, action).
        transitions = defaultdict(Counter)
        observations = defaultdict(Counter)
        starts = defaultdict(Counter)
        costs = defaultdict(list)
        ncorrect = 0
        for row in rows:
            tree, gold = row["tree"], row["answer"]
            initial = tree[""]
            state = int(correct(initial, gold))
            ncorrect += state
            starts[state][confidence(initial)] += 1
            for path, node in walk(tree):
                state = int(correct(node, gold))
                for action in ACTIONS:
                    child = path + ("/" if path else "") + action
                    if child not in tree:
                        continue
                    nxt = tree[child]
                    next_state = int(correct(nxt, gold))
                    transitions[(action, state)][next_state] += 1
                    observations[(action, next_state)][confidence(nxt)] += 1
                    costs[(action, state)].append(node_cost(nxt))
        if not rows:
            raise ValueError("Training data is empty")
        trans = {(a, s): (transitions[(a, s)][1] + alpha) / (sum(transitions[(a, s)].values()) + 2 * alpha) for a in ACTIONS for s in (0, 1)}
        obs = {(a, s, o): (observations[(a, s)][o] + alpha) / (sum(observations[(a, s)].values()) + len(OBS) * alpha) for a in ACTIONS for s in (0, 1) for o in OBS}
        initial_obs = {(s, o): (starts[s][o] + alpha) / (sum(starts[s].values()) + len(OBS) * alpha) for s in (0, 1) for o in OBS}
        # State-specific costs are averaged when no samples exist; zero is unsafe.
        all_cost = [node_cost(n) for row in rows for path, n in walk(row["tree"]) if path]
        fallback = sum(all_cost) / len(all_cost) if all_cost else 0
        mean_costs = {(a, s): sum(costs[(a, s)]) / len(costs[(a, s)]) if costs[(a, s)] else fallback for a in ACTIONS for s in (0, 1)}
        return cls((ncorrect + alpha) / (len(rows) + 2 * alpha), trans, obs, mean_costs, initial_obs)

    def initial_belief(self, o):
        x = self.prior * self.initial_obs[(1, o)]
        y = (1 - self.prior) * self.initial_obs[(0, o)]
        return x / (x + y)

    def update(self, belief, action, observation):
        predicted = (1 - belief) * self.trans[(action, 0)] + belief * self.trans[(action, 1)]
        x = predicted * self.obs[(action, 1, observation)]
        y = (1 - predicted) * self.obs[(action, 0, observation)]
        return x / (x + y), x + y

    def values(self, belief, remaining, penalty):
        values = {"stop": belief}
        if remaining <= 0:
            return values
        for action in ACTIONS:
            cost = (1 - belief) * self.costs[(action, 0)] + belief * self.costs[(action, 1)]
            future = 0.0
            for o in OBS:
                posterior, prob = self.update(belief, action, o)
                future += prob * max(self.values(posterior, remaining - 1, penalty).values())
            values[action] = future - penalty * cost / 1000
        return values
