"""Compute decision features using only visible question and solver history."""

from dataclasses import dataclass
import math
import re

from collect import canonical_answer
from scoring import parse


@dataclass(frozen=True)
class Attempt:
    answer: str
    confidence: str
    valid: bool
    truncated: bool
    reasoning: str
    output_tokens: int


@dataclass(frozen=True)
class Context:
    domain: str
    subject: str
    question: str
    history: tuple[Attempt, ...]
    last_action: str


def observed(row, path):
    attempts = []
    for prefix in visited(path):
        node = row['tree'][prefix]
        obj, _ = parse(node['text'])
        valid = node['answer_parse_valid']
        attempts.append(Attempt(
            canonical_answer(node['answer']) if valid else '', node['confidence'], valid,
            node['finish_reason'] == 'length', str(obj.get('reasoning', '')),
            node['output_tokens'],
        ))
    return Context(row['domain'], row['subject'], row['question'], tuple(attempts), path.split('/')[-1])


def visited(path):
    parts = path.split('/') if path else []
    return tuple('/'.join(parts[:i]) for i in range(len(parts) + 1))


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
EQUATION = re.compile(rf'(?<![\d.])([\d.()+*/\- \t]+)\s*=\s*({NUMBER})(?![\d.])')
NUMERIC = re.compile(r'-?\d+(?:/\d+)?')
OPTION = re.compile(r'\b(?:option|choice|answer(?:\s+is)?)\s*\**\(?([A-J])\b', re.IGNORECASE)


def arithmetic(reasoning, answer):
    text = reasoning.replace(r'\times', '*').replace(r'\cdot', '*').replace(r'\div', '/')
    text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
    for source, target in [('×', '*'), ('÷', '/'), ('−', '-'), ('$', ''), (r'\(', ''), (r'\)', '')]:
        text = text.replace(source, target)
    checks = []
    for left, right in EQUATION.findall(text):
        canonical = canonical_answer(left.strip())
        if NUMERIC.fullmatch(canonical) and not NUMERIC.fullmatch(left.strip()):
            checks.append((canonical == canonical_answer(right), canonical == answer))
    return [bool(checks), any(not correct for correct, _ in checks), bool(checks and checks[-1][1])]


def basic(context):
    attempt = context.history[-1]
    return [
        context.domain == 'math', context.last_action == 'retry', attempt.valid,
        attempt.confidence == 'high', attempt.confidence == 'medium', attempt.truncated,
    ]


def trace(context):
    attempt = context.history[-1]
    reasoning = attempt.reasoning.lower()
    prior = context.history[-2] if len(context.history) > 1 else None
    numeric = NUMERIC.fullmatch(attempt.answer)
    value = int(attempt.answer.split('/')[0]) if numeric and len(attempt.answer) <= 32 else 0
    options = OPTION.findall(attempt.reasoning)
    return [
        math.log1p(len(context.question)), math.log1p(attempt.output_tokens),
        bool(prior and attempt.answer and prior.answer == attempt.answer),
        bool(prior and prior.confidence == 'high'),
        math.log1p(prior.output_tokens) if prior else 0,
        bool(re.search(r'\b(?:mistake|error|incorrect|wrong)\b', reasoning)),
        bool(re.search(r'\b(?:perhaps|probably|approximate\w*|round\w*|uncertain)\b', reasoning)),
        bool(re.search(r'\b(?:previous|original|initial)\b', reasoning)),
        bool(re.search(r'\b(?:correct|confirm\w*|indeed)\b', reasoning)),
        context.subject == 'biology', context.subject == 'physics',
        bool(options), bool(options and options[-1].lower() == attempt.answer),
        math.log1p(abs(value)), bool(numeric and '/' in attempt.answer),
        *arithmetic(attempt.reasoning, attempt.answer),
    ]


def critic_features(context, signal):
    attempt = context.history[-1]
    signal_answer = canonical_answer(signal['answer'])
    evidence = signal['evidence']
    if context.domain == 'science':
        support = evidence.get(attempt.answer.upper(), 0.0)
        prob = evidence.get(signal_answer.upper(), 0.0)
    else:
        prob = math.exp(evidence.get('mean_logprob') or 0) if signal_answer else 0.0
        support = prob if signal_answer == attempt.answer else 0.0
    return [
        bool(signal_answer), bool(signal_answer and signal_answer == attempt.answer),
        prob, support, bool(len(context.history) > 1 and signal_answer and signal_answer == context.history[-2].answer),
    ]
