"""Check complete arithmetic equalities already produced by the compact solver."""

from fractions import Fraction
import re

from study_data import FINAL
from scoring import canonical_answer


RATIONAL = re.compile(r'-?\d+(?:/[1-9]\d*)?')
SCALAR = re.compile(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+)?)\s*(?:[a-zA-Z][a-zA-Z ]*)?\.?\s*')
EXPRESSION = re.compile(r'[0-9.()+*/\- \t]+')
ROUNDING = re.compile(r'\b(?:round(?:ed|ing)?|nearest|approximately|estimate[d]?|about)\b', re.IGNORECASE)


def normalize(expression):
    for _ in range(4):
        reduced = re.sub(r'\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', expression)
        if reduced == expression:
            break
        expression = reduced
    expression = re.sub(r'\\(?:text|mathrm)\{[^{}]*\}', '', expression)
    expression = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', expression)
    for old, new in ((r'\times', '*'), (r'\cdot', '*'), (r'\div', '/'), ('×', '*'), ('÷', '/'),
                     ('−', '-'), (r'\(', ''), (r'\)', ''), ('$', ''), ('`', '')):
        expression = expression.replace(old, new)
    return expression


def evaluate_expression(part):
    part = part.strip()
    if len(part) > 160 or not EXPRESSION.fullmatch(part):
        return None
    canonical = canonical_answer(part)
    if not RATIONAL.fullmatch(canonical):
        return None
    return Fraction(canonical)


def correction(row):
    """Return an exact alternative when the last equality uniquely implies it."""
    if row['domain'] != 'math':
        return ''
    response = row['compact']
    given = canonical_answer(response['answer']) if response['answer'] else ''
    if not RATIONAL.fullmatch(given):
        return ''
    final = FINAL.search(response['text'])
    if not final:
        return ''
    scratch = normalize(response['text'][:final.start()])
    equations = [part.strip() for part in re.split(r'[;\n]+|(?<=[?!])\s+', scratch) if '=' in part]
    if not equations:
        return ''
    equation = equations[-1]
    if any(mark in equation for mark in ('^', '%', '≈', '<', '>')) or ROUNDING.search(equation + row['question']):
        return ''
    parts = equation.split('=')
    reported = SCALAR.fullmatch(parts[-1])
    if not reported or canonical_answer(reported[1]) != given:
        return ''
    prefix = parts[0].rsplit(':', 1)[-1].strip()
    # Allow a prose label before the first arithmetic expression, while requiring
    # the entire expression and all intervening equality terms to parse.
    label = re.match(r'^[A-Za-z][A-Za-z _\-]*\s+', prefix)
    if label:
        prefix = prefix[label.end():]
    terms = [prefix, *parts[1:-1]] if prefix and re.search(r'\d', prefix) else parts[1:-1]
    if not terms or not re.search(r'\d[^\d=]*[+*/-]|\d\s*\)', terms[0]):
        return ''
    values = [evaluate_expression(term) for term in terms]
    exact, original = values[0], Fraction(given)
    if exact is None or any(value != exact for value in values) or exact == original:
        return ''
    if exact * original < 0 or abs(exact - original) <= Fraction(1, 100):
        return ''
    return str(exact)
