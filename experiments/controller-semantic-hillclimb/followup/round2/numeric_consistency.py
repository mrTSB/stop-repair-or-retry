"""Cheap, reference-free numeric consistency observations for the POMDP.

These are noisy observations of written arithmetic, not correctness labels.
Only a small arithmetic AST is interpreted; model text is never executed.
"""

import ast
import math
import re


def number(expression):
    if len(expression) > 100:
        return None
    try:
        tree = ast.parse(expression.strip(), mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 32:
            return None
        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = float(node.value)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp):
                a, b = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add): value = a + b
                elif isinstance(node.op, ast.Sub): value = a - b
                elif isinstance(node.op, ast.Mult): value = a * b
                elif isinstance(node.op, ast.Div): value = a / b
                elif isinstance(node.op, ast.Pow) and abs(b) <= 8: value = a ** b
                else: raise ValueError('Unsupported operation')
            else:
                raise ValueError('Non-numeric expression')
            if not math.isfinite(value) or abs(value) > 1e18:
                raise ValueError('Unbounded expression')
            return value
        return visit(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError, RecursionError):
        return None


def observations(text):
    text = text.replace('−', '-').replace('×', '*').replace('÷', '/').replace('^', '**')
    text = text.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/').replace('$', '')
    text = re.sub(r'\\frac\{([-+\d.]+)\}\{([-+\d.]+)\}', r'(\1/\2)', text)
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', text)
    text = re.sub(r'(?<=[\d)])\s+of\s+(?=[\d(])', ' * ', text)
    text = re.sub(r'(?<=\d)\s*[xX]\s*(?=\d)', '*', text)
    checked, matches, mismatches, failures = 0, 0, 0, 0
    relative = []
    last = 0.
    for line in text.splitlines():
        for equal in re.finditer('=', line):
            left = re.search(r'([-+(\d.][\d .()+*/-]*)\s*$', line[:equal.start()])
            right = re.match(r'\s*([-+(\d.][\d .()+*/-]*)', line[equal.end():])
            if not left or not right:
                continue
            lhs, rhs = left.group(1).strip().rstrip('.'), right.group(1).strip().rstrip('.')
            if not any(c in lhs for c in '+-*/'):
                continue
            a, b = number(lhs), number(rhs)
            if a is None or b is None:
                failures += 1
                continue
            checked += 1
            error = abs(a - b) / max(1., abs(a), abs(b))
            relative.append(min(1., error))
            # Written rounded decimals are allowed. Intentional integer rounding
            # can still disagree and is learned as a noisy feature, never a veto.
            decimals = re.search(r'\.([0-9]+)$', rhs)
            rounding = .5 * 10 ** -len(decimals.group(1)) if decimals else 1e-8
            same = abs(a - b) <= max(rounding + 1e-8, 1e-6 * max(abs(a), abs(b)))
            matches += int(same)
            mismatches += int(not same)
            last = 1. if same else -1.
    return [math.log1p(checked), math.log1p(matches), math.log1p(mismatches),
            math.log1p(failures), float(mismatches > 0), last,
            max(relative, default=0.), sum(relative) / max(1, len(relative))]
