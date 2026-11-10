"""Frozen answer parsing/scoring for this dataset; no controller implementation."""
import re,ast,json
from fractions import Fraction

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

def parse(text):
    decoder=json.JSONDecoder()
    for m in re.finditer(r'\{',text):
        try:
            obj,_=decoder.raw_decode(text[m.start():])
            if isinstance(obj,dict):return obj,'json'
        except json.JSONDecodeError:pass
    obj={}
    for field in ('answer','confidence'):
        match=re.search(r'"'+field+r'"\s*:\s*(?:"([^"\r\n]*)"|([a-zA-Z]+|[+-]?(?:\d+(?:\.\d+)?|\.\d+)))\s*(?=,|\}|\s*"(?:answer|confidence)"\s*:)',text)
        if match:obj[field]=match.group(1) if match.group(1) is not None else match.group(2)
    return obj,'explicit_field_recovery' if obj else 'unparseable'
