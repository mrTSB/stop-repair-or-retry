"""Small deterministic numerical tool; parses expressions without executing code."""
import ast
import math
import re

FUNCTIONS = {name: getattr(math, name) for name in
    ('sqrt', 'exp', 'log', 'log10', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
     'sinh', 'cosh', 'tanh', 'degrees', 'radians')}
FUNCTIONS['abs'] = abs
CONSTANTS = {'pi': math.pi, 'e': math.e}
MAX_EXPRESSIONS = 8


def calculate(expression):
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError('Expression length limit')
    tree = ast.parse(expression.strip(), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 96:
        raise ValueError('Expression complexity limit')

    def checked(value):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError('Finite bounded real value required')
        return float(value)

    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant): return checked(node.value)
        if isinstance(node, ast.Name) and node.id in CONSTANTS: return CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): result = left+right
            elif isinstance(node.op, ast.Sub): result = left-right
            elif isinstance(node.op, ast.Mult): result = left*right
            elif isinstance(node.op, ast.Div): result = left/right
            elif isinstance(node.op, ast.Pow):
                if abs(right) > 100: raise ValueError('Exponent limit')
                result = left**right
            else: raise ValueError('Unsupported arithmetic operator')
            return checked(result)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
            return checked(FUNCTIONS[node.func.id](visit(node.args[0])))
        raise ValueError('Only numerical expressions and listed unary functions are allowed')

    return visit(tree)


def evaluate_text(text):
    expressions = re.findall(r'<calc>(.*?)</calc>', text, flags=re.DOTALL)[:MAX_EXPRESSIONS]
    results = []
    for expression in expressions:
        expression = expression.strip()
        try: result = {'expression': expression[:256], 'value': format(calculate(expression), '.12g'), 'ok': True}
        except (ValueError, SyntaxError, ArithmeticError, RecursionError, TypeError):
            result = {'expression': expression[:256], 'error': 'Invalid, unsupported or out-of-bounds expression', 'ok': False}
        results.append(result)
    return results
