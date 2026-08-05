"""A small, expression-only calculator."""
import ast
import math
import operator

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression using arithmetic and safe math functions.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Math expression"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_NAMES = {name: getattr(math, name) for name in ("sqrt", "sin", "cos", "tan", "log", "log10", "ceil", "floor", "fabs")}
_NAMES.update({"pi": math.pi, "e": math.e, "abs": abs, "round": round})


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _NAMES:
        function = _NAMES[node.func.id]
        if not callable(function) or node.keywords:
            raise ValueError("Unsupported function call")
        return function(*[_evaluate(argument) for argument in node.args])
    raise ValueError("Only arithmetic and approved math functions are allowed")


def run(args: dict) -> str:
    expression = args.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("'expression' must be a non-empty string")
    if len(expression) > 500:
        raise ValueError("Expression is too long")
    result = _evaluate(ast.parse(expression, mode="eval"))
    return str(result)