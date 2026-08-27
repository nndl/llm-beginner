"""A small, expression-only calculator."""
import ast
import math
import operator
# ============================================================
# 工具 Schema
# ============================================================
TOOL_SCHEMA = {
    "type": "function",         # 表示这是一个“函数工具（Function Tool）”。
    "function": {
        "name": "calculator",       # 工具名称。后面的 Agent 可以通过这个名字知道应该调用 calculator。
        "description": "Evaluate a mathematical expression using arithmetic and safe math functions.",      # 工具描述。告诉大模型：这个工具用于计算数学表达式，支持基本算术运算以及安全的数学函数。
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Math expression"}}, # expression 参数必须是字符串。参数说明：需要计算的数学表达式。
            "required": ["expression"],     # expression 是必填参数。调用 calculator 时不能不提供 expression。
            "additionalProperties": False,  # 不允许传入 Schema 中没有定义的额外参数。
        },
    },
}
# ============================================================
# 支持的二元运算
# ============================================================
_BINARY = {ast.Add: operator.add,  # 加法：a + b
           ast.Sub: operator.sub,   # 减法：a - b
           ast.Mult: operator.mul,  # 乘法：a * b
           ast.Div: operator.truediv, # 普通除法：a / b
           ast.FloorDiv: operator.floordiv, # 整除：a // b
           ast.Pow: operator.pow,  # 幂运算：a ** b
           ast.Mod: operator.mod}  # 取模：a % b
# ============================================================
# 支持的一元运算
# ============================================================
_UNARY = {ast.UAdd: operator.pos,  # 正号：+a
          ast.USub: operator.neg}  # 负号：-a
# ============================================================
# 允许使用的数学函数和数学常量
# ============================================================
_NAMES = {
    name: getattr(math, name)
    for name in (
        "sqrt",   # 平方根，例如 sqrt(16) = 4
        "sin",    # 正弦
        "cos",    # 余弦
        "tan",    # 正切
        "log",    # 自然对数
        "log10",  # 以 10 为底的对数
        "ceil",   # 向上取整
        "floor",  # 向下取整
        "fabs"    # 浮点数绝对值
    )
}
_NAMES.update({
    "pi": math.pi,  # 圆周率 π
    "e": math.e,    # 自然常数 e
    "abs": abs,     # 绝对值，例如 abs(-5) = 5
    "round": round  # 四舍五入，例如 round(3.14) = 3
})
# ============================================================
# 核心计算函数
# ============================================================
def _evaluate(node):
    """
    递归计算 AST 节点。

    这里不是直接执行 Python 代码，而是：
    1. 先把表达式解析成 AST；
    2. 检查 AST 中出现的节点类型；
    3. 只有我们明确允许的节点才执行；
    4. 其他节点直接报错。

    这样可以避免用户通过表达式执行任意 Python 代码。
    """

    # --------------------------------------------------------
    # 1. 处理整个表达式
    # --------------------------------------------------------
    if isinstance(node, ast.Expression):
        # ast.Expression 表示整个数学表达式。
        #
        # 例如：
        #     1 + 2
        #
        # AST 大致结构为：
        #
        # Expression
        #     └── BinOp
        #
        # 真正需要计算的是 Expression 下面的 body。
        return _evaluate(node.body)
    # --------------------------------------------------------
    # 2. 处理数字常量
    # --------------------------------------------------------
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        # 只允许整数和浮点数。
        #
        # 例如：
        #     123
        #     3.14
        #
        # 都属于 Constant。
        return node.value
    # --------------------------------------------------------
    # 3. 处理二元运算
    # --------------------------------------------------------
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        # ast.BinOp 表示二元运算，例如：
        #
        #     1 + 2
        #     3 * 4
        #     10 / 2
        #
        # type(node.op) 用来判断具体是什么运算。
        #
        # 例如：
        #     +  -> ast.Add
        #     *  -> ast.Mult
        #
        # 然后从 _BINARY 中找到对应的 Python 运算函数。
        # 例如：
        #
        # 1 + 2
        #
        # 最终相当于：
        #
        # operator.add(1, 2)
        #
        # 得到：
        #
        # 3
        return _BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    # --------------------------------------------------------
    # 4. 处理一元运算
    # --------------------------------------------------------
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        # ast.UnaryOp 表示只有一个操作数的运算。
        #
        # 例如：
        #     -5
        #     +5
        return _UNARY[type(node.op)](_evaluate(node.operand))
    # --------------------------------------------------------
    # 5. 处理名称
    # --------------------------------------------------------
    if isinstance(node, ast.Name) and node.id in _NAMES:
        # ast.Name 表示一个名称。
        #
        # 例如：
        #     pi
        #     e
        #     sqrt
        #
        # 这里只允许 _NAMES 中已经明确规定的名称。
        #
        # 因此：
        #     pi      ✓
        #     sqrt    ✓
        #     abc     ✗
        #     os      ✗
        return _NAMES[node.id]
    # --------------------------------------------------------
    # 6. 处理函数调用
    # --------------------------------------------------------
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _NAMES:
        # ast.Call 表示函数调用。
        #
        # 例如：
        #     sqrt(16)
        #
        # node.func.id 就是：
        #     "sqrt"
        #
        # 然后从 _NAMES 中找到真正的函数。
        function = _NAMES[node.func.id]
        if not callable(function) or node.keywords:
            # callable() 判断这个名称是不是一个可以调用的函数。
            #
            # 例如：
            #     sqrt  -> 可以调用
            #     pi    -> 不能调用
            #
            # node.keywords 表示是否使用了关键字参数。
            #
            # 例如：
            #     round(number=3.14)
            #
            # node.keywords 就不是空的。
            #
            # 这里直接禁止关键字参数，只允许普通的位置参数。
            raise ValueError("Unsupported function call")
        # 递归计算函数的所有参数。
        #
        # 例如：
        #
        #     sqrt(16 + 9)
        #
        # 会先计算：
        #
        #     16 + 9 = 25
        #
        # 再执行：
        #
        #     sqrt(25) = 5
        return function(*[_evaluate(argument) for argument in node.args])
    # --------------------------------------------------------
    # 7. 其他所有 AST 节点都拒绝
    # --------------------------------------------------------
    raise ValueError("Only arithmetic and approved math functions are allowed")

# ============================================================
# 对外提供的工具入口
# ============================================================
def run(args: dict) -> str:
    """
    calculator 工具的统一入口。

    参数：
        args:
            一个字典，例如：
            {"expression": "sqrt(16) + 2"}

    返回：
        字符串形式的计算结果。
    """

    # --------------------------------------------------------
    # 1. 获取 expression 参数
    # --------------------------------------------------------
    # 从调用参数中获取 expression。
    #
    # 例如：
    #
    # args = {
    #     "expression": "1 + 2"
    # }
    #
    # expression 就是：
    #
    # "1 + 2"
    expression = args.get("expression")
    # --------------------------------------------------------
    # 2. 检查 expression 是否合法
    # --------------------------------------------------------
    if not isinstance(expression, str) or not expression.strip():
        # expression 必须：
        # 1. 是字符串；
        # 2. 不能是空字符串；
        # 3. 不能只有空格。
        raise ValueError("'expression' must be a non-empty string")
    # --------------------------------------------------------
    # 3. 限制表达式长度
    # --------------------------------------------------------
    if len(expression) > 500:
        # 限制最大长度为 500 个字符。
        #
        # 主要目的是避免输入过长的表达式，
        # 降低资源消耗和潜在的异常情况。
        raise ValueError("Expression is too long")
    # --------------------------------------------------------
    # 4. 将字符串解析为 AST
    # --------------------------------------------------------
    result = _evaluate(ast.parse(expression, mode="eval"))
    # ast.parse(..., mode="eval")：
    #
    # 把字符串形式的数学表达式解析成 AST。
    #
    # 例如：
    #
    #     "1 + 2 * 3"
    #
    # 会被解析成类似：
    #
    #             +
    #            / \
    #           1   *
    #              / \
    #             2   3
    #
    # 然后交给 _evaluate() 递归计算。

    # --------------------------------------------------------
    # 5. 将计算结果转换成字符串返回
    # --------------------------------------------------------
    return str(result)