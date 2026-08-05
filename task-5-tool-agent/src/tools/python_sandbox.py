"""Teaching-only Python sandbox; never use it for untrusted code."""
import ast
import builtins
import contextlib
import io
import multiprocessing as mp

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": "Run a short Python calculation and return its printed output.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "len",
    "list", "map", "max", "min", "print", "range", "reversed", "round", "set", "sorted",
    "str", "sum", "tuple", "zip")}


def _reject_unsafe(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal)):
            raise ValueError("Imports and scope-changing statements are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Dunder names are not allowed")


def _worker(code: str, queue) -> None:
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(code, "<sandbox>", "exec"), {"__builtins__": _BUILTINS}, {})
        queue.put((True, output.getvalue()))
    except Exception as error:
        queue.put((False, f"{type(error).__name__}: {error}"))


def run(args: dict) -> str:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("'code' must be a non-empty string")
    if len(code) > 4_000:
        raise ValueError("Code is too long")
    _reject_unsafe(code)
    queue = mp.Queue()
    process = mp.Process(target=_worker, args=(code, queue))
    process.start()
    process.join(2)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Sandbox timed out after 2 seconds")
    if queue.empty():
        raise RuntimeError("Sandbox exited without a result")
    ok, result = queue.get()
    if not ok:
        raise RuntimeError(result)
    return result.rstrip() or "(no output)"