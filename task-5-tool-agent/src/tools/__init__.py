"""Tools available to the ReAct agent."""

from . import calculator, file_search, python_sandbox, wiki

TOOLS = {
    "calculator": calculator.run,
    "python_sandbox": python_sandbox.run,
    "file_search": file_search.run,
    "wiki": wiki.run,
}