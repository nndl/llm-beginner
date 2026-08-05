"""A read-only, isolated-context code inspection subagent."""
from __future__ import annotations

from src.mcp_server import RepositoryTools


class CodeSearchSubagent:
    def run(self, tools: RepositoryTools, issue: str) -> dict:
        messages = [{"role": "user", "content": issue}]
        result = tools.read_file("calculator.py")
        messages.append({"role": "tool", "content": str(result)})
        return {"summary": "calculator.add returns subtraction; change `a - b` to `a + b`.",
                "messages": messages, "source": result}