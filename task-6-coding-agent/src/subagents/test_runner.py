"""A test-only isolated-context subagent."""
from __future__ import annotations

from src.mcp_server import RepositoryTools


class TestRunnerSubagent:
    def run(self, tools: RepositoryTools) -> dict:
        messages = [{"role": "user", "content": "Run the repository test suite."}]
        result = tools.run_tests()
        messages.append({"role": "tool", "content": str(result)})
        summary = "Tests passed." if result.get("ok") else "Tests failed; inspect the output."
        return {"summary": summary, "messages": messages, "result": result}