"""A small, workspace-confined MCP tool server for coding tasks."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def list_tools() -> list[dict[str, Any]]:
    """Return MCP-compatible tool metadata without starting the server."""
    return [
        {"name": "read_file", "description": "Read a UTF-8 file inside the repository.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "write_file", "description": "Write a UTF-8 file inside the repository.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        {"name": "run_tests", "description": "Run pytest in the repository.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "git_diff", "description": "Show uncommitted changes in the repository.",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "git_apply", "description": "Apply a unified diff inside the repository.",
         "input_schema": {"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}},
    ]


class RepositoryTools:
    """Stateless tool implementations with a strict repository boundary."""

    def __init__(self, repo_path: str):
        self.root = Path(repo_path).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repository does not exist: {repo_path}")

    def _path(self, value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError("Absolute paths are not allowed")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("Path escapes the repository")
        return resolved

    @staticmethod
    def _result(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        return {"ok": process.returncode == 0, "returncode": process.returncode,
                "output": (process.stdout + process.stderr)[-8000:]}

    def read_file(self, path: str) -> dict[str, Any]:
        try:
            return {"ok": True, "content": self._path(path).read_text(encoding="utf-8")}
        except (OSError, ValueError, UnicodeError) as error:
            return {"ok": False, "error": str(error)}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        try:
            target = self._path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
            return {"ok": True, "path": str(target.relative_to(self.root))}
        except (OSError, ValueError, UnicodeError) as error:
            return {"ok": False, "error": str(error)}

    def run_tests(self) -> dict[str, Any]:
        try:
            return self._result(subprocess.run(
                ["python", "-m", "pytest", "-q"], cwd=self.root, text=True,
                capture_output=True, timeout=60, check=False))
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": str(error)}

    def git_diff(self) -> dict[str, Any]:
        try:
            return self._result(subprocess.run(
                ["git", "diff", "--", "."], cwd=self.root, text=True,
                capture_output=True, timeout=20, check=False))
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": str(error)}

    def git_apply(self, patch: str) -> dict[str, Any]:
        try:
            return self._result(subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"], cwd=self.root, input=patch,
                text=True, capture_output=True, timeout=20, check=False))
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": str(error)}


def main() -> None:
    """Run via stdio when the optional official MCP SDK is installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise SystemExit("Install mcp to run the stdio server: pip install mcp") from error
    import os

    repository = os.environ.get("CODING_AGENT_REPO", ".")
    tools = RepositoryTools(repository)
    server = FastMCP("local-coding-tools")

    @server.tool()
    def read_file(path: str) -> dict[str, Any]:
        return tools.read_file(path)

    @server.tool()
    def write_file(path: str, content: str) -> dict[str, Any]:
        return tools.write_file(path, content)

    @server.tool()
    def run_tests() -> dict[str, Any]:
        return tools.run_tests()

    @server.tool()
    def git_diff() -> dict[str, Any]:
        return tools.git_diff()

    @server.tool()
    def git_apply(patch: str) -> dict[str, Any]:
        return tools.git_apply(patch)

    server.run()


if __name__ == "__main__":
    main()