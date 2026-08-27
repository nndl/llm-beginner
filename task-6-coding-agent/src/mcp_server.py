"""A small, workspace-confined MCP tool server for coding tasks."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def list_tools() -> list[dict[str, Any]]:
    """Return MCP-compatible tool metadata without starting the server."""
    """
    返回 MCP 工具的元数据。

    该函数本身不会启动 MCP Server，只负责描述当前提供了哪些工具、
    每个工具的功能以及输入参数格式，方便外部 Agent 或测试代码了解
    当前工具服务器支持的能力。
    """
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
    """
    RepositoryTools 负责实现实际的代码操作。

    该类不负责决定“什么时候调用哪个工具”，
    而是负责执行具体操作，例如读文件、写文件、运行测试和 Git 操作。

    所有文件操作都会经过 _path() 检查，从而保证操作范围不会
    超出指定的代码仓库目录。
    """

    def __init__(self, repo_path: str):
        self.root = Path(repo_path).resolve()    # 将仓库路径转换为绝对路径并规范化
        if not self.root.is_dir():   # 如果指定的仓库目录不存在，则直接报错
            raise ValueError(f"Repository does not exist: {repo_path}")

    def _path(self, value: str) -> Path:
        """
        将用户/Agent 提供的相对路径转换为仓库内的绝对路径。

        这里最重要的作用是进行路径安全检查，防止 Agent 通过
        ../ 等路径访问仓库之外的文件。
        """
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError("Absolute paths are not allowed")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("Path escapes the repository")
        return resolved

    @staticmethod
    def _result(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        """
        将 subprocess 的执行结果统一转换为字典。

        这样 run_tests、git_diff 和 git_apply 都可以使用统一的
        返回格式，方便 MCP Client / Agent 解析。
        """
        return {"ok": process.returncode == 0, "returncode": process.returncode,
                "output": (process.stdout + process.stderr)[-8000:]}

    def read_file(self, path: str) -> dict[str, Any]:
        """
        读取仓库内的 UTF-8 文本文件。

        参数：
            path：相对于仓库根目录的文件路径。

        返回：
            成功时返回文件内容；
            失败时返回错误信息。
        """
        try:
            return {"ok": True, "content": self._path(path).read_text(encoding="utf-8")}
        except (OSError, ValueError, UnicodeError) as error:
            return {"ok": False, "error": str(error)}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """
        将 UTF-8 文本写入仓库内的文件。

        如果目标文件的父目录不存在，会自动创建。
        """
        try:
            target = self._path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
            return {"ok": True, "path": str(target.relative_to(self.root))}
        except (OSError, ValueError, UnicodeError) as error:
            return {"ok": False, "error": str(error)}

    def run_tests(self) -> dict[str, Any]:
        """
        在仓库根目录运行 pytest 测试。

        实际执行的命令相当于：

            python -m pytest -q

        最长允许运行 60 秒，防止测试程序无限运行。
        """
        try:
            return self._result(subprocess.run(
                ["python", "-m", "pytest", "-q"], cwd=self.root, text=True,
                capture_output=True, timeout=60, check=False))
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": str(error)}

    def git_diff(self) -> dict[str, Any]:
        """
        查看当前 Git 仓库中的未提交修改。

        实际执行：

            git diff -- .
        """
        try:
            return self._result(subprocess.run(
                ["git", "diff", "--", "."], cwd=self.root, text=True,
                capture_output=True, timeout=20, check=False))
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": str(error)}

    def git_apply(self, patch: str) -> dict[str, Any]:
        """
        将 Agent 提供的 unified diff 补丁应用到 Git 仓库。

        patch 不需要先保存成临时文件，而是直接通过 stdin
        传给 git apply。
        """
        try:
            return self._result(subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"], cwd=self.root, input=patch,
                text=True, capture_output=True, timeout=20, check=False))
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": str(error)}


def main() -> None:
    """Run via stdio when the optional official MCP SDK is installed."""
    """
    启动 MCP Server。

    该 Server 使用 stdio 与 MCP Client 通信。
    只有安装官方 MCP Python SDK 后才能正常运行。
    """
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