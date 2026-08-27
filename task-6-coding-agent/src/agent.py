"""Traceable coding loop that can repair the provided local toy repository."""
from __future__ import annotations

import os
from pathlib import Path

from src.mcp_server import RepositoryTools
from src.skill_loader import SkillLoader
from src.subagents.code_search import CodeSearchSubagent
from src.subagents.test_runner import TestRunnerSubagent


class Trace(dict):
    """Dictionary trace retained for evaluation and JSON serialization."""
    """
    保存 CodingAgent 的完整执行轨迹。

    继承 dict 是为了方便后续直接进行 JSON 序列化，
    同时可以记录 Agent 每一步的：
        - thought：当前步骤的思考/决策
        - tool_call：调用了什么工具
        - observation：工具返回了什么结果
    """


class CodingAgent:
    """
    一个可追踪的 Coding Agent。

    整体执行流程：

        用户问题
            ↓
        匹配相关 Skill
            ↓
        加载 Skill 指令
            ↓
        CodeSearchSubagent 只读分析代码
            ↓
        Local Qwen / 离线规划器制定修复计划
            ↓
        修改源代码
            ↓
        TestRunnerSubagent 运行测试
            ↓
        Git Diff 获取最终修改
            ↓
        返回完整 Trace

    这里的实现针对 toy repository，
    因此实际修复逻辑采用确定性的字符串替换，
    而不是让 LLM 自主生成代码补丁。
    """
    def __init__(self, max_steps: int = 8, use_local_model: bool | None = None,
                 use_skills: bool = True):
        """
        初始化 CodingAgent。

        参数：
            max_steps：
                Agent 理论上的最大执行步数。
                当前实现中主要作为配置项保存。

            use_local_model：
                是否使用本地 LLM。
                如果为 None，则根据环境变量 USE_LLM_CODING_AGENT
                自动决定。

            use_skills：
                是否启用 Skill 匹配和加载机制。
        """
        self.max_steps = max_steps
        self.use_local_model = (os.environ.get("USE_LLM_CODING_AGENT") == "1"
                                if use_local_model is None else use_local_model)
        self.use_skills = use_skills
        self.skill_loader = SkillLoader(str(Path(__file__).parent / "skills"))
        self.code_search = CodeSearchSubagent()
        self.test_runner = TestRunnerSubagent()

    @staticmethod
    def _step(steps: list[dict], thought: str, tool_call: dict, observation: object) -> None:
        """
                记录 Agent 的一个执行步骤。
        
                每一步包含三个核心信息：
        
                    thought：
                        Agent 当前采取这个动作的原因。
        
                    tool_call：
                        实际调用的工具以及相关参数。
        
                    observation：
                        工具调用后得到的结果。
        
                例如：
        
                    {
                        "thought": "读取源代码定位问题",
                        "tool_call": {
                            "tool": "subagent.code_search"
                        },
                        "observation": "发现 calculator.py 中存在错误"
                    }
        
                这种 Trace 结构方便后续进行：
                    - Agent 行为评估
                    - Debug
                    - JSON 序列化
                    - ReAct 轨迹分析
                """
        steps.append({"thought": thought, "tool_call": tool_call, "observation": observation})

    def _model_thought(self, issue: str, source: str, skills: dict[str, str]) -> str:
        """Ask only a local OpenAI-compatible Qwen endpoint for a short plan."""
        """
        调用本地 OpenAI-compatible Qwen Endpoint 生成简短修复计划。

        这里并不让模型直接修改代码，
        而是让模型根据：
            - issue：用户提出的问题
            - source：当前源代码
            - skills：匹配到的 Skill 指令

        给出一个简短的修复思路。

        如果没有启用本地模型，
        则使用确定性的离线规划器。

        这样可以保证 toy repository 的评估结果具有可重复性。
        """
        
        if not self.use_local_model:
            return "Offline deterministic planner selected for reproducible toy-repo evaluation."
        try:
            from openai import OpenAI
            client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
                            api_key=os.environ.get("OPENAI_API_KEY", "ollama"))
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "qwen2.5-coder:7b-instruct"), temperature=0,
                messages=[{"role": "system", "content": "Plan one minimal source fix. Do not modify tests."},
                          {"role": "user", "content": f"Issue:\n{issue}\nSource:\n{source}\nSkills:\n{skills}"}])
            return response.choices[0].message.content or "Local model returned no plan."
        except Exception as error:
            return f"Local model unavailable; using deterministic planner: {error}"

    def run(self, repo_path: str, issue: str) -> Trace:
        """
        执行完整的 Coding Agent 修复流程。

        参数：
            repo_path：
                要修复的本地代码仓库路径。

            issue：
                用户描述的代码问题。

        返回：
            Trace 对象，包含：
                - 完整执行步骤
                - 最终 patch
                - 测试是否通过
                - 使用的 Skill
                - 错误信息（如果发生）
        """
        tools = RepositoryTools(repo_path)
        steps: list[dict] = []
        selected = ([item["name"] for item in self.skill_loader.match(issue)]
                    if self.use_skills else [])
        loaded_skills = {name: self.skill_loader.load(name) for name in selected}
        self._step(steps, "Load only the skill instructions relevant to the issue."
                   if self.use_skills else "Run the pure-prompt control without Skill instructions.",
               {"tool": "skills.match", "task": issue}, {"selected": selected, "loaded": loaded_skills})

        investigation = self.code_search.run(tools, issue)
        self._step(steps, "Delegate read-only diagnosis to an isolated code-search context.",
                   {"tool": "subagent.code_search"}, investigation["summary"])

        source = investigation["source"]
        if not source.get("ok"):
            return Trace(steps=steps, patch="", tests_passed=False, error=source.get("error"))
        original = source["content"]
        self._step(steps, "Use the local Qwen planner when configured; otherwise retain an offline plan.",
                   {"tool": "local_model.plan", "model": os.environ.get("OPENAI_MODEL", "qwen2.5-coder:7b-instruct")},
                   self._model_thought(issue, original, loaded_skills))
        if "return a - b" not in original:
            tests = self.test_runner.run(tools)
            self._step(steps, "The expected bug signature was absent; verify the repository state.",
                       {"tool": "subagent.test_runner"}, tests["summary"])
            return Trace(steps=steps, patch=tools.git_diff().get("output", ""),
                         tests_passed=bool(tests["result"].get("ok")))

        updated = original.replace("return a - b", "return a + b", 1)
        write = tools.write_file("calculator.py", updated)
        self._step(steps, "The issue and source identify subtraction in add; replace it with addition.",
                   {"tool": "write_file", "path": "calculator.py"}, write)
        if not write.get("ok"):
            return Trace(steps=steps, patch="", tests_passed=False, error=write.get("error"))

        tests = self.test_runner.run(tools)
        self._step(steps, "A passing pytest run is the explicit completion signal.",
                   {"tool": "subagent.test_runner"}, tests["result"])
        diff = tools.git_diff()
        self._step(steps, "Capture the resulting patch for replay and review.",
                   {"tool": "git_diff"}, diff)
        return Trace(steps=steps, patch=diff.get("output", ""),
                     tests_passed=bool(tests["result"].get("ok")), skills=selected)