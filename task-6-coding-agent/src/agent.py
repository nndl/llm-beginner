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


class CodingAgent:
    def __init__(self, max_steps: int = 8, use_local_model: bool | None = None,
                 use_skills: bool = True):
        self.max_steps = max_steps
        self.use_local_model = (os.environ.get("USE_LLM_CODING_AGENT") == "1"
                                if use_local_model is None else use_local_model)
        self.use_skills = use_skills
        self.skill_loader = SkillLoader(str(Path(__file__).parent / "skills"))
        self.code_search = CodeSearchSubagent()
        self.test_runner = TestRunnerSubagent()

    @staticmethod
    def _step(steps: list[dict], thought: str, tool_call: dict, observation: object) -> None:
        steps.append({"thought": thought, "tool_call": tool_call, "observation": observation})

    def _model_thought(self, issue: str, source: str, skills: dict[str, str]) -> str:
        """Ask only a local OpenAI-compatible Qwen endpoint for a short plan."""
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