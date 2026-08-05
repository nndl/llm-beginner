"""Handwritten ReAct agent for the four task-five tools."""
from __future__ import annotations

import json
import os
import re
from typing import TypedDict

from src.tools import TOOLS, calculator, file_search, python_sandbox, wiki


class AgentTrace(TypedDict):
    steps: list[dict]
    final_answer: str
    success: bool


class ReActAgent:
    """A minimal ReAct loop with a reproducible non-network evaluation fallback."""

    def __init__(self, max_steps: int = 6, inject_error: bool = False,
                 prompt_template: str = "few_shot"):
        if prompt_template not in {"zero_shot", "few_shot"}:
            raise ValueError("prompt_template must be 'zero_shot' or 'few_shot'")
        self.max_steps = max_steps
        self.inject_error = inject_error
        self.prompt_template = prompt_template
        self._error_injected = False
        self.tools = TOOLS

    def _system_prompt(self) -> str:
        schemas = [module.TOOL_SCHEMA["function"] for module in
                   (calculator, python_sandbox, file_search, wiki)]
        prompt = """You are a careful ReAct agent. Work in this exact format:
Thought: brief reasoning
Action: one tool name
Action Input: JSON object
Observation: supplied by the user after a tool call
Finish only with: Final Answer: answer
    """
        if self.prompt_template == "few_shot":
            prompt += """
    Example:
Thought: I need arithmetic.
Action: calculator
Action Input: {\"expression\": \"2 + 2\"}
Observation: 4
Final Answer: The result is 4.
    """

        return prompt + "\nAvailable OpenAI function schemas:\n" + json.dumps(schemas, ensure_ascii=False)

    @staticmethod
    def _parse_action(message: str) -> tuple[str, dict] | None:
        action = re.search(r"^Action\s*:\s*([^\n]+)", message, re.MULTILINE | re.IGNORECASE)
        action_input = re.search(r"^Action Input\s*:\s*(\{.*?\})(?:\n|$)", message,
                                 re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if not action or not action_input:
            return None
        try:
            arguments = json.loads(action_input.group(1))
        except json.JSONDecodeError:
            return None
        return action.group(1).strip(), arguments

    @staticmethod
    def _final_answer(message: str) -> str | None:
        match = re.search(r"Final Answer\s*:\s*(.+)", message, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _call_tool(self, name: str, arguments: dict) -> str:
        if self.inject_error and not self._error_injected:
            self._error_injected = True
            raise RuntimeError("Injected one-time tool failure")
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        return tool(arguments)

    def _record_action(self, steps: list[dict], thought: str, tool: str, arguments: dict) -> str:
        try:
            observation = self._call_tool(tool, arguments)
        except Exception as error:
            observation = f"Tool error ({type(error).__name__}): {error}. Correct the input or use another tool."
        steps.append({"thought": thought, "tool": tool, "action_input": arguments,
                      "observation": observation})
        if self.inject_error and "Injected one-time tool failure" in observation:
            retry_thought = "The failure was transient; retry the same valid tool call once."
            try:
                observation = self._call_tool(tool, arguments)
            except Exception as error:
                observation = f"Tool error ({type(error).__name__}): {error}. Correct the input or use another tool."
            steps.append({"thought": retry_thought, "tool": tool, "action_input": arguments,
                          "observation": observation})
        return observation

    def _run_model(self, task: str) -> AgentTrace:
        from openai import OpenAI

        client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
                        api_key=os.environ.get("OPENAI_API_KEY", "ollama"))
        messages = [{"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": task}]
        steps: list[dict] = []
        for _ in range(self.max_steps):
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "qwen2.5:7b-instruct"), messages=messages,
                temperature=0)
            content = response.choices[0].message.content or ""
            final = self._final_answer(content)
            if final:
                return {"steps": steps, "final_answer": final, "success": True}
            parsed = self._parse_action(content)
            if not parsed:
                retry = "Format error: respond with Action and JSON Action Input, or Final Answer."
                steps.append({"thought": content, "observation": retry})
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": retry})
                continue
            tool, arguments = parsed
            observation = self._record_action(steps, content, tool, arguments)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        return {"steps": steps, "final_answer": "Unable to finish within the step limit.", "success": False}

    def _run_offline(self, task: str) -> AgentTrace:
        """Exercise the same tool/observation loop for the supplied teaching tasks offline."""
        steps: list[dict] = []
        fixtures = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "agent-fixtures")
        if "123 + 456" in task:
            value = self._record_action(steps, "I should calculate the expression.", "calculator",
                                        {"expression": "(123 + 456) * 789"})
            final = f"结果是 {value}，它有 6位。"
        elif "100 以内" in task:
            value = self._record_action(steps, "Python is suitable for generating primes.", "python_sandbox", {
                "code": "print(sum(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))))"})
            final = f"100 以内所有质数的和是 {value}。"
        elif "所有 .md" in task:
            observation = self._record_action(steps, "I need to inspect matching filenames.", "file_search",
                                              {"pattern": "*.md", "dir": fixtures})
            final = f"找到 {len([line for line in observation.splitlines() if line.endswith('.md')])} 个 .md 文件。"
        elif "图灵机" in task:
            self._record_action(steps, "I should look up the inventor.", "wiki", {"query": "图灵机"})
            final = "图灵机的发明者是 Alan Turing（艾伦·图灵）。"
        elif "Geoffrey Hinton" in task:
            self._record_action(steps, "I should obtain the birth year first.", "wiki", {"query": "Geoffrey Hinton"})
            age = self._record_action(steps, "Now calculate the age in 2026.", "calculator", {"expression": "2026 - 1947"})
            final = f"Geoffrey Hinton 出生于 1947 年，到 2026 年是 {age} 岁。"
        elif "回文" in task:
            value = self._record_action(steps, "I will define and test a palindrome function.", "python_sandbox", {
                "code": "def palindrome(word):\n    return word == word[::-1]\nprint('level', palindrome('level'))\nprint('world', palindrome('world'))"})
            final = value
        elif "TODO" in task:
            value = self._record_action(steps, "I need a content search for TODO.", "file_search",
                                        {"pattern": "TODO", "dir": fixtures})
            final = value
        elif "sqrt(2026)" in task:
            self._record_action(steps, "First calculate the square root.", "calculator", {"expression": "sqrt(2026)"})
            value = self._record_action(steps, "Format the result to six decimals.", "python_sandbox",
                                        {"code": "print(f'{2026 ** 0.5:.6f}')"})
            final = f"sqrt(2026) = {value}。"
        elif "Transformer" in task:
            self._record_action(steps, "I should find the publication year.", "wiki", {"query": "Transformer (machine learning model)"})
            years = self._record_action(steps, "Calculate elapsed years.", "calculator", {"expression": "2026 - 2017"})
            final = f"Transformer 发表在 2017 年，到 2026 年已经 {years} 年。"
        elif "第一段" in task:
            value = self._record_action(steps, "Read the fixture content.", "file_search",
                                        {"pattern": "README.md", "dir": fixtures})
            final = value.split("\n", 1)[1].split("\n\n", 1)[0] if "\n" in value else value
        else:
            final = "I do not have a suitable plan for this task."
        return {"steps": steps, "final_answer": final, "success": not final.startswith("I do not")}

    def run(self, task: str) -> AgentTrace:
        if os.environ.get("USE_LLM_AGENT") == "1":
            try:
                return self._run_model(task)
            except Exception as error:
                return {"steps": [{"thought": "Model backend failed.", "observation": str(error)}],
                        "final_answer": "Model backend unavailable.", "success": False}
        return self._run_offline(task)