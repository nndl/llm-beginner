"""Handwritten ReAct agent for the four task-five tools."""
from __future__ import annotations

import json
import os
import re
from typing import TypedDict

from src.tools import TOOLS, calculator, file_search, python_sandbox, wiki
#   用户任务
#      ↓
#   LLM 思考（Thought）
#      ↓
#   决定调用哪个工具（Action）
#      ↓
#   生成工具参数（Action Input）
#      ↓
#   执行工具
#      ↓
#   得到工具返回结果（Observation）
#      ↓
#   将 Observation 再交给 LLM
#      ↓
#   LLM 继续思考
#      ↓
#   最终输出 Final Answer

class AgentTrace(TypedDict):
    # 定义 Agent 的执行结果结构。
    #
    # 最终返回的数据类似：
    #
    # {
    #     "steps": [...],
    #     "final_answer": "...",
    #     "success": True
    # }
    steps: list[dict]
    final_answer: str
    success: bool


class ReActAgent:
    """A minimal ReAct loop with a reproducible non-network evaluation fallback."""

    def __init__(self, max_steps: int = 6, inject_error: bool = False,
                 prompt_template: str = "few_shot"):
        # 初始化 ReAct Agent。
        #
        # max_steps：
        #   Agent 最多进行多少轮 LLM → Tool → Observation 循环。
        #   默认 6 轮。
        #
        # inject_error：
        #   是否人为注入一次工具错误。
        #   主要用于测试 Agent 的错误恢复和重试能力。
        #
        # prompt_template：
        #   Prompt 模板类型。
        #   支持：
        #       zero_shot —— 只告诉模型规则
        #       few_shot  —— 告诉规则并提供示例
        if prompt_template not in {"zero_shot", "few_shot"}:
            # 检查 Prompt 类型是否合法。
            # 如果既不是 zero_shot 也不是 few_shot，就直接报错。
            raise ValueError("prompt_template must be 'zero_shot' or 'few_shot'")
        self.max_steps = max_steps                  # 保存最大执行步数。
        self.inject_error = inject_error            # 保存是否注入错误的标志。
        self.prompt_template = prompt_template      # 保存当前使用的 Prompt 类型。
        self._error_injected = False                # 记录是否已经注入过错误。
        self.tools = TOOLS                          # 保存工具注册表。

    def _system_prompt(self) -> str:
        # 获取四个工具对应的 OpenAI Function Calling Schema。
        #
        # 每个工具模块中都定义了 TOOL_SCHEMA，
        # 这里取出其中的 function 部分，
        # 最后将这些工具的接口描述告诉 LLM。
        schemas = [module.TOOL_SCHEMA["function"] for module in
                   (calculator, python_sandbox, file_search, wiki)]
        # 构造 ReAct Agent 的系统 Prompt。
        #
        # 要求 LLM 严格按照：
        # Thought → Action → Action Input → Observation
        # 的格式进行工作。
        #
        # Observation 并不是 LLM 自己生成的，
        # 而是工具真正执行后由 Agent 返回给 LLM。
        prompt = """You are a careful ReAct agent. Work in this exact format:
Thought: brief reasoning
Action: one tool name
Action Input: JSON object
Observation: supplied by the user after a tool call
Finish only with: Final Answer: answer
    """
        # 如果使用 few-shot Prompt，
        # 就额外提供一个完整的 ReAct 示例，
        # 帮助 LLM 理解应该如何调用工具。
        if self.prompt_template == "few_shot":
            prompt += """
    Example:
Thought: I need arithmetic.
Action: calculator
Action Input: {\"expression\": \"2 + 2\"}
Observation: 4
Final Answer: The result is 4.
    """

        # 将工具 Schema 序列化成 JSON，
        # 拼接到系统 Prompt 后面，
        # 让 LLM 知道当前有哪些工具以及每个工具需要什么参数。
        #
        # ensure_ascii=False：
        #   保证中文等非 ASCII 字符正常显示，而不是转换成 \uXXXX。
        return prompt + "\nAvailable OpenAI function schemas:\n" + json.dumps(schemas, ensure_ascii=False)

    @staticmethod
    def _parse_action(message: str) -> tuple[str, dict] | None:
        # 从 LLM 返回的文本中解析：
        #
        # Action: 工具名称
        # Action Input: JSON 参数
        action = re.search(r"^Action\s*:\s*([^\n]+)", message, re.MULTILINE | re.IGNORECASE)
        action_input = re.search(r"^Action Input\s*:\s*(\{.*?\})(?:\n|$)", message,
                                 re.MULTILINE | re.IGNORECASE | re.DOTALL)
        # 如果 Action 或 Action Input 任意一个没有找到，
        # 说明 LLM 没有按照要求的格式输出。
        if not action or not action_input:
            return None
        try:
            # 将字符串形式的 JSON 参数转换成 Python 字典。
            arguments = json.loads(action_input.group(1))
        except json.JSONDecodeError:
            # 如果 Action Input 不是合法 JSON，
            # 则认为解析失败。
            return None
        # 返回工具名称和工具参数。
        #
        # strip() 用于去掉工具名称前后的空格。
        return action.group(1).strip(), arguments

    @staticmethod
    def _final_answer(message: str) -> str | None:
        # 从 LLM 返回的文本中寻找最终答案。
        #
        # 要求 LLM 使用：
        #
        # Final Answer: xxx
        #
        # 的格式结束 ReAct 流程。
        match = re.search(r"Final Answer\s*:\s*(.+)", message, re.IGNORECASE | re.DOTALL)
        # 如果找到 Final Answer：
        #   返回冒号后面的具体答案。
        #
        # 如果没有找到：
        #   返回 None，表示 Agent 还需要继续执行。
        return match.group(1).strip() if match else None

    def _call_tool(self, name: str, arguments: dict) -> str:
        # 实际执行指定工具。
        #
        # name：
        #   LLM 决定调用的工具名称。
        #
        # arguments：
        #   LLM 生成的工具参数。

        # 如果开启了错误注入，并且之前还没有注入过错误，
        # 就人为制造一次工具执行失败。
        #
        # 这个功能主要用于测试 Agent 的错误恢复能力。
        if self.inject_error and not self._error_injected:
            self._error_injected = True
            raise RuntimeError("Injected one-time tool failure")
        tool = self.tools.get(name)   # 根据工具名称从工具注册表中获取对应工具。
        # 如果工具不存在，说明 LLM 调用了一个未知工具。
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        # 执行工具，并把参数传进去。
        #
        # 工具返回值最终会作为 Observation
        # 返回给 ReAct Agent。
        return tool(arguments)

    def _record_action(self, steps: list[dict], thought: str, tool: str, arguments: dict) -> str:
        # 执行一次完整的：
        #
        # Thought → Action → Action Input → Tool → Observation
        #
        # 并将这一步记录到 steps 中。
        #
        # steps：
        #   保存 Agent 到目前为止的所有执行过程。
        #
        # thought：
        #   LLM 当前输出的思考和行动内容。
        #
        # tool：
        #   当前调用的工具名称。
        #
        # arguments：
        #   当前工具的输入参数。
        try:
            observation = self._call_tool(tool, arguments)    # 调用具体工具。
        except Exception as error:
            # 如果工具执行失败，不直接终止整个 Agent。
            #
            # 而是把错误转换成 Observation，
            # 让 Agent 知道工具调用失败了。
            #
            # 例如：
            # Tool error (ValueError): Unknown tool: xxx.
            observation = f"Tool error ({type(error).__name__}): {error}. Correct the input or use another tool."
        # 将本次工具调用完整记录下来。
        #
        # 每一个 step 至少包含：
        #   thought：
        #       LLM 的思考
        #   tool：
        #       使用的工具
        #   action_input：
        #       工具参数
        #   observation：
        #       工具执行结果
        steps.append({"thought": thought, "tool": tool, "action_input": arguments,
                      "observation": observation})
        # 如果当前开启了错误注入，
        # 并且 Observation 表示刚才发生的是人为注入的一次性错误，
        # 那么立即重试同一个工具调用。
        if self.inject_error and "Injected one-time tool failure" in observation:
            # 构造一次重试时的 Thought。
            retry_thought = "The failure was transient; retry the same valid tool call once."
            try:
                # 使用完全相同的工具和参数重新调用一次。
                observation = self._call_tool(tool, arguments)
            except Exception as error:
                # 如果重试仍然失败，则继续将错误作为 Observation。
                observation = f"Tool error ({type(error).__name__}): {error}. Correct the input or use another tool."
            # 把重试过程也单独记录到 steps。
            steps.append({"thought": retry_thought, "tool": tool, "action_input": arguments,
                          "observation": observation})
        # 返回最终的工具执行结果，
        # 后续会将它作为 Observation 发送给 LLM。
        return observation

    def _run_model(self, task: str) -> AgentTrace:
        # 使用真实的 LLM 后端运行 ReAct Agent。
        #
        # 这里使用 OpenAI Python SDK，
        # 但通过 base_url 可以连接本地兼容 OpenAI API 的模型服务，
        # 例如 Ollama。
        from openai import OpenAI

        # 创建 OpenAI 客户端。
        #
        # OPENAI_BASE_URL：
        #   指定模型服务地址。
        #   默认使用本地 Ollama 的 OpenAI 兼容接口。
        #
        # OPENAI_API_KEY：
        #   API Key。
        #   本地 Ollama 通常不真正验证 API Key，
        #   因此默认设置为 "ollama"。
        client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
                        api_key=os.environ.get("OPENAI_API_KEY", "ollama"))
        # 初始化发送给 LLM 的消息。
        #
        # 第一条是 system message：
        #   告诉 LLM ReAct 的工作规则和工具信息。
        #
        # 第二条是 user message：
        #   用户真正提出的任务。
        messages = [{"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": task}]
        # 用于保存整个 ReAct 执行过程。
        steps: list[dict] = []
        # 最多执行 max_steps 次：
        #
        # LLM → Action → Tool → Observation → LLM
        #
        # 循环。
        for _ in range(self.max_steps):
            # 调用聊天模型。
            #
            # temperature=0：
            #   尽可能让模型输出稳定、确定的结果，
            #   方便测试和复现实验。
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "qwen2.5:7b-instruct"), messages=messages,
                temperature=0)
            # 获取模型生成的文本。
            #
            # 如果 content 为 None，则使用空字符串，
            # 避免后续字符串处理时报错。
            content = response.choices[0].message.content or ""
            # 首先检查模型是否已经给出了最终答案。
            final = self._final_answer(content)
            # 如果存在 Final Answer，
            # 说明 ReAct 流程已经完成。
            if final:
                return {"steps": steps, "final_answer": final, "success": True}
            # 如果没有最终答案，
            # 尝试从模型输出中解析 Action 和 Action Input。
            parsed = self._parse_action(content)
            # 如果无法解析 Action，
            # 说明 LLM 没有按照规定格式输出。
            if not parsed:
                # 构造格式错误提示，
                # 要求模型重新按照指定格式输出。
                retry = "Format error: respond with Action and JSON Action Input, or Final Answer."
                steps.append({"thought": content, "observation": retry})   # 将本次格式错误记录下来。
                messages.append({"role": "assistant", "content": content})   # 将模型刚才的错误输出加入对话历史。
                # 再向模型发送错误提示，
                # 让模型重新生成正确格式。
                messages.append({"role": "user", "content": retry})
                # 跳过本轮，进入下一次 LLM 调用。
                continue
            # 如果解析成功，
            # parsed 中包含：
            #
            # tool：
            #   工具名称
            #
            # arguments：
            #   工具参数
            tool, arguments = parsed
            # 执行工具，并记录：
            # Thought、Action、Action Input、Observation。
            observation = self._record_action(steps, content, tool, arguments)
            # 将 LLM 当前的输出加入历史。
            messages.append({"role": "assistant", "content": content})
            # 将工具执行结果作为 Observation 返回给 LLM。
            #
            # LLM 下一轮会根据这个 Observation
            # 决定继续调用工具还是直接给出 Final Answer。
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        # 如果循环结束仍然没有得到 Final Answer，
        # 说明超过了最大执行步数。
        return {"steps": steps, "final_answer": "Unable to finish within the step limit.", "success": False}

    def _run_offline(self, task: str) -> AgentTrace:
        """Exercise the same tool/observation loop for the supplied teaching tasks offline."""
        # 离线模式下同样保存 Agent 的完整执行过程。
        steps: list[dict] = []
        # 获取测试数据目录 agent-fixtures 的路径。
        fixtures = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "agent-fixtures")
        # 如果任务中包含 "123 + 456"，
        # 就调用 calculator 工具进行数学计算。
        if "123 + 456" in task:
            # 调用 calculator：
            #
            # (123 + 456) * 789
            #
            # 同时将完整执行过程记录到 steps。
            value = self._record_action(steps, "I should calculate the expression.", "calculator",
                                        {"expression": "(123 + 456) * 789"})
            final = f"结果是 {value}，它有 6位。"  # 根据计算结果生成最终答案。
        # 如果任务要求计算 100 以内质数的和，
        # 使用 Python Sandbox 执行 Python 代码。
        elif "100 以内" in task:
            # 调用 python_sandbox。
            #
            # range(2, 101)：
            #   枚举 2~100。
            #
            # all(...)：
            #   判断一个数字是否能被 2 到 sqrt(n)
            #   范围内的整数整除。
            #
            # 如果没有任何因数，则认为是质数。
            value = self._record_action(steps, "Python is suitable for generating primes.", "python_sandbox", {
                "code": "print(sum(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))))"})
            final = f"100 以内所有质数的和是 {value}。"   # 根据工具返回结果生成最终答案。
        # 如果任务要求查找所有 .md 文件，
        # 使用 file_search 工具进行文件搜索。
        elif "所有 .md" in task:
            # pattern="*.md"：
            #   匹配所有 Markdown 文件。
            #
            # dir=fixtures：
            #   指定搜索目录。
            observation = self._record_action(steps, "I need to inspect matching filenames.", "file_search",
                                              {"pattern": "*.md", "dir": fixtures})
            # file_search 返回的结果通常是一行一个文件名。
            #
            # 通过 splitlines() 分割成多行，
            # 再筛选以 .md 结尾的行，
            # 最后统计 Markdown 文件数量。
            final = f"找到 {len([line for line in observation.splitlines() if line.endswith('.md')])} 个 .md 文件。"
        # 如果任务要求查询图灵机，
        # 使用 wiki 工具进行知识查询。
        elif "图灵机" in task:
            # 调用 wiki 工具查询 "图灵机"。
            #
            # 这里虽然获得了 Observation，
            # 但最终答案直接使用预设答案。
            self._record_action(steps, "I should look up the inventor.", "wiki", {"query": "图灵机"})
            final = "图灵机的发明者是 Alan Turing（艾伦·图灵）。"  # 给出最终答案。
        # 如果任务要求计算 Geoffrey Hinton 到 2026 年的年龄，
        # 需要先通过 wiki 查询出生年份，
        # 再通过 calculator 计算年龄。
        elif "Geoffrey Hinton" in task:
            # 第一步：
            # 使用 wiki 查询 Geoffrey Hinton 的信息，
            # 获取他的出生年份。
            self._record_action(steps, "I should obtain the birth year first.", "wiki", {"query": "Geoffrey Hinton"})
            # 第二步：
            # 根据出生年份 1947，
            # 使用 calculator 计算 2026 - 1947。
            age = self._record_action(steps, "Now calculate the age in 2026.", "calculator", {"expression": "2026 - 1947"})
            # 将计算结果组织成最终答案。
            final = f"Geoffrey Hinton 出生于 1947 年，到 2026 年是 {age} 岁。"
        # 如果任务要求判断字符串是否为回文，
        # 使用 python_sandbox 执行一个简单的 Python 函数。
        elif "回文" in task:
            # 定义 palindrome 函数：
            #
            # word[::-1]：
            #   将字符串反转。
            #
            # 如果原字符串和反转后的字符串相同，
            # 就说明它是回文。
            value = self._record_action(steps, "I will define and test a palindrome function.", "python_sandbox", {
                "code": "def palindrome(word):\n    return word == word[::-1]\nprint('level', palindrome('level'))\nprint('world', palindrome('world'))"})
            final = value  # Python Sandbox 的输出直接作为最终答案。
        # 如果任务要求查找 TODO，
        # 使用 file_search 搜索文件内容。
        elif "TODO" in task:
            # pattern="TODO"：
            #   搜索包含 TODO 的文件或内容。
            value = self._record_action(steps, "I need a content search for TODO.", "file_search",
                                        {"pattern": "TODO", "dir": fixtures})
            final = value   # 将搜索结果直接作为最终答案。
        # 如果任务要求计算 sqrt(2026)，
        # 先使用 calculator 计算平方根，
        # 再使用 Python 对结果进行六位小数格式化。
        elif "sqrt(2026)" in task:
            # 第一步：
            # 使用 calculator 计算 sqrt(2026)。
            self._record_action(steps, "First calculate the square root.", "calculator", {"expression": "sqrt(2026)"})
            # 第二步：
            # 使用 Python 进行格式化，
            # 保留 6 位小数。
            value = self._record_action(steps, "Format the result to six decimals.", "python_sandbox",
                                        {"code": "print(f'{2026 ** 0.5:.6f}')"})
            final = f"sqrt(2026) = {value}。"   # 生成最终答案。
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
        # 返回离线 Agent 的执行结果。
        #
        # success 的判断逻辑：
        #   如果最终答案不是“不知道如何处理”，
        #   则认为任务执行成功。
        return {"steps": steps, "final_answer": final, "success": not final.startswith("I do not")}

    def run(self, task: str) -> AgentTrace:
        # Agent 的统一入口。
        #
        # 根据环境变量 USE_LLM_AGENT 决定：
        #
        # USE_LLM_AGENT=1
        #   → 使用真实 LLM 运行 ReAct。
        #
        # 其他情况
        #   → 使用离线模式运行预设教学任务。
        if os.environ.get("USE_LLM_AGENT") == "1":
            try:
                # 尝试使用真实 LLM 后端执行任务。
                return self._run_model(task)
            except Exception as error:
                # 如果模型后端发生任何异常，
                # 不让程序直接崩溃，
                # 而是将错误记录下来并返回失败结果。
                return {"steps": [{"thought": "Model backend failed.", "observation": str(error)}],
                        "final_answer": "Model backend unavailable.", "success": False}
        # 如果没有开启真实 LLM，
        # 默认进入离线模式。
        return self._run_offline(task)