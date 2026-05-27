# 任务五：工具调用 Agent

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 目标

实现 ReAct 循环，让 LLM 自主调用工具完成多步任务。**扩展**实践书 v2 ReAct 节的单工具示例：扩展到 4 类工具、错误恢复、与 Qwen-Agent 框架对照。

## 前置阅读

- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
- 实践书 v2《大语言模型与智能体》「ReAct 智能体」一节

## 准备

```bash
pip install -r requirements.txt

# 部署本地模型（Qwen2.5-7B-Instruct 经 Ollama 提供 OpenAI 兼容 API）
ollama pull qwen2.5:7b-instruct
ollama serve  # 默认 http://localhost:11434/v1

# 生成自建评测任务集（已内置在脚本里）
python data/download.py
```

## 实施步骤

1. **工具实现**（`src/tools/`）：每个工具一个文件
   - `calculator.py`：四则运算 + 高级函数
   - `python_sandbox.py`：受限 exec（限制 import、超时、stdout 捕获）
   - `file_search.py`：本地目录文件名/内容检索
   - `wiki.py`：维基百科 API 查询
2. **ReAct 循环**（`src/agent.py`）：手写约 200 行
   - prompt 模板：Thought / Action / Action Input / Observation
   - 工具路由 + 调用 + 错误捕获
   - 终止条件：模型输出 Final Answer 或步数上限
3. **对照框架**：用 Qwen-Agent 写一版功能相同的，对比成功率

## 实现约定

| 文件 | 必须导出 |
|---|---|
| `src/tools/{name}.py` | `TOOL_SCHEMA: dict`（OpenAI function calling 格式）、`def run(args: dict) -> str` |
| `src/agent.py` | `class ReActAgent` 含 `run(task: str) -> AgentTrace`；`AgentTrace` 是 dict 含 `steps`、`final_answer`、`success: bool` |

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 |
|---|---|
| `tools_individual` | 4 个工具各自跑一组单元测试全部通过 |
| `multi_tool_success_rate` | 在自建 10 题任务集（`data/tasks.json`）上成功率 > 60% |
| `error_recovery` | 注入 1 次错误工具响应后，agent 仍能完成任务的比例 > 40% |

## AI Tutor 反馈

`eval/tutor_prompt.md`。

## 实验建议

- 手写 ReAct vs Qwen-Agent 原生 function calling
- 不同模型尺寸（1.5B / 7B / 14B）的成功率
- 不同 prompt 模板对工具调用准确率的影响
- 是否使用任务三 plugin SFT 后的模型，对比 zero-shot

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions)。

## 时间

约 2 周。
