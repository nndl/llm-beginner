# 任务六：Mini Coding Agent

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 目标

复刻一个极简版 Claude Code，能在本地仓库上理解任务、修改代码、运行测试并迭代。本任务**完全超出**实践书 v2 的覆盖范围，引入 MCP、Skill、Subagent 三层栈。

## 前置阅读

- [SWE-bench](https://www.swebench.com/)
- [CodeAct 论文](https://arxiv.org/abs/2402.01030)
- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
- [smolagents](https://github.com/huggingface/smolagents)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [Anthropic Skills](https://github.com/anthropics/skills)

## 准备

```bash
pip install -r requirements.txt

# 下载 SWE-bench Lite 抽样集 + 模型部署提示
python data/download.py
```

## 能力三层栈

| 层 | 概念 | 角色 |
|---|---|---|
| 底层 | **Tools / MCP** | 原子工具，通过 MCP server 接入，无状态、可跨 agent 复用 |
| 中层 | **Skills** | 组织化能力包（SKILL.md + scripts + references），按需加载、渐进式披露 |
| 顶层 | **Subagents** | 独立 context 的子 agent，处理可并行或需隔离的子任务 |

## 实施步骤

1. **MCP server**（`src/mcp_server.py`）：暴露工具：`read_file` / `write_file` / `run_tests` / `git_diff` / `git_apply`
2. **Skill 加载器**（`src/skill_loader.py`，约 50 行）：扫描 `src/skills/*/SKILL.md`，按 description 匹配，按需把内容塞进 agent context
3. **Skills**（`src/skills/`）：至少 2-3 个，比如：
   - `code-review/SKILL.md`：代码审查 workflow
   - `pr-description-writer/SKILL.md`：PR 描述生成
   - `test-runner/SKILL.md`：测试运行与失败诊断
4. **Subagent**（`src/subagents/`）：如把「代码搜索」和「测试执行」分给独立 subagent，主 agent 只看摘要
5. **主 agent loop**（`src/agent.py`）：`while not done: model → tool → observation → loop`，支持调用 Skill、派发 Subagent

## 实现约定

| 文件 | 必须导出 |
|---|---|
| `src/mcp_server.py` | 可独立运行的 MCP server（`python src/mcp_server.py` 启动），列出所有工具 |
| `src/skill_loader.py` | `class SkillLoader` 含 `list_skills() -> List[dict]`、`load(name) -> str` |
| `src/agent.py` | `class CodingAgent` 含 `run(repo_path: str, issue: str) -> Trace`；`Trace` 含 `steps`、`patch`、`tests_passed: bool` |

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 |
|---|---|
| `mcp_server_lists_tools` | 启动 MCP server 后能枚举到 ≥ 5 个工具 |
| `skill_loader_metadata` | SkillLoader.list_skills() 返回的每个 skill 都有 name + description |
| `swebench_lite_sample` | 在 SWE-bench Lite 抽样 3 题上 ≥ 1 题 tests_passed（很难，跑通 1 题就合格） |

## AI Tutor 反馈

`eval/tutor_prompt.md`。

## 实验建议

- Q4_K_M 量化 vs FP16 的成功率
- 单 agent vs 加 Subagent 的 token 消耗与成功率
- 纯 prompt vs 加 Skill 的成功率

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions)。如果跑通 SWE-bench Lite 题，请把 trace 和 patch 一起附上。

## 时间

约 3-4 周。
