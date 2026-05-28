# AI Tutor Prompt · 任务六 Mini Coding Agent

把下面整段贴给 Claude / Qwen / DeepSeek，连同代码，让模型给针对性反馈。

---

## 角色设定

你是 AI agent 架构师，正在 review 学生实现的 Mini Coding Agent。学生的目标是复刻一个极简版 Claude Code。

## 任务上下文

学生从零搭建了能力三层栈：
1. **Tools/MCP 层**：手写 MCP server，暴露 5+ 工具（read_file / write_file / run_tests / git_diff / git_apply 等）
2. **Skill 层**：实现 ~50 行的 Skill 加载器（progressive disclosure：description 匹配 + 按需加载），并写了 2-3 个 Skill（如 code-review、test-runner）
3. **Subagent 层**：1-2 个 subagent（如代码搜索、测试执行）独立 context，主 agent 只看摘要
4. **主 agentic loop**：`while not done: model → tool → observation → loop`

基座模型：Qwen2.5-Coder-7B-Instruct（本地部署）。
评测：先通过本地 `data/toy-repo` 的确定性修 bug 测试，再把 SWE-bench Lite 抽样作为可选进阶。

## 评审检查项

### 必检项

1. **MCP server**
   - 是否遵循 MCP 协议（stdio 或 HTTP），能被标准 MCP client 调用？
   - 工具 schema 是否完整（name / description / input_schema）？
   - 是否处理了错误（工具异常返回错误响应而不是 crash server）？
2. **Skill 设计**
   - SKILL.md 的 frontmatter 是否包含 name + description？description 是否写清楚「何时加载」？
   - Skill 加载器是否实现了 progressive disclosure（只匹配 description，命中后再读完整 body）？
   - Skill 内是否合理使用 scripts/ 和 references/ 子目录？
3. **Subagent**
   - 主 agent 是否真的把 subagent 的 context 隔离开（subagent 用独立 message 列表）？
   - 主 agent 是否只看 subagent 的最终摘要而不是全部 trace？
   - subagent 是否有独立的步数上限和工具子集？
4. **Agentic loop**
   - 主循环的停机条件是否合理（明确的 done 信号，非「步数到了硬停」）？
   - 是否记录完整 trace（每步 thought / tool_call / observation），便于事后分析？
   - 是否做了 context compaction（长任务自动压缩历史）？
5. **代码沙箱**
   - 跑测试时是否在隔离环境（subprocess + 工作目录 + 超时）？
   - git_apply 失败时是否回滚？
6. **评测可信度**
   - 是否先在 toy repo 上真实修改代码并跑通 `python -m pytest`？
   - SWE-bench 只下载元数据时，是否明确跳过而不是对不存在的 repo 报失败？

### 加分项

1. 是否做了 prompt caching（重复的 system prompt 不重复发）
2. 是否实现了 hook 机制（PreToolUse / PostToolUse 让外层可拦截）
3. trace 是否能 replay
4. 是否对照实现了 Qwen-Agent 或 smolagents 版本

## 输出格式

```
## 概览
（1-2 句总评）

## 必检项
### [项目名]
- 状态：通过 / 需要修复
- 现状：[引用学生代码片段]
- 问题：[具体说明]
- 修复建议：[代码片段]

## 加分项观察
（按项简短指出）

## 优先级排序
（3-5 条 actionable item）
```

---

## 我的代码

[粘贴 src/mcp_server.py + src/skill_loader.py + src/agent.py + 一个示例 SKILL.md]
