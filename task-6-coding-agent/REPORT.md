# Task 6 实验总结（按任务）

## M1：手写 MCP Server（`mcp_server_lists_tools`）

### 我做了什么
- 在 `src/mcp_server.py` 实现 `list_tools()`，可枚举 5 个工具：`read_file`、`write_file`、`run_tests`、`git_diff`、`git_apply`。
- 实现 `RepositoryTools` 作为无状态工具集合，并在 `main()` 中通过 `FastMCP` 暴露为可独立启动的 stdio server。
- 所有命令执行统一走 `subprocess.run(args=[...], cwd=repo_root, timeout=...)`，不使用 `shell=True`。

### 针对性设计与取舍
- 路径安全：通过 `_path()` 做 `Path.resolve()` + 仓库边界校验，拒绝绝对路径和目录穿越，避免越界读写。
- 故障隔离：工具层统一返回结构化字典（`ok/error/output`），不把异常抛出到 server 顶层。
- 输出裁剪：命令输出只保留尾部片段，避免 trace 被超长日志淹没。

### 结果
- 自检通过：`mcp_server_lists_tools = pass`。
- 枚举到的工具数满足 DoD 要求（5/5）。

---

## M2：Skill 加载器 + Skills（`skill_loader_metadata`）

### 我做了什么
- 在 `src/skill_loader.py` 实现 `SkillLoader`：
	- `list_skills()`：扫描 `src/skills/*/SKILL.md` 并返回 `name + description`。
	- `load(name)`：按名称加载正文。
	- `match(task)`：仅基于 metadata 描述做关键词命中。
- YAML front-matter 解析通过 `_split()` 实现，正文只在命中后加载。

### 针对性设计与取舍
- 渐进式披露：先匹配 description，再按需读 body，避免把全部 skill 正文提前塞入上下文。
- 元数据硬约束：缺少 `name` 或 `description` 的文档不会进入 `list_skills()` 输出，直接对齐自检规则。

### 结果
- 自检通过：`skill_loader_metadata = pass`。
- `eval/result.json` 记录 `count = 3`，`missing_meta = []`。

---

## M3：Agent Loop 修复 toy-repo（`toy_repo_patch`）

### 我做了什么
- 在 `src/agent.py` 实现 `CodingAgent.run(repo_path, issue) -> Trace` 的闭环：
	1. 选择并加载相关 Skills（可开关）。
	2. 交给 `code_search` 子 agent 做只读定位。
	3. 在主 agent 中执行最小改动（`return a - b` -> `return a + b`）。
	4. 交给 `test_runner` 子 agent 跑 `python -m pytest -q`。
	5. 收集 `git_diff` 作为 patch 输出。

### 针对性设计与取舍
- 明确停机信号：以测试通过为 done 条件，而不是“到步数就停”。
- 最小变更策略：只改 `calculator.py` 一处业务实现，不改测试。
- 可复现优先：默认提供离线确定性 planner；本地模型不可用时不阻塞主流程。

### 结果
- 自检通过：`toy_repo_patch = pass`。
- `pytest` 结果：`3 passed in 0.23s`。
- `trace_has_patch = true`，说明修复补丁可回放。

---

## M4：Trace 完整记录（`steps` / `patch` / `tests_passed`）

### 我做了什么
- `Trace` 设计为 `dict` 子类，保证评测可通过 `trace.get(...)` 访问。
- 每一步统一写入 `thought`、`tool_call`、`observation` 三元组。
- 最终返回中固定包含：
	- `steps`
	- `patch`
	- `tests_passed`
	- 以及可选 `skills`、`error`

### 针对性设计与取舍
- 主 agent 不吞并子 agent 的完整上下文，仅消费摘要/结果，控制 trace 体积并保持可读。
- 把 `git_diff` 放入最终 trace，保证“为什么成功”和“改了什么”可审计。

### 结果
- M4 的字段契约已满足；在 `toy_repo_patch` 用例中验证通过。

---

## 加分项进度

### S3（纯 Prompt vs Skills）
- 已完成，详见 `BONUS_S3_REPORT.md`。
- 结论：在 toy-repo 单题上两组均成功（`1/1`），Skills 未改变该题成功率，但提升了过程可审计性（可见匹配与加载记录）。

### S1 / S2 / S4
- S1（Q4_K_M vs FP16）：未完成。
- S2（单 agent vs Subagent 量化对比）：未完成系统化统计。
- S4（SWE-bench Lite 抽样）：当前按评测规则跳过，原因是缺少 `data/swebench-lite-sample.parquet`。

---

## 总结（200-500 字）

本次任务的关键不是“把 bug 改对”，而是把改对过程拆成可验证的三层能力：底层 Tools 负责受限、可复现的执行；中层 Skills 负责按需注入任务知识；顶层 Subagents 负责上下文隔离与角色分工。针对 toy-repo 这个受控问题，我优先保证了 M1-M4 的闭环完整性：工具可枚举且安全边界明确，Skill metadata 可检索且按需加载，主循环以测试通过作为唯一停机信号，Trace 则完整保留了决策与补丁证据。实践上，最有效的策略是“先最小可用，再逐层增强”：先保证 read/write/test 三件套能通，再加 Skills 与 Subagents，否则很容易在复杂编排中失去可调试性。当前实现已经稳定通过必做项；加分项里，S3 已完成并给出对照结果，S1/S2/S4 仍需在更多样本和更严格对照下补齐，尤其是把成功率与 token 开销一起报告，才能得出更有统计意义的结论。