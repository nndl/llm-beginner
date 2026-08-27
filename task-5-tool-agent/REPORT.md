# Task 5 任务完成情况分析报告

## 1. 总体结论

本任务必做项 M1-M4 已完成；在当前环境下自检结果为：

- `tools_individual`：通过（`wiki` 因网络不可用按规则跳过，不影响其余工具判定）
- `multi_tool_success_rate`：通过（10/10，成功率 1.0，阈值 > 0.6）
- `error_recovery`：通过（3/3，恢复率 1.0，阈值 > 0.4）

评测依据来自 `eval/result.json`，实现依据来自 `src/agent.py`、`src/tools/*`、`eval/run.py`。

## 2. 必做任务逐项验收（M1-M4）

### M1：实现四个工具并可单测

要求：`calculator / python_sandbox / file_search / wiki` 均实现，且接口匹配评测约定。

```
        calculator
             │
     ┌───────┴────────┐
     ↓                ↓
TOOL_SCHEMA         run(args)
     │                │
“我是什么”        “我怎么执行”
“需要什么参数”    “执行具体逻辑”
```
证据：

- `tools_individual` 结果：`calculator=true`、`python_sandbox=true`、`file_search=true`
- `wiki` 在工具单测中显示 `skip(网络不可用)`，符合 README 中“网络不可用可跳过”的判定规则
- 评测脚本 `eval/run.py` 已按固定参数键调用各工具，说明接口兼容

结论：M1 完成。

### M2：手写 ReAct 循环（Thought/Action/Action Input/Observation）

要求：不能依赖外部 agent 封装，需自行实现循环、路由、终止。

证据（`src/agent.py`）：

- `_parse_action` 解析 `Action` 与 `Action Input(JSON)`
- `_record_action` 记录 thought、tool、action_input、observation 到 `steps`
- `_final_answer` 检测 `Final Answer`
- `run()` 中设置 `max_steps`，防止无限循环

结论：M2 完成。

### M3：工具异常不能 crash，需转为 Observation 继续

要求：工具调用失败后，agent 仍可继续决策并尝试恢复。

证据（`src/agent.py` + `eval/run.py`）：

- `_record_action` 中 `except Exception` 将错误写为 `Tool error (...)` Observation
- `inject_error=True` 时首次工具调用人为抛错，再追加一次重试 step
- `error_recovery` 结果 3/3，且 `retried=true`

结论：M3 完成。

### M4：10 题任务集成功率 > 60%

要求：按关键词命中判分，不信任自报 success。

证据（`eval/result.json`）：

- `n=10`，`rate=1.0`，`pass=true`
- 每题 `used_expected_tools=true`

结论：M4 完成。

## 3. 10 道任务逐题完成情况

以下按 `data/tasks.json` 的任务定义与 `eval/result.json` 的执行结果逐条对应。

### Task 1：表达式计算与位数

- 目标：计算 `(123 + 456) * 789` 并给位数
- 期望工具：`calculator`
- 实际：命中 `456831` 与“6位”关键词；工具使用匹配
- 结论：完成

### Task 2：100 以内质数和

- 目标：用 Python 计算质数和
- 期望工具：`python_sandbox`
- 实际：输出包含 `1060`；工具使用匹配
- 结论：完成

### Task 3：统计 .md 文件数量

- 目标：在 `data/agent-fixtures` 中统计 `.md`
- 期望工具：`file_search`
- 实际：答案为“找到 2 个 .md 文件”；工具使用匹配
- 结论：完成

### Task 4：查询图灵机发明者

- 目标：给出发明者
- 期望工具：`wiki`
- 实际：答案包含 `Turing/图灵`；工具使用匹配
- 结论：完成

### Task 5：Hinton 出生年与年龄计算

- 目标：查出生年并计算到 2026 年年龄
- 期望工具：`wiki + calculator`
- 实际：答案含 `1947` 与 `79`；两工具均调用
- 结论：完成

### Task 6：回文函数与样例测试

- 目标：判断 `level`、`world`
- 期望工具：`python_sandbox`
- 实际：答案含 `level True`、`world False`
- 结论：完成

### Task 7：查找含 TODO 的文件

- 目标：返回包含 TODO 的路径
- 期望工具：`file_search`
- 实际：答案包含 `todo_note.md`
- 结论：完成

### Task 8：sqrt(2026) 保留 6 位小数

- 目标：数值计算 + 格式化
- 期望工具：`calculator + python_sandbox`
- 实际：答案含 `45.011110`；两工具均调用
- 结论：完成

### Task 9：Transformer 年份差

- 目标：查发表年份并算到 2026 经过年数
- 期望工具：`wiki + calculator`
- 实际：答案含 `2017`、`9`；两工具均调用
- 结论：完成

### Task 10：读取 README 第一段

- 目标：读取并返回首段内容
- 期望工具：`file_search`
- 实际：答案命中“任务五”“本地文件检索测试文件”
- 结论：完成

## 4. 加分任务完成情况

### S3：Prompt 模板消融（完成实现，待在线模型复现实验）

- 已完成内容：提供 `zero_shot/few_shot` 两模板、`eval/prompt_ablation.py`、`BONUS_S3_REPORT.md`
- 当前状态：本机未启动可用的本地 OpenAI 兼容模型服务时，无法产出真实对比数值
- 结论：工程实现完成，实验数据采集待环境就绪后复跑

### S4：故障注入与恢复（已完成并验证）

- 已完成内容：`inject_error` 钩子 + 重试路径 + 评测用例
- 结果：`error_recovery` 3/3，恢复率 1.0
- 结论：S4 完成

## 5. 风险与改进建议

- `python_sandbox` 仅为教学级防护，不能当真实不可信代码沙箱
- `wiki` 工具受网络影响，线上评测前应先做连通性检查
- 若启用真实 LLM（`USE_LLM_AGENT=1`），建议补充：
	- 输出格式失败率统计
	- 工具误调用率统计
	- 失败样本的 trace 归因（解析失败/路由失败/事实检索失败）

## 6. 结语

本次不是“仅跑通 demo”，而是完成了按 DoD 可验收的工具化 ReAct Agent：

- 工具接口可单测
- 循环行为可追踪
- 异常可恢复
- 任务集达标且逐题可解释

后续重点应放在真实在线模型条件下的鲁棒性指标，而非继续堆砌离线成功率。