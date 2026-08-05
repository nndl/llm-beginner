# 任务五 S4 加分任务交付文档

## 任务目标

完成 README 中的 S4：实现 `inject_error` 钩子，在一次工具调用失败后让 Agent 保留错误 Observation、恢复执行，并以成功率验证恢复能力。

## 实现说明

`src/agent.py` 的 `ReActAgent` 支持 `inject_error=True`。首次工具调用会抛出一次 `RuntimeError`，ReAct 循环将异常转换为 Observation：

```text
Tool error (RuntimeError): Injected one-time tool failure. Correct the input or use another tool.
```

随后循环保留该步骤，并对同一有效调用重试。`eval/run.py` 的 `error_recovery` 测试会在任务 1、5、8 中分别创建带故障注入的 Agent，检查：

- trace 中存在被注入的错误 Observation；
- 错误后的后续工具调用得到非错误 Observation；
- 最终答案仍命中任务定义的关键词。

## 复现命令

```bash
cd task-5-tool-agent
python eval/run.py
```

## 实验结果

运行结果已写入 `eval/result.json`：

| 测试 | 结果 |
| --- | --- |
| `tools_individual` | 通过；wiki 网络不可用时按任务约定跳过 |
| `multi_tool_success_rate` | 通过，10/10，成功率 1.0 |
| `error_recovery` | 通过，3/3，恢复率 1.0 |

恢复率高于 S4 要求的 0.4。三道测试分别覆盖单步计算、wiki 后计算和计算后 Python 格式化；错误不会中断整个运行，也不会被最终答案掩盖，因为评测同时检查 trace 和关键词。

## 结论

**这是加分任务 S4 的交付。** 已实现并验证一次工具故障后的错误恢复，`error_recovery` 自检通过。