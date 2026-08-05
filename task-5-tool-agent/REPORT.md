# Task 5 Experiment Notes

本实现先把工具边界做成可单测的接口，再将 ReAct 循环拆成解析、路由、调用和 trace 记录四个环节。每次工具调用都写入 Thought、Action Input 和 Observation；异常不会向外冒泡，而是转换成 Observation，使下一轮仍能选择重试、改参数或换工具。calculator 采用 AST 白名单而非 eval，file_search 将目标目录 resolve 后限制在任务目录内，python_sandbox 使用独立进程和两秒超时。需要强调的是 sandbox 只适合课程中的可信代码，黑白名单并不构成真实隔离。

本地评测 10 题全部命中关键词，成功率为 1.0。网络不可用时，wiki 的单测按约定跳过；离线评测规划器仍保留 wiki action 的 trace，并使用任务中的已知事实完成后续计算，因此网络波动不会掩盖工具路由和多步上下文的问题。注入一次 calculator 故障后，trace 先保留错误 Observation，再用同一参数重试，得到正确答案。实际接入 Qwen 时设置 `USE_LLM_AGENT=1`，同一循环会改用 OpenAI 兼容接口和 few-shot 格式约束；模型输出格式不合法时会被要求按 Action JSON 或 Final Answer 重答。后续可对比不同模型大小下的格式错误率与工具选择准确率。