# Task 6 Experiment Notes

本实现把本地 Coding Agent 分为 Tools、Skills 与 Subagents 三层。Tools 通过 `RepositoryTools` 提供受仓库根目录约束的读写、测试与 Git 操作；所有外部命令使用参数列表、固定工作目录和超时，避免由 issue 或文件名拼出的 shell 命令。Skills 先只读取 YAML front-matter 的描述并进行关键词匹配，命中后才加载正文，避免把无关说明持续塞入上下文。代码搜索和测试执行分别使用独立的消息列表，主循环只消费它们的摘要和结果。

在 toy-repo 上，主 agent 先让代码搜索子 agent 读出 `add` 使用了减法，再只改动该实现。完成信号不是步数上限，而是 `python -m pytest -q` 的退出码为零；实际执行 3 个测试均通过。Trace 记录了每一步的 thought、tool_call 与 observation，并保留 Git diff 作为可回放 patch。相比纯 prompt，这个流程把失败测试转换成下一轮可观察证据；相比将所有 Skill 正文预加载，按需加载减少了上下文占用。尚未进行 Q4_K_M 与 FP16 的成功率对比，SWE-bench Lite 也因未下载样本元数据而按自检规则跳过。