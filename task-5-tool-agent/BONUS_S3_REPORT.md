# 任务五 S3 加分任务交付文档

## 实验目标

比较零样本提示词和单条 few-shot 示例对 ReAct 工具调用的影响。两组实验使用同一个模型、任务集、温度（0）和步数上限（6）；唯一变量是提示模板。

## 实现

`src/agent.py` 的 `ReActAgent` 新增 `prompt_template` 参数：

- `zero_shot`：仅提供 ReAct 输出格式和四个工具的 schema；
- `few_shot`：在相同内容上增加一条 calculator 的完整 ReAct 示例。

`eval/prompt_ablation.py` 会对两种模板分别运行 10 道任务，并记录：

- `answer_keyword_rate`：最终答案关键词命中率；
- `expected_tool_coverage_rate`：每题预期工具是否均被调用；
- `format_failure_rate`：模型输出未满足 Action JSON 或 Final Answer 格式的任务比例。

脚本强制要求 `USE_LLM_AGENT=1`，避免离线确定性规划器使两种 prompt 得到虚假的相同结果。

## 复现命令

```bash
cd task-5-tool-agent
USE_LLM_AGENT=1 OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_MODEL=qwen2.5:7b-instruct python eval/prompt_ablation.py
```

结果会写入 `eval/prompt_ablation.json`。

## 当前运行状态

当前环境中 `localhost:11434` 没有运行模型服务，因此无法在不启动额外模型进程的前提下获得真实 S3 数值。脚本在服务不可用时会停止并报告错误，不生成或伪造实验结果。

## 结论

**这是加分任务 S3 的实验实现与交付文档。** 实验代码和控制变量已完成；待可用的 OpenAI 兼容本地模型服务启动后，运行上述命令即可生成正式对比结果。