# LLM-Beginner：大模型与智能体入门练习

本系列是 [NLP-Beginner](legacy/README.md) 在大模型时代的重构版本，是一份**独立的入门教程**，面向有 Python 与深度学习基础的学习者。沿用原系列「任务渐进 × 每个 2-4 周」的节奏，按"熟悉 Transformer → 从零实现 mini-GPT → 指令微调与对齐 → RAG → 工具调用 Agent → Mini Coding Agent"六个任务展开，技术栈对齐 2025-2026 的主流。

llm-beginner 可独立完成，无任何前置依赖。如果同时在读《神经网络与深度学习（第二版）》（下文简称 **NNDL2**）与配套的《神经网络与深度学习案例与实践（第二版）》（下文简称 **实践书 v2**），每个任务的"延伸阅读"会指向对应章节，配合读会更顺畅。

## 相关教材

作为本教程的学习材料，同步撰写了**《大模型与智能体》**，围绕共用基础、大模型、智能体、边界与未来四条主线展开，共 17 章。 [完整章节列表](https://nndl.ai/llm-agent/)。

这本教材可以作为 llm-beginner 的**前置知识**配合阅读：教材讲清原理，本仓库的 6 个任务负责把原理动手跑通。两者都可独立完成，不强相关。

参考：

1. 《[神经网络与深度学习](https://nndl.github.io/)》
2. 《[大模型与智能体](https://nndl.ai/llm-agent/)》（2026 出版）
3. 原始版本：[NLP-Beginner](legacy/README.md)（2019 年发布，已归档供对比阅读）
4. 不懂问搜索引擎与大模型

## 通用说明

- **设备基准**：8GB 消费级 GPU（如 RTX 3060/4060）可完成任务一至四；任务五、六推荐 16GB+ 显存，或使用 Q4_K_M 量化在 8GB 上跑。Mac M 系列通过 MPS / llama.cpp 兜底。
- **模型生态**：通义千问 Qwen 系列贯穿全程，国内可直接从 Hugging Face / ModelScope 下载。
- **语言**：以中文为主，仅在英文数据显著更好时使用英文（如部分小模型预训练语料）。
- **教学路线**：每项任务「先手写、再对照框架」——理解原理在前，工程效率在后。

---

### 任务一：熟悉 Transformer

> 详细资源、数据下载与自检脚本见 [task-1-transformer/](task-1-transformer/)

手写 self-attention 与 Transformer block，在小任务上跑通并可视化注意力权重。本任务和实践书 v2《注意力机制》章节有意重叠——为独立读者打下从零起步的 Transformer 基础。

1. 参考
   1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   2. [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
   3. [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
   4. **延伸阅读（配合读可加深理解，非必需）**
      - NNDL2 第 8 章《注意力机制与 Transformer》之「注意力机制」「自注意力」「Transformer 模型」三节
      - 实践书 v2《注意力机制》章「基于双向 LSTM 和多头自注意力的文本分类」「基于自注意力模型的文本语义匹配」两节
2. 数据集
   1. 文本分类：[ChnSentiCorp](https://huggingface.co/datasets/seamew/ChnSentiCorp)（中文情感分类）或 [LCQMC](https://huggingface.co/datasets/shibing624/nli_zh)（文本匹配，呼应实践书 v2 的语义匹配任务）
   2. Toy 任务（可选）：序列 copy / sort，便于打印注意力矩阵观察学习过程
3. 实现要求
   1. 手写 scaled dot-product attention（缩放、softmax、mask）
   2. 手写 multi-head attention
   3. 手写完整 Transformer encoder block（attention + FFN + residual + LayerNorm）
   4. 用 padding mask 跑文本分类任务
   5. 再用 causal mask 跑一遍 toy 语言模型（为任务二预热）
   6. 用 matplotlib 可视化句子内部的注意力热图
4. 知识点
   1. Self-attention 的 QKV 计算
   2. Padding mask vs causal mask
   3. Multi-head 的并行视角
   4. Pre-LN vs Post-LN
   5. 注意力可视化的解读方法
5. 实验
   1. Head 数 / 层数对分类准确率的影响
   2. 移除 residual / LayerNorm 后训练是否还能收敛
   3. 注意力热图：观察模型是否"看对了关键词"
6. 时间：2 周

---

### 任务二：从零实现 mini-GPT

> 详细资源、数据下载与自检脚本见 [task-2-mini-gpt/](task-2-mini-gpt/)

用 PyTorch 从零搭一个 decoder-only 模型，先在中文小语料上预训练并自回归生成，进阶再切到 TinyStories 或中文故事语料观察小模型叙事能力。本任务**扩展**实践书 v2「nanoGPT 模型」的带读：加入 BPE、RoPE、KV cache。

1. 参考
   1. [nanoGPT](https://github.com/karpathy/nanoGPT)
   2. [TinyStories 原论文](https://arxiv.org/abs/2305.07759)
   3. [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864)
   4. **延伸阅读**
      - NNDL2 第 8 章「现代 Transformer 的常见优化」（含 RoPE / FlashAttention / KV 缓存 / GQA / MoE）
      - 实践书 v2《大语言模型与智能体》章「nanoGPT 模型」「预训练循环」「解码 / 采样策略」三节
2. 数据集（三档渐进，按设备与目标选择）
   1. **Quick-start（~1MB）**：[poetryFromTang.txt](poetryFromTang.txt) —— 5 分钟跑通 pipeline、验证代码正确性
   2. **正式训练（~100MB，CPU 也能跑）**：[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 或社区中文版 TinyStoriesChinese ——[原论文](https://arxiv.org/abs/2305.07759)证明 10M 参数模型就能学会语法和叙事，能直观看到「涌现」
   3. **进阶训练（~1GB+，建议 GPU）**：[SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) 子集（昆仑万维开源高质量中文预训练语料）
3. 实现要求
   1. 手写简化版 BPE tokenizer（不用 tiktoken / sentencepiece）
   2. 手写 decoder-only 模型，集成 **RoPE**（超出实践书 v2 nanoGPT 使用的绝对位置编码）
   3. 实现 **KV cache**（推理加速，实践书 v2 只讲不实现）
   4. 实现采样策略：greedy / top-k / top-p / temperature
4. 知识点
   1. BPE 分词与 merge 过程
   2. RoPE 旋转矩阵的原理与外推优势
   3. KV cache 的内存换计算
   4. 困惑度与生成质量的关系
   5. 采样策略对生成多样性的影响
5. 实验
   1. 参数量扫描（10M / 50M / 100M）对训练损失与困惑度的影响
   2. 绝对位置编码 vs RoPE 在长序列外推上的差异
   3. KV cache 开 / 关 的推理速度对比
   4. TinyStories 上是否能复现 10M 参数模型涌现叙事能力
6. 时间：3 周

---

### 任务三：指令微调与偏好对齐

> 详细资源、数据下载与自检脚本见 [task-3-sft-dpo/](task-3-sft-dpo/)

在小尺寸预训练模型上做 SFT + DPO 两阶段对齐，理解从 base model 到 chat model 的全过程。本任务**扩展**实践书 v2「监督微调与 LoRA」「偏好对齐：DPO」两节：手写 LoRA、改用 MOSS 中文数据、做完整的"指令格式 → 偏好"两阶段闭环。

1. 参考
   1. [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
   2. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
   3. [Hugging Face TRL 文档](https://huggingface.co/docs/trl)
   4. [Hugging Face PEFT 文档](https://huggingface.co/docs/peft)
   5. **延伸阅读**：实践书 v2《大语言模型与智能体》章「监督微调与 LoRA」「偏好对齐：DPO」两节
2. 基座模型：[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)（8GB 显存可全量微调）
3. 数据集
   1. SFT：[MOSS-003-sft-data](https://huggingface.co/datasets/OpenMOSS-Team/moss-003-sft-data)（复旦 NLP 组发布的 110 万条中文多轮对话，取 1-5 万子集即可）
   2. DPO：自行在 Hugging Face 寻找中文偏好数据（如 UltraFeedback 翻译版），或先用英文数据集走通流程
   3. **贯通任务五**：同时下载 [MOSS-003 with-tools 数据](https://huggingface.co/datasets/OpenMOSS-Team/moss-003-sft-data)（带工具调用对话的子集），任务三教模型学会工具调用的输出格式，任务五在 Agent 循环里真的去调
4. 实现要求
   1. 先**手写 LoRA**（亲手实现低秩矩阵注入、forward 与梯度），再用 PEFT 对照
   2. SFT 与 DPO 都跑一遍，对比输出质量
5. 知识点
   1. Chat template 与 Qwen 对话格式
   2. Loss masking：只对 assistant turn 计算 loss
   3. LoRA 数学：rank、scaling、初始化
   4. DPO 损失函数与 reference model 的作用
   5. SFT vs RLHF vs DPO 的简化逻辑
6. 实验
   1. 全量微调 vs LoRA：显存占用与下游质量
   2. LoRA rank 消融（r = 4 / 8 / 16 / 32）
   3. 灾难性遗忘评估（在 C-Eval 子集上对比微调前后）
   4. SFT-only vs SFT+DPO 在偏好上的差异
7. 时间：2-3 周

---

### 任务四：RAG 文档问答

> 详细资源、数据下载与自检脚本见 [task-4-rag/](task-4-rag/)

构建端到端中文检索增强生成系统，并量化每个环节的提升。本任务**扩展**实践书 v2 RAG 节的最小示例：加入 reranker、chunking 策略消融、RAGAS 评测。

1. 参考
   1. [Retrieval-Augmented Generation 综述](https://arxiv.org/abs/2312.10997)
   2. [BGE Embedding 系列](https://huggingface.co/BAAI)
   3. [RAGAS 评测框架](https://github.com/explodinggradients/ragas)
   4. **延伸阅读**：实践书 v2《大语言模型与智能体》章「检索增强生成（RAG）」一节
2. 技术栈
   1. Embedding：[bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
   2. 向量库：FAISS（本地、轻量）
   3. Reranker：[bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
   4. 生成模型：[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)（Q4_K_M 量化版可在 8GB 跑）
3. 数据
   1. 知识库：《神经网络与深度学习（第二版）》PDF（默认下载到 `task-4-rag/data/kb.pdf`）
   2. 评测：基于 `../神经网络与深度学习2/` LaTeX 正文设计的 `task-4-rag/data/gold_qa.jsonl`
4. 实现要求：手写 RAG 流水线，**不**使用 LlamaIndex / LangChain 的高层封装
5. 知识点
   1. Chunking 策略：固定大小 / 递归切分 / 语义切分
   2. 向量检索原理与近似算法（IVF、HNSW）
   3. 两阶段检索架构（embedding 召回 + reranker 精排）
   4. 评测指标：Recall@k、MRR、faithfulness、answer relevancy
   5. Query rewriting 与 HyDE 等增强策略
6. 实验
   1. Chunk size 扫描（128 / 256 / 512 / 1024 token）对检索质量的影响
   2. 加 / 不加 reranker 的端到端提升
   3. Query rewriting 的有效性
   4. 用 RAGAS 打端到端分数
7. 时间：2 周

---

### 任务五：工具调用 Agent

> 详细资源、数据下载与自检脚本见 [task-5-tool-agent/](task-5-tool-agent/)

实现 ReAct 循环，让 LLM 自主调用工具完成多步任务。本任务**扩展**实践书 v2 ReAct 节的单工具示例：扩展到 4 类工具、错误恢复、与 Qwen-Agent 框架对照。

1. 参考
   1. [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
   2. [Toolformer](https://arxiv.org/abs/2302.04761)
   3. [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
   4. **延伸阅读**：实践书 v2《大语言模型与智能体》章「ReAct 智能体」一节
2. 基座模型：[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)（原生支持 function calling）
3. 部署：Ollama / vLLM / llama.cpp 本地推理
4. 工具实现（自己写，不依赖任何 agent 框架）
   1. 计算器
   2. Python sandbox（受限执行环境）
   3. 本地文件检索
   4. 维基百科 API
5. 实现要求
   1. **第一阶段**：手写 ReAct 循环（~200 行，纯 Python + HTTP 调用本地模型）
   2. **第二阶段**：对照 Qwen-Agent 的实现，理解工程封装与原生 ReAct 的差异
   3. **贯通任务三**：可选地使用任务三中基于 moss-003-sft-plugin 微调过的模型，对比 zero-shot 与微调后的工具调用成功率
6. 知识点
   1. Function calling 协议（OpenAI 兼容格式）
   2. ReAct 范式：Thought / Action / Observation 循环
   3. Prompt 构造与停止条件设计
   4. 错误恢复与重试策略
   5. Token 预算与上下文管理
7. 实验
   1. 手写 ReAct vs Qwen-Agent 原生 function calling 的成功率对比
   2. 不同模型尺寸（1.5B / 7B / 14B）的工具调用准确率
   3. 错误注入下的恢复能力
8. 时间：2 周

---

### 任务六：Mini Coding Agent

> 详细资源、数据下载与自检脚本见 [task-6-coding-agent/](task-6-coding-agent/)

复刻一个极简版 Claude Code，能在本地仓库上理解任务、修改代码、运行测试并迭代。本任务**完全超出**实践书 v2 的覆盖范围，引入 MCP、Skill、Subagent 三层栈。

1. 参考
   1. [SWE-bench](https://www.swebench.com/)
   2. [CodeAct：用代码执行代替 JSON 工具调用](https://arxiv.org/abs/2402.01030)
   3. [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
   4. [smolagents](https://github.com/huggingface/smolagents)（Hugging Face 极简 agent 框架，原生 CodeAct 风格）
   5. [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
   6. [Anthropic Skills](https://github.com/anthropics/skills)（Skill 设计参考，源码可读）
   7. **延伸阅读**：本任务为进阶扩展，教材未覆盖；可阅读 Anthropic 发布的 Claude Code / Skill / Subagent 系列博客作为补充
2. 基座模型：[Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
3. 能力三层栈

   | 层 | 概念 | 角色 |
   | --- | --- | --- |
   | 底层 | **Tools / MCP** | 原子工具，通过 MCP server 接入，无状态、可跨 agent 复用 |
   | 中层 | **Skills** | 组织化的能力包（SKILL.md + scripts + references），按需加载、渐进式披露 |
   | 顶层 | **Subagents** | 独立 context 的子 agent，处理可并行或需隔离的子任务 |

4. 实现要求
   1. 手写一个 **MCP server**（提供工具，如「读写本地代码 + 运行测试 + 跑 git 命令」）
   2. 手写 2-3 个 **Skill**（如 `code-review`、`pr-description-writer`、`test-runner`），并实现一个约 50 行的 mini-Skill 加载器（description 匹配 + 按需加载文件内容到上下文）
   3. 1-2 个 **Subagent**（如把「代码搜索」与「测试执行」分给独立 subagent，主 agent 只看摘要）
   4. 主线 **agentic loop**：`while not done: model -> tool -> observation -> loop`
   5. 主框架用 **Qwen-Agent**，对照阅读 **smolagents** 的 CodeAct 实现
5. 任务范围
   1. 输入：本地 Git 仓库 + 一个 issue 描述
   2. 输出：能自动定位代码、修改、跑测试、生成 patch 的完整 trace
   3. 评测：在 SWE-bench Lite 上抽样跑通几条
6. 知识点
   1. Agentic loop 的本质：循环 + 工具 + 模型自主停机
   2. MCP 协议规范与 server 实现
   3. Skill 设计模式：progressive disclosure、description-based 路由
   4. Subagent 与 context 隔离的工程价值
   5. 长任务的 context compaction 策略
   6. Sandbox 代码执行的安全考虑
   7. Trace 评测：步骤数、成功率、token 消耗
7. 实验
   1. 同一任务在 Q4_K_M 量化 vs FP16 下的成功率差异
   2. 单 agent vs 加 Subagent 的 token 消耗与成功率
   3. 纯 prompt vs 加 Skill 的成功率提升
8. 时间：3-4 周

---

## 整体设计原则

| 维度 | 设计选择 |
| --- | --- |
| 定位 | 独立入门教程，与 NNDL2 + 实践书 v2 形成"理论 → 带读 → 深度练习"三层递进，但无强依赖 |
| 教学路线 | 每项任务「先手写、再对照框架」，理解原理在前 |
| 技术栈 | 本地化优先，依赖 Hugging Face / 阿里 Qwen / 国产开源生态，国内畅通 |
| 数据贯通 | MOSS 系列从任务三贯通到任务五（教格式 → 真调用）；唐诗数据从原 nlp-beginner 任务五延续到新任务二 |
| 模型贯通 | Qwen2.5 体系全程：0.5B（任务 2-3）→ 7B-Instruct（任务 4-5）→ Coder-7B（任务 6） |
| 现代性 | 覆盖 2025-2026 主流概念：RoPE、LoRA、DPO、RAG、ReAct、MCP、Skill、Subagent、CodeAct |
