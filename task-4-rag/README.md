# 任务四：RAG 文档问答

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 目标

构建端到端中文检索增强生成系统，并量化每个环节的提升。**扩展**实践书 v2 RAG 节的最小示例：加入 reranker、chunking 策略消融、RAGAS 评测。

## 前置阅读

- [RAG 综述](https://arxiv.org/abs/2312.10997)
- [BGE Embedding 系列](https://huggingface.co/BAAI)
- [RAGAS 评测框架](https://github.com/explodinggradients/ragas)
- 实践书 v2《大语言模型与智能体》「检索增强生成（RAG）」一节

## 准备

```bash
pip install -r requirements.txt

# 下载 embedding / reranker 模型、NNDL v2 PDF，并检查随任务提供的 gold QA
python data/download.py

# 如果只想先准备 PDF 并检查 gold QA，可跳过模型下载
python data/download.py --skip-models
```

默认知识库是《神经网络与深度学习（第二版）》PDF，下载到 `data/kb.pdf`：

```text
https://github.com/nndl/nndl/releases/download/book-pdf/nndl-v2.pdf
```

评测题目保存在 `data/gold_qa.jsonl`，共 30 条。这些题目基于工作区中的 `../神经网络与深度学习2/` LaTeX 正文设计，每条题目都包含 `source_file` 和 `gold_anchors`；下载脚本会检查 LaTeX 来源锚点，并在可提取 PDF 文本时确认每题至少一个 anchor 能在 `data/kb.pdf` 中命中。学生的 RAG 索引仍应从 `data/kb.pdf` 构建，而不是直接索引 LaTeX 源文档。

## 实施步骤

1. **Chunking**（`src/chunker.py`）：从 PDF 提取文本，做固定大小 / 递归 / 语义切分
2. **索引**（`src/indexer.py`）：embedding + FAISS 索引建立
3. **检索**（`src/retriever.py`）：query embedding + top-k 召回
4. **Rerank**（`src/reranker.py`）：bge-reranker 精排
5. **生成**（`src/generator.py`）：把检索结果拼成 prompt，调本地 Qwen2.5-7B-Instruct
6. **端到端**（`src/rag.py`）：串起来 + 评测

## 实现约定

| 文件 | 必须导出 |
|---|---|
| `src/chunker.py` | `chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]` |
| `src/retriever.py` | `class Retriever` 含 `retrieve(query: str, k: int) -> List[dict]`（每个 dict 含 `text`、`score`、`source`） |
| `src/rag.py` | `answer(query: str) -> dict(answer: str, sources: List[dict])` |

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 |
|---|---|
| `chunking_sanity` | chunk 数 > 10；平均长度在 (chunk_size * 0.5, chunk_size * 1.2) |
| `nndl_gold_recall_at_10` | 在 `data/gold_qa.jsonl` 上 Recall@10 > 0.6；同时输出 Recall@1/3/5/10 与 MRR；召回 chunk 需命中至少一个 gold anchor |
| `rag_end_to_end` | 端到端能返回 answer 且 sources 非空（手动验证语义） |

## 客观评价口径

把检索和生成拆开评估。检索部分使用固定的 `data/gold_qa.jsonl`：每个问题都有从 LaTeX 正文抽取的 `gold_anchors`，但系统只能索引 `data/kb.pdf`；如果 top-k chunk 中出现任一 gold anchor，就记为该题命中。这样可以稳定报告 Recall@1/3/5/10 和 MRR，指标不依赖生成模型的表达能力。

生成部分不要让流畅答案掩盖检索失败。建议基于同一批题目检查：答案是否覆盖 `answer` 中的关键点、是否能被返回 sources 支持、是否出现上下文外断言，以及不知道时是否拒答。RAGAS 或 LLM-as-judge 可以作为辅助，但 judge 提示必须只允许依据检索证据打分，最终仍以 gold anchor 召回作为稳定核心指标。

## AI Tutor 反馈

`eval/tutor_prompt.md`。

## 实验建议

- Chunk size 扫描（128 / 256 / 512 / 1024 token）
- 加 / 不加 reranker 的 Recall + faithfulness
- Query rewriting / HyDE 的有效性
- 用 RAGAS 打端到端分数

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions)。

## 时间

约 2 周。
