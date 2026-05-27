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

# 下载 embedding / reranker 模型；生成模型与文档自行准备（脚本给提示）
python data/download.py
```

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
| `retriever_recall_at_10` | 在 CMRC 2018 dev 抽样上 Recall@10 > 0.6 |
| `rag_end_to_end` | 端到端能返回 answer 且 sources 非空（手动验证语义） |

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
