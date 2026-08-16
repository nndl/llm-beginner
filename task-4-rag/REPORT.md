# 任务四：RAG 文档问答报告（按 README 任务分项）

## 1. 必做任务（M1-M4）完成情况

| 任务 | 目标 | 完成状态 | 证据 |
|---|---|---|---|
| M1 | 实现 chunk_text，且 chunking_sanity 通过 | 已完成 | chunking_sanity: pass=true，chunks=17，avg_len=241.9（期望区间 128.0-307.2） |
| M2 | 从 data/kb.pdf 建索引并实现 Retriever.retrieve | 已完成 | 索引元数据 source=data/kb.pdf；检索返回 text/score/source |
| M3 | nndl_gold_recall_at_10 通过（Recall@10 > 0.6） | 已完成 | Recall@1/3/5/10=0.433/0.533/0.700/0.833，MRR=0.538 |
| M4 | 端到端 answer 返回非空 answer 和 sources | 已完成 | rag_end_to_end: pass=true，answer_preview 非空 |

## 2. M1：Chunking（PDF 文本切分）

### 2.1 实现内容

1. 从 PDF 抽取文本：按页提取后拼接，未使用 LaTeX 源文档。
2. 文本规范化：统一换行、合并多空格、压缩连续空行。
3. 按字符切分：chunk_size/overlap 均按字符计。
4. 断点优化：优先在段落边界或中文句号/问号/叹号处截断，减少语义截断。

### 2.2 自检结果

1. chunk 数量 17（>10）。
2. 平均 chunk 长度 241.9，落在自检要求区间内。

结论：M1 满足 DoD。

## 3. M2：索引构建与召回实现

### 3.1 实现内容

1. 只用 data/kb.pdf 构建知识库。
2. 向量化策略：优先 BGE；若本地模型不可用，退化到字符级 TF-IDF（2-5 gram，max_features=100000）。
3. 检索后端：优先 FAISS 内积索引；不可用时回退到 numpy/sparse。
4. Retriever.retrieve(query, k)：支持 top-k 召回，输出每条包含 text、score、source。
5. 候选池策略：先召回 max(k*4, 20) 作为候选，再进入可选 rerank。

### 3.2 结果与说明

1. 索引来源满足任务约束（PDF-only）。
2. 检索接口格式满足 README 约定。

结论：M2 满足 DoD。

## 4. M3：检索评测（Recall@k 与 MRR）

### 4.1 主实验结果（eval/result.json）

1. 样本数 n=30。
2. Recall@1=0.433。
3. Recall@3=0.533。
4. Recall@5=0.700。
5. Recall@10=0.833（通过阈值 > 0.6）。
6. MRR=0.538。

### 4.2 未命中样本观察

在 30 条题目中，仍有少量问题在 top-10 未命中 gold anchor（例如 learning-types、gradient-issues、position-encoding-rope、causal-mask、gnn-layer-depth）。主要原因可能是：

1. PDF 抽文本的版式噪声导致关键短语断裂。
2. 锚点跨 chunk 边界时，当前 overlap 仍不足以完整覆盖。
3. 纯词面特征对部分抽象问题语义匹配能力有限。

结论：M3 已达成 DoD，并保留后续优化空间。

## 5. M4：端到端 RAG 回答

### 5.1 实现内容

1. answer(query) 串联：retrieve -> generate。
2. prompt 约束：明确“只能依据给定上下文，不足则拒答”。
3. 输出结构：返回 {answer, sources}，保证评测接口可用。
4. 兜底策略：若本地 Qwen 接口不可用，返回基于首条来源的可追溯离线答案。

### 5.2 自检结果

1. rag_end_to_end 通过。
2. answer 与 sources 均非空。

结论：M4 满足 DoD。

RAG 架构
```text
                 ┌──────────────┐
                 │  data/kb.pdf │
                 └──────┬───────┘
                        ↓
                 PDF文本提取
                        ↓
                    文本切块
                        ↓
                 BGE Embedding
                        ↓
                 FAISS 建立索引
                        │
                        │
用户 Query ─────────────┤
                        ↓
                    Retriever
                        ↓
                   Top-20 召回
                        ↓
                    Reranker
                        ↓
                      Top-3
                        ↓
                   generate()
                        ↓
                    Qwen API
                        ↓
                最终答案 + Sources
```
## 6. 加分项（S1-S4）

### S1：chunk_size 扫描（已完成）

不加 reranker 的结果如下：

| chunk/overlap | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| 128/32 | 0.300 | 0.433 | 0.533 | 0.600 | 0.390 |
| 256/64 | 0.367 | 0.567 | 0.633 | 0.800 | 0.497 |
| 512/128 | 0.467 | 0.767 | 0.800 | 0.867 | 0.609 |
| 1024/256 | 0.633 | 0.800 | 0.867 | 0.933 | 0.735 |

观察：随着 chunk 增大，召回整体提升明显，在当前题集上 1024/256 最优。

### S2：有无 reranker 对比（已完成）

| chunk/overlap | 无 reranker Recall@10 | lexical_offline Recall@10 | 变化 |
|---|---:|---:|---:|
| 128/32 | 0.600 | 0.700 | +0.100 |
| 256/64 | 0.800 | 0.833 | +0.033 |
| 512/128 | 0.867 | 0.833 | -0.034 |
| 1024/256 | 0.933 | 0.933 | +0.000 |

观察：小 chunk 下 rerank 有增益；中大 chunk 下增益变弱，甚至可能扰动原始排序。

### S3：Query rewriting / HyDE（未完成）

当前未实现 query 改写链路，后续可补充“原始 query vs 改写 query”对照实验。

### S4：RAGAS faithfulness / answer relevancy（未完成）

当前仅记录 citation_grounded_rate=1.0（可追溯性代理指标），尚未接入 RAGAS 的语义忠实性评分。

## 7. 200-500 字实验观察

本次任务严格使用 data/kb.pdf 构建索引，避免了直接索引 LaTeX 源文档带来的指标失真。整体上，检索质量对端到端效果起决定作用：在 S1 扫描中，chunk 从 128/32 增至 1024/256 时，Recall@10 从 0.600 提升到 0.933，MRR 从 0.390 提升到 0.735，说明更大的文本块在这批题目上更容易保留完整 gold anchor。S2 显示 reranker 不是总能提分：小块文本（128、256）下 lexical_offline 可带来 0.033-0.100 的 Recall@10 增益，但在 512 块上反而下降，表明当召回候选已较稳定时，额外词面重排可能引入噪声。主实验最终配置达成 Recall@10=0.833，超过 DoD 阈值。当前主要短板在少数长依赖问题（如位置编码、因果掩码）命中不稳，后续可通过更稳健的跨块拼接、query rewriting 以及真正语义级 reranker 进一步提升，并补齐 RAGAS 评测以更完整地衡量回答忠实性。