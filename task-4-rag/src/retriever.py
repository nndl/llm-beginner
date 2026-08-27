"""Retrieve PDF chunks using BGE/FAISS and optional reranking."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
from scipy import sparse

try:
    import faiss
except ImportError:
    faiss = None

from .indexer import INDEX_DIR, Embedder, build_index
from .reranker import Reranker

# 定义 RAG 的检索器。
    #
    # Retriever 的主要任务：
    #
    #   用户问题
    #      ↓
    #   Query Embedding
    #      ↓
    #   向量数据库 / FAISS
    #      ↓
    #   召回候选文本
    #      ↓
    #   Reranker 二次排序（可选）
    #      ↓
    #   返回 Top-K 文本
    #
    # 它连接了前面几个模块：
    #
    #   PDF
    #    ↓
    #   chunk_text()
    #    ↓
    #   Embedder
    #    ↓
    #   build_index()
    #    ↓
    #   Retriever
    #    ↓
    #   Reranker
    #    ↓
    #   generate()
class Retriever:
    def __init__(self, use_reranker: bool = True) -> None:
        # 初始化 Retriever。
        #
        # use_reranker：
        #   是否启用 Reranker。
        #
        # True：
        #   先进行向量召回，再使用 Reranker 精排。
        #
        # False：
        #   只进行向量检索，不进行二次排序。
        #
        # 默认开启 Reranker。

        # ============================================================
        # 1. 检查向量索引是否已经建立
        # ============================================================

        # 检查 INDEX_DIR 中是否至少存在一种索引文件：
        #
        # chunks.faiss：
        #   BGE + FAISS
        #
        # chunks.npy：
        #   BGE + NumPy
        #
        # chunks.npz：
        #   TF-IDF + Sparse Matrix
        #
        # any(...)：
        #   只要其中一个文件存在，就认为索引已经建立。
        #
        # 如果三种文件都不存在，
        # 就自动调用 build_index() 创建索引。
        if not any((INDEX_DIR / name).exists() for name in ("chunks.faiss", "chunks.npy", "chunks.npz")):
            build_index()  # 如果没有索引，自动从 data/kb.pdf 建立索引。
            
        metadata = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))  # 读取前面 build_index() 生成的 chunks.json。
        # ============================================================
        # 3. 读取文本块
        # ============================================================
        self.chunks = metadata["chunks"]   # 取出所有 PDF 切分后的文本块。
        self.source = metadata["source"]   # 保存原始数据来源。
        self.vector_backend = metadata.get("vector_backend", "faiss")   # 获取向量检索后端。
        # ============================================================
        # 4. 根据不同后端加载向量索引
        # ============================================================

        # -------------------------------
        # 情况一：FAISS
        # -------------------------------
        if self.vector_backend == "faiss" and faiss is not None:
            self.index = faiss.read_index(str(INDEX_DIR / "chunks.faiss"))
        # -------------------------------
        # 情况二：TF-IDF
        # -------------------------------
        elif self.vector_backend == "sparse_tfidf":
            self.index = sparse.load_npz(INDEX_DIR / "chunks.npz")
        # -------------------------------
        # 情况三：NumPy Dense Vector
        # -------------------------------
        else:
            self.index = np.load(INDEX_DIR / "chunks.npy")
        # ============================================================
        # 5. 创建 Embedding 模型
        # ============================================================
        self.embedder = Embedder()
        if not self.embedder.uses_bge:
            with (INDEX_DIR / "tfidf.pkl").open("rb") as file:
                self.embedder.vectorizer = pickle.load(file)
        # ============================================================
        # 6. 创建 Reranker
        # ============================================================
        self.reranker = Reranker() if use_reranker else None

    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        # 根据用户问题检索最相关的文本。
        #
        # query：
        #   用户的问题。
        #
        # k：
        #   最终返回多少条结果。
        #
        # 默认：
        #
        #   Top-5
        #
        # 返回结果类似：
        #
        # [
        #     {
        #         "text": "...",
        #         "score": 0.82,
        #         "source": "data/kb.pdf",
        #         "rerank_score": 0.91
        #     },
        #     ...
        # ]

        # ============================================================
        # 1. 检查输入是否合法
        # ============================================================
        if not query.strip() or k <= 0:
            return []
        # ============================================================
        # 2. 确定第一阶段召回多少候选文本
        # ============================================================
        # 注意这里非常重要。
        #
        # 最终可能只需要 Top-5，
        # 但是这里并不是只召回 5 个。
        #
        # candidate_count：
        #   第一阶段 Retriever 的候选数量。
        #
        # max(k * 4, 20)：
        #
        #   如果 k=5：
        #       max(20,20) = 20
        #
        #   如果 k=10：
        #       max(40,20) = 40
        #
        # 也就是说：
        #
        #   最终需要 Top-5
        #       ↓
        #   先召回 Top-20
        #       ↓
        #   Reranker 再从 Top-20 中挑 Top-5
        #
        # min(len(self.chunks), ...)：
        #   防止候选数量超过知识库中的 chunk 数量。
        candidate_count = min(len(self.chunks), max(k * 4, 20))
        # ============================================================
        # 3. 将用户 Query 转换成向量
        # ============================================================
        # 使用 BGE 或 TF-IDF 将用户问题转换成向量。
        #
        # BGE：
        #
        #   query
        #      ↓
        #   BGE
        #      ↓
        #   Dense Embedding
        #
        # TF-IDF：
        #
        #   query
        #      ↓
        #   TfidfVectorizer
        #      ↓
        #   Sparse Vector
        query_vector = self.embedder.encode_query(query)
        # ============================================================
        # 4. 第一种检索方式：FAISS
        # ============================================================
        if self.vector_backend == "faiss" and faiss is not None:
            query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)  # 对 Query Vector 做 L2 归一化。
            # 使用 FAISS 进行相似度搜索。
            #
            # query_vector：
            #   当前用户问题的向量
            #
            # candidate_count：
            #   要返回多少个候选结果
            #
            # 返回：
            #
            # scores：
            #   每个候选文本的相似度分数
            #
            # indices：
            #   每个候选文本在 chunks 中的下标
            #
            # 例如：
            #
            # scores：
            #   [[0.91, 0.87, 0.83, ...]]
            #
            # indices：
            #   [[15, 82, 3, ...]]
            scores, indices = self.index.search(query_vector, candidate_count)
            # FAISS 返回的是二维数组，
            # 因为允许一次查询多个 Query。
            #
            # 当前这里只查询一个 Query，
            # 所以：
            #
            # scores[0]
            # indices[0]
            ranked = zip(scores[0], indices[0])
        # ============================================================
        # 5. 第二种检索方式：Sparse TF-IDF
        # ============================================================
        elif self.vector_backend == "sparse_tfidf":
            # TF-IDF 使用稀疏矩阵乘法计算相似度。
            #
            # self.index：
            #   所有文档的 TF-IDF 矩阵
            #
            # query_vector.T：
            #   Query 的 TF-IDF 向量转置
            #
            # 两者相乘：
            #
            #   Document Matrix × Query Vector
            #
            # 得到每个文档与 Query 的相似度。
            #
            # toarray().ravel()：
            #   将稀疏结果转换成一维 NumPy 数组。
            scores = (self.index @ query_vector.T).toarray().ravel()
            indices = np.argsort(scores)[::-1][:candidate_count]  # 对所有文档的分数进行降序排序。只保留前 candidate_count 个
            # 创建一个生成器，
            # 将：
            #
            #   score
            #   index
            #
            # 配对。
            ranked = ((scores[index], index) for index in indices)
        # ============================================================
        # 6. 第三种检索方式：NumPy Dense Vector
        # ============================================================
        else:
            query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)  # 对 Query Vector 做 L2 归一化。
            scores = self.index @ query_vector[0]   # 使用矩阵乘法计算 Query 与所有文档的相似度。
            # 对所有文档按照相似度从高到低排序，
            # 只保留前 candidate_count 个。
            indices = np.argsort(scores)[::-1][:candidate_count]
            ranked = ((scores[index], index) for index in indices)
        # ============================================================
        # 7. 将检索结果转换成统一的数据结构
        # ============================================================
        # 将上面三种检索方式得到的结果，
        # 全部转换成统一的 dict 格式。
        #
        # 每个结果包含：
        #
        #   text：
        #       文本内容
        #
        #   score：
        #       第一阶段向量检索分数
        #
        #   source：
        #       原始知识库来源
        #
        # index：
        #   就是 chunks 中的下标。
        results = [{"text": self.chunks[index], "score": float(score), "source": self.source}
                   for score, index in ranked if index >= 0]
        # ============================================================
        # 8. 第二阶段 Rerank
        # ============================================================

        # 如果启用了 Reranker：
        #
        #   candidate_count 个候选
        #       ↓
        #   Reranker
        #       ↓
        #   Top-K
        #
        # 如果没有启用 Reranker：
        #
        #   直接取第一阶段检索的 Top-K。
        return self.reranker.rerank(query, results, k) if self.reranker else results[:k]
