"""Optional BGE cross-encoder reranking with a lexical offline fallback."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "bge-reranker-base"

# 定义一个 Reranker（重排序器）。
    #
    # Reranker 的作用不是从整个知识库中寻找文档，
    # 而是对 Retriever 第一阶段已经召回的候选结果进行再次排序。
    #
    # 整体流程：
    #
    # 用户问题
    #    ↓
    # Retriever
    #    ↓
    # Top-20 / Top-40 候选结果
    #    ↓
    # Reranker
    #    ↓
    # Top-K
    #
    # 例如：
    #
    # 用户问题
    #    ↓
    # BGE + FAISS
    #    ↓
    # Top-20
    #    ↓
    # CrossEncoder
    #    ↓
    # Top-5
    #
    # 这里的 CrossEncoder 是真正的 Reranker 模型。
class Reranker:
    def __init__(self) -> None:
        self.model = None   # 初始化 Reranker。
        # ============================================================
        # 获取 Reranker 模型路径
        # ============================================================
        model_path = Path(os.getenv("RERANKER_MODEL_PATH", str(MODEL_DIR)))
        # ============================================================
        # 判断本地是否存在 Reranker 模型
        # ============================================================

        # 如果模型目录存在，
        # 就尝试加载 CrossEncoder。
        if model_path.exists():
            from sentence_transformers import CrossEncoder
            # 加载本地 CrossEncoder 模型。
            #
            # CrossEncoder 与前面的 BGE Embedding 模型
            # 工作方式不同。
            #
            # BGE：
            #
            #   query
            #      ↓
            #   向量
            #
            #   document
            #      ↓
            #   向量
            #
            #   两个向量计算相似度
            #
            #
            # CrossEncoder：
            #
            #   query + document
            #          ↓
            #      CrossEncoder
            #          ↓
            #      relevance score
            #
            # 因此 CrossEncoder 更适合用于精排。
            self.model = CrossEncoder(str(model_path))

    def rerank(self, query: str, results: List[dict], k: int) -> List[dict]:
        # ============================================================
        # 情况一：存在真正的 CrossEncoder Reranker 模型
        # ============================================================
        if self.model is not None:
            # 构造 CrossEncoder 的输入。
            #
            # CrossEncoder 的输入格式是：
            #
            #   [query, document]
            #
            # 所以这里把每个候选文本和用户问题组成一对。
            #
            # 例如：
            #
            # query：
            #   "什么是机器学习？"
            #
            # results：
            #   文本A
            #   文本B
            #   文本C
            #
            # 最终输入：
            #
            # [
            #     ["什么是机器学习？", "文本A"],
            #     ["什么是机器学习？", "文本B"],
            #     ["什么是机器学习？", "文本C"]
            # ]
            #
            # CrossEncoder 会分别给每一对计算相关性分数。
            scores = self.model.predict([[query, result["text"]] for result in results])
        # ============================================================
        # 情况二：没有真正的 Reranker 模型
        # ============================================================
        else:
            # 将用户 Query 转换成字符集合。
            #
            # replace(" ", "")：
            #   删除空格。
            #
            # set()：
            #   去掉重复字符。
            #
            # 例如：
            #
            # query = "机器学习是什么"
            #
            # 得到类似：
            #
            # {"机", "器", "学", "习", "是", "什", "么"}
            query_chars = set(query.replace(" ", ""))
            # 没有 CrossEncoder 时，
            # 使用一个简单的规则作为 fallback。
            #
            # 最终分数：
            #
            # rerank_score =
            #
            #     原始检索分数
            #     +
            #     query 与文本的字符重合比例
            #
            # 其中：
            #
            # result["score"]
            #   ↓
            #   第一阶段 BGE / TF-IDF 检索得到的分数
            #
            #
            # len(query_chars & set(result["text"]))
            #   ↓
            #   Query 中有多少不同字符也出现在文本中
            #
            #
            # max(len(query_chars), 1)
            #   ↓
            #   防止 Query 为空导致除零错误。
            scores = [result["score"] + len(query_chars & set(result["text"])) / max(len(query_chars), 1)
                      for result in results]
        ranked = []
        # ============================================================
        # 将 Reranker 分数写回每个结果
        # ============================================================
        for result, score in zip(results, np.asarray(scores).tolist()):
            item = dict(result)  # 创建当前 result 的一个副本。
            item["rerank_score"] = float(score)  
            ranked.append(item) # 将加入 rerank_score 后的结果放入 ranked。
         # ============================================================
        # 根据 Reranker 分数进行最终排序
        # ============================================================

        return sorted(ranked, key=lambda result: result["rerank_score"], reverse=True)[:k]
