"""Public end-to-end RAG entry point."""
from __future__ import annotations

from .generator import generate
from .retriever import Retriever

_retriever: Retriever | None = None

# answer() 是整个 RAG 问答系统的最终入口函数。
    #
    # 输入：
    #   query
    #       用户提出的问题。
    #
    # 输出：
    #   一个字典，包含：
    #
    #   {
    #       "answer": "...",       # LLM 最终生成的答案
    #       "sources": [...]       # 检索到的参考资料
    #   }
    #
    # 整体流程：
    #
    #   用户问题
    #       ↓
    #   Retriever
    #       ↓
    #   Top-3 相关文本
    #       ↓
    #   generate()
    #       ↓
    #   Qwen
    #       ↓
    #   最终答案
    #
    # 所以这个函数相当于把整个 RAG Pipeline 串起来。
def answer(query: str) -> dict:
    # ============================================================
    # 1. 使用全局 Retriever
    # ============================================================

    # 声明这里要使用全局变量 _retriever。
    #
    # _retriever 通常在文件其他位置定义，例如：
    #
    #     _retriever = None
    #
    # 如果不写 global，
    # Python 会把函数内部对 _retriever 的赋值理解成创建局部变量。
    #
    # 这里使用 global 的目的是：
    #
    #   第一次调用 answer()
    #       ↓
    #   创建 Retriever
    #       ↓
    #   保存到全局 _retriever
    #       ↓
    #   后面继续调用 answer()
    #       ↓
    #   直接复用之前创建的 Retriever
    global _retriever
    # ============================================================
    # 2. 第一次调用时初始化 Retriever
    # ============================================================

    # 判断 Retriever 是否已经创建。
    #
    # 如果：
    #
    #     _retriever is None
    #
    # 说明这是第一次调用 answer()，
    # 或者之前的 Retriever 已经被重置。
    if _retriever is None:
        # 创建 Retriever。
        #
        # 这里没有传 use_reranker 参数，
        # 所以使用默认值：
        #
        #     use_reranker=True
        #
        # 也就是说：
        #
        #   Retriever
        #       ↓
        #   第一阶段向量召回
        #       ↓
        #   Reranker 二次排序
        _retriever = Retriever()
    # ============================================================
    # 3. 检索相关资料
    # ============================================================

    # 使用 Retriever 根据用户问题进行检索。
    #
    # k=3：
    #   最终只保留 3 条最相关的资料。
    #
    # 注意：
    #
    #   这里虽然写的是 k=3，
    #   但根据前面的 Retriever 实现：
    #
    #       candidate_count = max(k * 4, 20)
    #
    #   所以实际过程是：
    #
    #       Query
    #         ↓
    #       BGE / TF-IDF
    #         ↓
    #       先召回 Top-20
    #         ↓
    #       Reranker
    #         ↓
    #       最终 Top-3
    #
    # 因此 sources 中最终只有 3 个文本。
    sources = _retriever.retrieve(query, k=3)
    # ============================================================
    # 4. 根据检索结果生成最终答案
    # ============================================================

    # 调用前面定义的 generate() 函数。
    #
    # generate() 会：
    #
    #   1. 把 sources 中的文本拼接成上下文
    #
    #   2. 构造 Prompt：
    #
    #      只能根据下列上下文回答问题...
    #
    #   3. 将 Prompt 发送给：
    #
    #      QWEN_API_URL
    #
    #   4. 默认调用：
    #
    #      qwen2.5:7b-instruct
    #
    #   5. 要求模型只能根据检索到的资料回答
    #
    #   6. 最终生成答案。
    #
    # 所以：
    #
    #     sources
    #        ↓
    #     generate()
    #        ↓
    #     Qwen
    #        ↓
    #     answer
    #
    # generate() 返回的是字符串。
    return {"answer": generate(query, sources), "sources": sources}
