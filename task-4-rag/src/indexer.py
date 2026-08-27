"""Build and persist a PDF-only vector index for the RAG pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss
except ImportError:
    faiss = None

from .chunker import chunk_text, extract_pdf_text

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    # 定义一个文本向量化器类。
    #
    # 主要功能：
    #   1. 优先使用 BGE 模型生成文本 Embedding
    #   2. 如果本地没有 BGE 模型，则使用 TF-IDF 作为备用方案
    #
    # 这样设计的好处是：
    #   有 BGE → 使用语义向量进行检索
    #   没有 BGE → 仍然可以使用 TF-IDF 完成基本的文本检索

    def __init__(self) -> None:
        self.model = None
        self.vectorizer: TfidfVectorizer | None = None
        model_path = os.getenv("BGE_MODEL_PATH", str(MODEL_DIR))
        if Path(model_path).exists():
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_path)

    @property
    def uses_bge(self) -> bool:
        return self.model is not None

    def fit_documents(self, documents: Sequence[str]) -> None:
        # 对文档进行“训练/拟合”。
        #
        # 注意：
        #   BGE 不需要针对当前文档重新训练，
        #   因此只有在没有 BGE、使用 TF-IDF 时，
        #   才需要执行 fit。
        #
        # documents：
        #   文档字符串序列，例如：
        #
        #   [
        #       "这是第一篇文档",
        #       "这是第二篇文档",
        #       ...
        #   ]
        if self.model is None:
            self.vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                sublinear_tf=True,
                max_features=100_000,
            )
            self.vectorizer.fit(documents)

    def encode_documents(self, documents: Sequence[str]):
        # 将多个文档转换成向量。
        #
        # 如果使用 BGE：
        #   返回 Dense Embedding
        #
        # 如果使用 TF-IDF：
        #   返回 TF-IDF Sparse Matrix
        if self.model is not None:
            return self.model.encode(documents, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer has not been fitted")
        return self.vectorizer.transform(documents).astype("float32")

    def encode_query(self, query: str):
        # 将用户查询 query 转换成 Embedding。
        #
        # 注意：
        #   文档和 query 必须使用相同的向量空间，
        #   才能进行后续相似度计算。
        if self.model is not None:
            return self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer has not been fitted")
        return self.vectorizer.transform([query]).astype("float32")


def build_index(chunk_size: int = 512, overlap: int = 128) -> None:
    # 构建知识库向量索引。
    #
    # chunk_size：
    #   每个文本块的最大字符数，默认 512
    #
    # overlap：
    #   相邻文本块之间重叠的字符数，默认 128
    #
    # 整个流程：
    #   PDF
    #    ↓
    #   提取文本
    #    ↓
    #   文本切块
    #    ↓
    #   BGE / TF-IDF 向量化
    #    ↓
    #   FAISS / NumPy / Sparse TF-IDF
    #    ↓
    #   保存索引和元数据
    pdf_path = DATA_DIR / "kb.pdf"      # 指定知识库 PDF 文件路径
    if not pdf_path.exists():   # 检查 PDF 文件是否存在
        # 如果文件不存在，直接抛出异常并提示如何准备数据
        raise FileNotFoundError("data/kb.pdf is missing; run python data/download.py --skip-models")
    # 第一步：从 PDF 中提取纯文本
    #
    # extract_pdf_text(pdf_path)：
    #   读取 PDF 并提取每一页的文本
    #
    # 第二步：调用 chunk_text()
    #
    # chunk_size：
    #   控制每个 chunk 的最大长度
    #
    # overlap：
    #   控制相邻 chunk 的重叠长度
    #
    # 最终得到：
    #
    # chunks = [
    #     "第一段知识...",
    #     "第二段知识...",
    #     "第三段知识...",
    #     ...
    # ]
    chunks = chunk_text(extract_pdf_text(pdf_path), chunk_size, overlap)
    # 如果 PDF 没有提取出任何有效文本，
    # 则无法继续构建向量索引。
    if not chunks:
        raise RuntimeError("No text could be extracted from data/kb.pdf")

    # 创建 Embedder
    #
    # Embedder 会自动判断：
    #
    #   有 BGE 模型 → 使用 BGE
    #
    #   没有 BGE → 使用 TF-IDF
    embedder = Embedder()
    # 如果使用 TF-IDF，
    # 这里会根据所有 chunks 建立 TF-IDF 的特征空间。
    #
    # 如果使用 BGE，则这个函数基本不会做额外训练。
    embedder.fit_documents(chunks)
    # 将所有文本块转换成向量
    #
    # BGE：
    #   得到 Dense Embedding，例如：
    #   [512维]、[768维] 等
    #
    # TF-IDF：
    #   得到 Sparse TF-IDF Matrix
    vectors = embedder.encode_documents(chunks)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)   # 创建索引保存目录
    # ============================================================
    # 情况一：使用 BGE + FAISS
    # ============================================================
    if embedder.uses_bge and faiss is not None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)   # 计算每个向量的 L2 范数
        vectors = vectors / np.maximum(norms, 1e-12)    # 对所有 Embedding 做 L2 归一化
        index = faiss.IndexFlatIP(vectors.shape[1])  # 创建 FAISS Inner Product 索引
        index.add(vectors)   # 将所有文档向量加入 FAISS 索引
        faiss.write_index(index, str(INDEX_DIR / "chunks.faiss"))  # 将 FAISS 索引保存到磁盘
        vector_backend = "faiss"   # 记录当前使用的向量检索后端
    # ============================================================
    # 情况二：使用 BGE，但没有 FAISS
    # ============================================================
    elif embedder.uses_bge:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        np.save(INDEX_DIR / "chunks.npy", vectors)
        vector_backend = "numpy"
    # ============================================================
    # 情况三：没有 BGE，使用 TF-IDF
    # ============================================================
    else:
        sparse.save_npz(INDEX_DIR / "chunks.npz", vectors)
        vector_backend = "sparse_tfidf"
    # ============================================================
    # 保存文本块以及索引相关的元数据
    # ============================================================
    # 将以下信息保存到 chunks.json：
    #
    # chunks：
    #   原始文本块。
    #
    # source：
    #   这些文本来自哪个 PDF。
    #
    # chunk_size：
    #   当时使用的 chunk 大小。
    #
    # overlap：
    #   当时使用的重叠长度。
    #
    # backend：
    #   使用 BGE 还是 TF-IDF。
    #
    # vector_backend：
    #   实际使用 FAISS、NumPy 还是 Sparse TF-IDF。
    (INDEX_DIR / "chunks.json").write_text(json.dumps({
        "chunks": chunks, "source": "data/kb.pdf", "chunk_size": chunk_size,
        "overlap": overlap, "backend": "bge" if embedder.uses_bge else "tfidf-char",
        "vector_backend": vector_backend,
    }, ensure_ascii=False), encoding="utf-8")
    # ============================================================
    # 如果使用 TF-IDF，额外保存训练好的 vectorizer
    # ============================================================
    if embedder.vectorizer is not None:
        import pickle
        with (INDEX_DIR / "tfidf.pkl").open("wb") as file:
            pickle.dump(embedder.vectorizer, file)
