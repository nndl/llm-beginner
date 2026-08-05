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
    """BGE embedder with a deterministic local TF-IDF fallback."""

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
        if self.model is None:
            self.vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                sublinear_tf=True,
                max_features=100_000,
            )
            self.vectorizer.fit(documents)

    def encode_documents(self, documents: Sequence[str]):
        if self.model is not None:
            return self.model.encode(documents, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer has not been fitted")
        return self.vectorizer.transform(documents).astype("float32")

    def encode_query(self, query: str):
        if self.model is not None:
            return self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer has not been fitted")
        return self.vectorizer.transform([query]).astype("float32")


def build_index(chunk_size: int = 512, overlap: int = 128) -> None:
    """Create an inner-product FAISS index from `data/kb.pdf` only."""
    pdf_path = DATA_DIR / "kb.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError("data/kb.pdf is missing; run python data/download.py --skip-models")
    chunks = chunk_text(extract_pdf_text(pdf_path), chunk_size, overlap)
    if not chunks:
        raise RuntimeError("No text could be extracted from data/kb.pdf")
    embedder = Embedder()
    embedder.fit_documents(chunks)
    vectors = embedder.encode_documents(chunks)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if embedder.uses_bge and faiss is not None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(INDEX_DIR / "chunks.faiss"))
        vector_backend = "faiss"
    elif embedder.uses_bge:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        np.save(INDEX_DIR / "chunks.npy", vectors)
        vector_backend = "numpy"
    else:
        sparse.save_npz(INDEX_DIR / "chunks.npz", vectors)
        vector_backend = "sparse_tfidf"
    (INDEX_DIR / "chunks.json").write_text(json.dumps({
        "chunks": chunks, "source": "data/kb.pdf", "chunk_size": chunk_size,
        "overlap": overlap, "backend": "bge" if embedder.uses_bge else "tfidf-char",
        "vector_backend": vector_backend,
    }, ensure_ascii=False), encoding="utf-8")
    if embedder.vectorizer is not None:
        import pickle
        with (INDEX_DIR / "tfidf.pkl").open("wb") as file:
            pickle.dump(embedder.vectorizer, file)
