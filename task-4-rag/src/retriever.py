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


class Retriever:
    def __init__(self, use_reranker: bool = True) -> None:
        if not any((INDEX_DIR / name).exists() for name in ("chunks.faiss", "chunks.npy", "chunks.npz")):
            build_index()
        metadata = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = metadata["chunks"]
        self.source = metadata["source"]
        self.vector_backend = metadata.get("vector_backend", "faiss")
        if self.vector_backend == "faiss" and faiss is not None:
            self.index = faiss.read_index(str(INDEX_DIR / "chunks.faiss"))
        elif self.vector_backend == "sparse_tfidf":
            self.index = sparse.load_npz(INDEX_DIR / "chunks.npz")
        else:
            self.index = np.load(INDEX_DIR / "chunks.npy")
        self.embedder = Embedder()
        if not self.embedder.uses_bge:
            with (INDEX_DIR / "tfidf.pkl").open("rb") as file:
                self.embedder.vectorizer = pickle.load(file)
        self.reranker = Reranker() if use_reranker else None

    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        if not query.strip() or k <= 0:
            return []
        candidate_count = min(len(self.chunks), max(k * 4, 20))
        query_vector = self.embedder.encode_query(query)
        if self.vector_backend == "faiss" and faiss is not None:
            query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
            scores, indices = self.index.search(query_vector, candidate_count)
            ranked = zip(scores[0], indices[0])
        elif self.vector_backend == "sparse_tfidf":
            scores = (self.index @ query_vector.T).toarray().ravel()
            indices = np.argsort(scores)[::-1][:candidate_count]
            ranked = ((scores[index], index) for index in indices)
        else:
            query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
            scores = self.index @ query_vector[0]
            indices = np.argsort(scores)[::-1][:candidate_count]
            ranked = ((scores[index], index) for index in indices)
        results = [{"text": self.chunks[index], "score": float(score), "source": self.source}
                   for score, index in ranked if index >= 0]
        return self.reranker.rerank(query, results, k) if self.reranker else results[:k]
