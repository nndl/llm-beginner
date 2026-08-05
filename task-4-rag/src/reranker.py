"""Optional BGE cross-encoder reranking with a lexical offline fallback."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "bge-reranker-base"


class Reranker:
    def __init__(self) -> None:
        self.model = None
        model_path = Path(os.getenv("RERANKER_MODEL_PATH", str(MODEL_DIR)))
        if model_path.exists():
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(str(model_path))

    def rerank(self, query: str, results: List[dict], k: int) -> List[dict]:
        if self.model is not None:
            scores = self.model.predict([[query, result["text"]] for result in results])
        else:
            query_chars = set(query.replace(" ", ""))
            scores = [result["score"] + len(query_chars & set(result["text"])) / max(len(query_chars), 1)
                      for result in results]
        ranked = []
        for result, score in zip(results, np.asarray(scores).tolist()):
            item = dict(result)
            item["rerank_score"] = float(score)
            ranked.append(item)
        return sorted(ranked, key=lambda result: result["rerank_score"], reverse=True)[:k]
