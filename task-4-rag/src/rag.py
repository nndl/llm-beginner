"""Public end-to-end RAG entry point."""
from __future__ import annotations

from .generator import generate
from .retriever import Retriever

_retriever: Retriever | None = None


def answer(query: str) -> dict:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    sources = _retriever.retrieve(query, k=3)
    return {"answer": generate(query, sources), "sources": sources}
