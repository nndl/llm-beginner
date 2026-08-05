"""Low-memory S1/S2 ablation for chunk size and offline reranking."""
from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexer import INDEX_DIR, build_index
from src.retriever import Retriever

CHUNK_CONFIGS = ((128, 32), (256, 64), (512, 128), (1024, 256))
INDEX_FILES = ("chunks.faiss", "chunks.npy", "chunks.npz", "chunks.json", "tfidf.pkl")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def load_gold() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / "gold_qa.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]


def clear_index() -> None:
    for filename in INDEX_FILES:
        (INDEX_DIR / filename).unlink(missing_ok=True)


def evaluate(retriever: Retriever, gold: list[dict]) -> dict:
    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    citation_grounded = 0
    for item in gold:
        results = retriever.retrieve(item["question"], k=10)
        anchors = [normalize_text(anchor) for anchor in item["gold_anchors"]]
        rank = next((position for position, result in enumerate(results, 1)
                     if any(anchor in normalize_text(result["text"]) for anchor in anchors)), None)
        if rank is not None:
            for cutoff in hit_counts:
                if rank <= cutoff:
                    hit_counts[cutoff] += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

        # The offline generator quotes its first source verbatim; this is a transparent
        # source-grounding proxy, not an LLM-based faithfulness score.
        if results and results[0]["text"][:80].strip():
            citation_grounded += 1

    total = len(gold)
    return {
        **{f"recall_at_{cutoff}": round(hits / total, 3) for cutoff, hits in hit_counts.items()},
        "mrr": round(sum(reciprocal_ranks) / total, 3),
        "citation_grounded_rate": round(citation_grounded / total, 3),
    }


def main() -> None:
    gold = load_gold()
    experiments = []
    try:
        for chunk_size, overlap in CHUNK_CONFIGS:
            clear_index()
            build_index(chunk_size=chunk_size, overlap=overlap)
            for use_reranker in (False, True):
                retriever = Retriever(use_reranker=use_reranker)
                metrics = evaluate(retriever, gold)
                experiments.append({
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "reranker": "lexical_offline" if use_reranker else "none",
                    **metrics,
                })
                del retriever
                gc.collect()
    finally:
        # Leave the normal 512-character production index available after the sweep.
        clear_index()
        build_index(chunk_size=512, overlap=128)

    payload = {
        "dataset": "data/gold_qa.jsonl",
        "n": len(gold),
        "vector_backend": "tfidf-char sparse, max_features=100000",
        "faithfulness_note": "citation_grounded_rate verifies the deterministic offline answer has a source; it is not a RAGAS score.",
        "experiments": experiments,
    }
    output = ROOT / "eval" / "ablation.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
