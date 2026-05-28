"""任务四自检：chunking 合理性 + NNDL gold 召回 + 端到端能否返回答案。"""
import json
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def normalize_text(text):
    return re.sub(r"\s+", "", str(text))


def load_gold_qa():
    path = ROOT / "data" / "gold_qa.jsonl"
    if not path.exists():
        return None, "data/gold_qa.jsonl 不存在；先跑 data/download.py"
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items, None


def test_chunking_sanity():
    from src.chunker import chunk_text
    chunk_size = 256
    sample = "这是一段测试文本。" * 400  # 足够长，避免短样本文本让 chunk 数阈值失真
    chunks = chunk_text(sample, chunk_size=chunk_size, overlap=32)
    if len(chunks) <= 10:
        return {"test": "chunking_sanity", "pass": False,
                "chunks": len(chunks), "error": "chunk 数太少"}
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    ok = chunk_size * 0.5 <= avg_len <= chunk_size * 1.2
    return {"test": "chunking_sanity", "pass": ok,
            "chunks": len(chunks), "avg_len": round(avg_len, 1),
            "expected_avg_range": [chunk_size * 0.5, chunk_size * 1.2]}


def test_nndl_gold_recall():
    from src.retriever import Retriever
    gold, skip = load_gold_qa()
    if skip:
        return {"test": "nndl_gold_recall_at_10", "pass": None, "skip": skip}
    if not gold:
        return {"test": "nndl_gold_recall_at_10", "pass": False,
                "error": "data/gold_qa.jsonl 为空"}

    retriever = Retriever()  # 学生 init 时已加载索引
    hit_counts = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    details = []
    for item in gold:
        anchors = [(a, normalize_text(a)) for a in item.get("gold_anchors", [])]
        results = retriever.retrieve(item["question"], k=10)
        matched = None
        matched_rank = None
        for rank, result in enumerate(results, 1):
            text = normalize_text(result.get("text", ""))
            for raw_anchor, anchor in anchors:
                if anchor and anchor in text:
                    matched = raw_anchor
                    matched_rank = rank
                    break
            if matched:
                break
        if matched_rank:
            for k in hit_counts:
                if matched_rank <= k:
                    hit_counts[k] += 1
            reciprocal_ranks.append(1 / matched_rank)
        else:
            reciprocal_ranks.append(0)
        details.append({
            "id": item.get("id"),
            "hit": matched is not None,
            "rank": matched_rank,
            "matched_anchor": matched,
            "source_file": item.get("source_file"),
        })
    metrics = {
        f"recall_at_{k}": round(hit_counts[k] / len(gold), 3)
        for k in sorted(hit_counts)
    }
    metrics["mrr"] = round(sum(reciprocal_ranks) / len(gold), 3)
    return {"test": "nndl_gold_recall_at_10",
            "pass": metrics["recall_at_10"] > 0.6,
            "n": len(gold), **metrics, "details": details}


def test_rag_end_to_end():
    from src.rag import answer
    try:
        gold, _ = load_gold_qa()
        query = gold[0]["question"] if gold else "什么是深度学习？"
        r = answer(query)
        ok = isinstance(r, dict) and "answer" in r and r.get("sources")
        return {"test": "rag_end_to_end", "pass": ok,
                "answer_preview": str(r.get("answer", ""))[:120]}
    except Exception as e:
        return {"test": "rag_end_to_end", "pass": False, "error": str(e)}


def main():
    results = []
    for fn in [test_chunking_sanity, test_nndl_gold_recall, test_rag_end_to_end]:
        try:
            r = fn()
        except Exception as e:
            r = {"test": fn.__name__.replace("test_", ""),
                 "pass": False, "error": str(e),
                 "trace": traceback.format_exc().splitlines()[-3:]}
        results.append(r)
        tag = "[通过]" if r.get("pass") is True else \
              ("[跳过]" if r.get("pass") is None else "[失败]")
        print(f"{tag} {r['test']}: {json.dumps(r, ensure_ascii=False)}")
    out = ROOT / "eval" / "result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
