"""任务四自检：chunking 合理性 + 检索 Recall@k + 端到端能否返回答案。"""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_chunking_sanity():
    from src.chunker import chunk_text
    sample = "这是一段测试文本。" * 200  # ~2000 字
    chunks = chunk_text(sample, chunk_size=256, overlap=32)
    if len(chunks) < 5:
        return {"test": "chunking_sanity", "pass": False,
                "chunks": len(chunks), "error": "chunk 数太少"}
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    ok = 128 <= avg_len <= 320
    return {"test": "chunking_sanity", "pass": ok,
            "chunks": len(chunks), "avg_len": round(avg_len, 1)}


def test_retriever_recall():
    from src.retriever import Retriever
    cmrc = ROOT / "data" / "cmrc-dev.parquet"
    if not cmrc.exists():
        return {"test": "retriever_recall_at_10", "pass": None,
                "skip": "data/cmrc-dev.parquet 不存在"}
    try:
        import pandas as pd
        df = pd.read_parquet(cmrc).head(50)  # 抽 50 条
    except Exception as e:
        return {"test": "retriever_recall_at_10", "pass": False, "error": str(e)}

    retriever = Retriever()  # 学生 init 时已加载索引
    hits = 0
    for _, row in df.iterrows():
        ctx_gold = row["context"][:200]  # 用 context 前缀作 anchor
        results = retriever.retrieve(row["question"], k=10)
        if any(ctx_gold[:50] in r["text"] for r in results):
            hits += 1
    recall = hits / len(df)
    return {"test": "retriever_recall_at_10", "pass": recall > 0.6,
            "recall": round(recall, 3), "n": len(df)}


def test_rag_end_to_end():
    from src.rag import answer
    try:
        r = answer("什么是深度学习？")
        ok = isinstance(r, dict) and "answer" in r and r.get("sources")
        return {"test": "rag_end_to_end", "pass": ok,
                "answer_preview": str(r.get("answer", ""))[:120]}
    except Exception as e:
        return {"test": "rag_end_to_end", "pass": False, "error": str(e)}


def main():
    results = []
    for fn in [test_chunking_sanity, test_retriever_recall, test_rag_end_to_end]:
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
