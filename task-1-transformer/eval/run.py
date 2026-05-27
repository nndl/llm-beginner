"""任务一自检：单元测试 + 文本分类准确率。

约定学生实现见同目录 ../README.md「实现约定」一节。
本脚本运行后会输出每项测试结果，并把结果写入 eval/result.json，可附在提交里。
"""
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_attention_correctness():
    """手写 attention 应与 torch 官方实现数值一致（误差 < 1e-5）。"""
    from src.attention import scaled_dot_product_attention

    torch.manual_seed(0)
    B, H, T, D = 2, 4, 8, 16
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)

    out_student = scaled_dot_product_attention(Q, K, V)
    out_ref = F.scaled_dot_product_attention(Q, K, V)

    diff = (out_student - out_ref).abs().max().item()
    return {
        "test": "attention_correctness",
        "pass": diff < 1e-5,
        "max_abs_diff": diff,
    }


def test_causal_mask():
    """causal mask 下，位置 i 的输出不应被位置 j>i 的 V 改动影响。"""
    from src.attention import scaled_dot_product_attention

    torch.manual_seed(0)
    B, H, T, D = 1, 1, 5, 8
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()  # True = 被屏蔽

    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    V2 = V.clone()
    V2[:, :, -1] = 999.0  # 改最后一位
    out2 = scaled_dot_product_attention(Q, K, V2, mask=mask)

    leaked = (out[:, :, :-1] - out2[:, :, :-1]).abs().max().item()
    return {
        "test": "causal_mask",
        "pass": leaked < 1e-6,
        "leaked_diff": leaked,
    }


def test_classifier_accuracy():
    """跑学生训练好的 checkpoint 在 ChnSentiCorp dev set 上的准确率。"""
    ckpt = ROOT / "ckpt" / "best.pt"
    if not ckpt.exists():
        return {"test": "classifier_accuracy", "pass": None,
                "skip": "ckpt/best.pt 不存在，先完成训练步骤"}

    try:
        from src.model import load_for_eval
    except ImportError as e:
        return {"test": "classifier_accuracy", "pass": False,
                "error": f"src/model.py 未导出 load_for_eval: {e}"}

    dev_path = ROOT / "data" / "validation.parquet"
    if not dev_path.exists():
        return {"test": "classifier_accuracy", "pass": None,
                "skip": "data/validation.parquet 不存在，先跑 data/download.py"}

    import pandas as pd
    dev = pd.read_parquet(dev_path)

    model, tokenize_fn = load_for_eval(str(ckpt))
    model.eval()

    correct = 0
    total = len(dev)
    with torch.no_grad():
        for _, row in dev.iterrows():
            ids = tokenize_fn(row["text"])
            logits = model(ids.unsqueeze(0))
            pred = int(logits.argmax(dim=-1).item())
            if pred == int(row["label"]):
                correct += 1

    acc = correct / total
    return {
        "test": "classifier_accuracy",
        "pass": acc >= 0.80,
        "accuracy": round(acc, 4),
        "baseline_reference": 0.85,
    }


def main():
    results = []
    for fn in [test_attention_correctness, test_causal_mask, test_classifier_accuracy]:
        try:
            r = fn()
        except Exception as e:
            r = {"test": fn.__name__.replace("test_", ""),
                 "pass": False, "error": str(e),
                 "trace": traceback.format_exc().splitlines()[-3:]}
        results.append(r)
        if r.get("pass") is True:
            tag = "[通过]"
        elif r.get("pass") is None:
            tag = "[跳过]"
        else:
            tag = "[失败]"
        print(f"{tag} {r['test']}: {json.dumps(r, ensure_ascii=False)}")

    out = ROOT / "eval" / "result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(ROOT)}，可附在 nndl-discussion 提交里。")


if __name__ == "__main__":
    main()
