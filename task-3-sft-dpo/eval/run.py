"""任务三自检：LoRA 参数量 + loss masking + SFT vs base 输出对比。"""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_lora_param_count():
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        return {"test": "lora_param_count", "pass": False,
                "error": "pip install transformers"}
    from src.lora import inject_lora

    model_path = ROOT / "models" / "Qwen2.5-0.5B"
    if not model_path.exists():
        return {"test": "lora_param_count", "pass": None,
                "skip": "models/Qwen2.5-0.5B 不存在"}
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    inject_lora(model, target_modules=["q_proj", "v_proj"], r=8, alpha=16)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total
    return {"test": "lora_param_count",
            "pass": ratio < 0.05,
            "trainable": trainable, "total": total, "ratio": round(ratio, 5)}


def test_loss_masking():
    """对 mock 多轮对话验证 build_labels 的 mask 比例合理。"""
    from src.chat import format_messages, build_labels
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {"test": "loss_masking", "pass": False,
                "error": "pip install transformers"}

    model_path = ROOT / "models" / "Qwen2.5-0.5B"
    if not model_path.exists():
        return {"test": "loss_masking", "pass": None,
                "skip": "models/Qwen2.5-0.5B 不存在"}

    msgs = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！很高兴见到你。"},
        {"role": "user", "content": "请介绍一下深度学习"},
        {"role": "assistant", "content": "深度学习是机器学习的一个分支，使用多层神经网络。"},
    ]
    text = format_messages(msgs)
    tok = AutoTokenizer.from_pretrained(str(model_path))
    ids = tok(text, return_tensors="pt").input_ids[0]
    labels = build_labels(ids, msgs)

    if labels.shape != ids.shape:
        return {"test": "loss_masking", "pass": False,
                "error": f"labels shape {labels.shape} != ids shape {ids.shape}"}

    mask_ratio = (labels == -100).float().mean().item()
    return {"test": "loss_masking",
            "pass": 0.2 < mask_ratio < 0.9,
            "mask_ratio": round(mask_ratio, 3),
            "note": "若不在 (0.2, 0.9) 范围，请检查 user/system 是否全部 -100"}


def test_sft_vs_base():
    sft_ckpt = ROOT / "ckpt" / "sft"
    if not sft_ckpt.exists():
        return {"test": "sft_vs_base", "pass": None,
                "skip": "ckpt/sft 不存在"}
    return {"test": "sft_vs_base", "pass": None,
            "skip": "请手动跑 src/compare.py 对比 base 与 SFT 在固定指令上的输出，附在提交里"}


def main():
    results = []
    for fn in [test_lora_param_count, test_loss_masking, test_sft_vs_base]:
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
