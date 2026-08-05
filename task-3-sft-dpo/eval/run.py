"""任务三自检：LoRA 参数量 + loss masking + SFT vs base 输出对比。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # from src.* —— 学生实现
sys.path.insert(0, str(ROOT.parent))    # from _eval_harness —— 共用运行壳

from _eval_harness import run_tests

LORA_MAX_TRAINABLE_RATIO = 0.05   # LoRA 可训练参数占比上限
MASK_RATIO_RANGE = (0.2, 0.9)     # loss mask 比例的合理区间（过低/过高都说明 mask 写错）


def has_model_weights(model_path: Path) -> bool:
    """Qwen 的 config/tokenizer 已下载但权重中断时，不把它误判成可加载模型。"""
    return any((model_path / name).exists()
               for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"))


def test_lora_param_count():
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        return {"test": "lora_param_count", "pass": None,
                "skip": "transformers 未安装；先 pip install -r requirements.txt"}
    from src.lora import inject_lora

    model_path = ROOT / "models" / "Qwen2.5-0.5B"
    if not model_path.exists() or not has_model_weights(model_path):
        return {"test": "lora_param_count", "pass": None,
                "skip": "models/Qwen2.5-0.5B 缺少完整模型权重；重新运行 data/download.py"}
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    inject_lora(model, target_modules=["q_proj", "v_proj"], r=8, alpha=16)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total
    return {"test": "lora_param_count",
            "pass": ratio < LORA_MAX_TRAINABLE_RATIO,
            "trainable": trainable, "total": total, "ratio": round(ratio, 5)}


def test_loss_masking():
    """对 mock 多轮对话验证 build_labels 的 mask 比例合理。"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {"test": "loss_masking", "pass": None,
                "skip": "transformers 未安装；先 pip install -r requirements.txt"}

    model_path = ROOT / "models" / "Qwen2.5-0.5B"
    if not model_path.exists():
        return {"test": "loss_masking", "pass": None,
                "skip": "models/Qwen2.5-0.5B 不存在"}

    from src.chat import format_messages, build_labels

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
    lo, hi = MASK_RATIO_RANGE
    return {"test": "loss_masking",
            "pass": lo < mask_ratio < hi,
            "mask_ratio": round(mask_ratio, 3),
            "note": f"若不在 {MASK_RATIO_RANGE} 范围，请检查 user/system 是否全部 -100"}


def test_sft_vs_base():
    sft_ckpt = ROOT / "ckpt" / "sft"
    if not sft_ckpt.exists():
        return {"test": "sft_vs_base", "pass": None,
                "skip": "ckpt/sft 不存在"}
    # 只校验 SFT 产物存在；输出质量需手动跑 src/compare.py 对比 base 与 SFT，并把结果附在提交里。
    has_artifacts = sft_ckpt.is_dir() and any(sft_ckpt.iterdir())
    return {"test": "sft_vs_base",
            "pass": has_artifacts,
            "note": "仅校验 ckpt/sft 非空；输出质量请手动跑 src/compare.py 对比 base 与 SFT 并附在提交里"}


if __name__ == "__main__":
    run_tests([test_lora_param_count, test_loss_masking, test_sft_vs_base], ROOT)
