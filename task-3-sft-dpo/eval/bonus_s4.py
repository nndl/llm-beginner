"""离线评估 SFT-only 与 SFT+DPO 的偏好差异，并绘制 reward-margin 曲线。"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat import build_labels, format_messages
from src.lora import inject_lora, load_lora
from train_dpo import load_pairs

MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"


def sequence_logprob(model, ids: torch.Tensor, labels: torch.Tensor) -> float:
    logits = model(input_ids=ids.unsqueeze(0)).logits[:, :-1]
    targets = labels[1:].unsqueeze(0)
    valid = targets != -100
    token_logps = F.log_softmax(logits, dim=-1)
    token_logps = token_logps.gather(-1, targets.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
    return float((token_logps * valid).sum().item())


def score_variant(adapter_dir: Path, tokenizer, device: torch.device,
                  pairs: list[tuple[str, str, str]], max_length: int) -> list[float]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()
    inject_lora(model, config["target_modules"], r=config["r"], alpha=config["alpha"])
    load_lora(model, adapter_dir)
    margins = []
    with torch.no_grad():
        for prompt, chosen, rejected in pairs:
            values = []
            for answer in (chosen, rejected):
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                ids = tokenizer(format_messages(messages), truncation=True, max_length=max_length,
                                return_tensors="pt").input_ids[0].to(device)
                labels = build_labels(ids.cpu(), messages).to(device)
                values.append(sequence_logprob(model, ids, labels))
            margins.append(values[0] - values[1])
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return margins


def write_curve(values: list[float], output: Path) -> None:
    width, height, padding = 720, 360, 48
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        minimum -= 1.0
        maximum += 1.0
    points = []
    for index, value in enumerate(values):
        x = padding + index * (width - 2 * padding) / max(1, len(values) - 1)
        y = height - padding - (value - minimum) * (height - 2 * padding) / (maximum - minimum)
        points.append(f"{x:.1f},{y:.1f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join((
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360">',
            '<rect width="720" height="360" fill="#f8fafc"/>',
            '<text x="48" y="28" font-family="sans-serif" font-size="18">DPO reward margin by update</text>',
            f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#334155"/>',
            f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#334155"/>',
            f'<polyline fill="none" stroke="#0f766e" stroke-width="3" points="{" ".join(points)}"/>',
            f'<text x="4" y="{padding + 5}" font-family="sans-serif" font-size="12">{maximum:.3f}</text>',
            f'<text x="4" y="{height - padding + 5}" font-family="sans-serif" font-size="12">{minimum:.3f}</text>',
            '<text x="48" y="344" font-family="sans-serif" font-size="12">DPO optimizer update</text>',
            '</svg>',
        )),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dpo_smoke.jsonl")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "bonus_s4.json")
    parser.add_argument("--curve", type=Path, default=ROOT / "eval" / "reward_margin.svg")
    args = parser.parse_args()
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"缺少基座模型：{MODEL_DIR}")
    pairs = load_pairs(args.data, limit=None)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft_margins = score_variant(ROOT / "ckpt" / "sft", tokenizer, device, pairs, args.max_length)
    dpo_margins = score_variant(ROOT / "ckpt" / "dpo", tokenizer, device, pairs, args.max_length)
    log = json.loads((ROOT / "ckpt" / "dpo" / "train_log.json").read_text(encoding="utf-8"))
    updates = log["reward_margins"]
    write_curve(updates, args.curve)
    result = {
        "task": "S4: SFT-only vs SFT+DPO preference comparison",
        "pairs": len(pairs),
        "sft_chosen_minus_rejected": sft_margins,
        "dpo_chosen_minus_rejected": dpo_margins,
        "mean_sft_margin": sum(sft_margins) / len(sft_margins),
        "mean_dpo_margin": sum(dpo_margins) / len(dpo_margins),
        "dpo_training_reward_margins": updates,
        "curve": str(args.curve.relative_to(ROOT)),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()