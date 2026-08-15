"""量化比较 base 与 SFT：assistant-token NLL/PPL 与逐样本胜率。"""
from __future__ import annotations

import argparse
import json
import math
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
from train_sft import messages_from_record


def has_model_weights(model_path: Path) -> bool:
    return any((model_path / name).exists() for name in (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
    ))


def load_records(path: Path, max_samples: int, skip_records: int = 0) -> list[list[dict]]:
    records: list[list[dict]] = []
    seen_valid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            msg = messages_from_record(json.loads(line))
            if not msg:
                continue
            if seen_valid < skip_records:
                seen_valid += 1
                continue
            records.append(msg)
            if len(records) >= max_samples:
                break
    if not records:
        raise ValueError(f"没有从 {path} 读取到可用样本")
    return records


def build_sample_tensors(tokenizer, messages: list[dict], max_length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    text = format_messages(messages)
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids[0]
    labels = build_labels(input_ids, messages)
    assistant_tokens = int((labels != -100).sum().item())
    return input_ids, labels, assistant_tokens


@torch.no_grad()
def seq_nll(model, input_ids: torch.Tensor, labels: torch.Tensor, device: torch.device) -> tuple[float, int]:
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    logits = model(input_ids=input_ids.unsqueeze(0)).logits[0]
    shift_logits = logits[:-1, :]
    shift_labels = labels[1:]
    valid = shift_labels != -100
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return 0.0, 0
    loss_sum = F.cross_entropy(
        shift_logits[valid],
        shift_labels[valid],
        reduction="sum",
    ).item()
    return loss_sum, valid_count


def load_sft_model(model_dir: Path, sft_dir: Path, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(device).eval()
    cfg = json.loads((sft_dir / "adapter_config.json").read_text(encoding="utf-8"))
    inject_lora(model, cfg["target_modules"], r=cfg["r"], alpha=cfg["alpha"])
    load_lora(model, sft_dir)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "moss-sft" / "extracted" / "moss-003-sft-no-tools.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "Qwen2.5-0.5B")
    parser.add_argument("--sft", type=Path, default=ROOT / "ckpt" / "sft")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--scan_multiplier", type=int, default=20)
    parser.add_argument("--skip_records", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "sft_vs_base_metrics.json")
    args = parser.parse_args()

    if not args.model.exists() or not has_model_weights(args.model):
        raise FileNotFoundError(f"模型权重缺失：{args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"数据文件不存在：{args.data}")
    if not (args.sft / "adapter_model.pt").exists() or not (args.sft / "adapter_config.json").exists():
        raise FileNotFoundError(f"SFT adapter 缺失：{args.sft}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))

    scan_count = max(args.max_samples * args.scan_multiplier, args.max_samples)
    records = load_records(args.data, scan_count, skip_records=args.skip_records)
    sample_tensors = []
    skipped_zero_assistant = 0
    for messages in records:
        input_ids, labels, assistant_tokens = build_sample_tensors(tokenizer, messages, args.max_length)
        if assistant_tokens == 0:
            skipped_zero_assistant += 1
            continue
        sample_tensors.append((input_ids, labels, assistant_tokens))
        if len(sample_tensors) >= args.max_samples:
            break
    if not sample_tensors:
        raise ValueError("所有候选样本在截断后 assistant token 都为 0；请增大 --max_length")

    base = AutoModelForCausalLM.from_pretrained(str(args.model)).to(device).eval()
    base_loss_sums: list[float] = []
    base_token_counts: list[int] = []
    for input_ids, labels, _ in sample_tensors:
        loss_sum, token_count = seq_nll(base, input_ids, labels, device)
        base_loss_sums.append(loss_sum)
        base_token_counts.append(token_count)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sft = load_sft_model(args.model, args.sft, device)
    sft_loss_sums: list[float] = []
    sft_token_counts: list[int] = []
    for input_ids, labels, _ in sample_tensors:
        loss_sum, token_count = seq_nll(sft, input_ids, labels, device)
        sft_loss_sums.append(loss_sum)
        sft_token_counts.append(token_count)
    del sft
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_total_loss = float(sum(base_loss_sums))
    sft_total_loss = float(sum(sft_loss_sums))
    base_total_tokens = int(sum(base_token_counts))
    sft_total_tokens = int(sum(sft_token_counts))
    if base_total_tokens == 0 or sft_total_tokens == 0:
        raise ValueError("有效 assistant token 数为 0，无法比较")

    base_mean_nll = base_total_loss / base_total_tokens
    sft_mean_nll = sft_total_loss / sft_total_tokens
    base_ppl = float(math.exp(base_mean_nll))
    sft_ppl = float(math.exp(sft_mean_nll))

    wins = 0
    ties = 0
    total = 0
    per_sample = []
    for idx, (b_sum, b_cnt, s_sum, s_cnt) in enumerate(zip(base_loss_sums, base_token_counts, sft_loss_sums, sft_token_counts), start=1):
        if b_cnt == 0 or s_cnt == 0:
            continue
        b_nll = b_sum / b_cnt
        s_nll = s_sum / s_cnt
        delta = b_nll - s_nll
        total += 1
        if delta > 0:
            wins += 1
            verdict = "sft_better"
        elif abs(delta) < 1e-12:
            ties += 1
            verdict = "tie"
        else:
            verdict = "base_better"
        per_sample.append({
            "sample_id": idx,
            "assistant_tokens": b_cnt,
            "base_nll": round(b_nll, 6),
            "sft_nll": round(s_nll, 6),
            "delta_base_minus_sft": round(delta, 6),
            "verdict": verdict,
        })

    result = {
        "data": str(args.data),
        "max_samples": args.max_samples,
        "effective_samples": total,
        "skipped_zero_assistant": skipped_zero_assistant,
        "skip_records": args.skip_records,
        "max_length": args.max_length,
        "device": str(device),
        "base": {
            "assistant_tokens": base_total_tokens,
            "mean_nll": round(base_mean_nll, 6),
            "ppl": round(base_ppl, 6),
        },
        "sft": {
            "assistant_tokens": sft_total_tokens,
            "mean_nll": round(sft_mean_nll, 6),
            "ppl": round(sft_ppl, 6),
        },
        "delta": {
            "mean_nll_base_minus_sft": round(base_mean_nll - sft_mean_nll, 6),
            "ppl_base_minus_sft": round(base_ppl - sft_ppl, 6),
        },
        "sample_winrate": {
            "sft_better_count": wins,
            "tie_count": ties,
            "base_better_count": total - wins - ties,
            "sft_better_rate": round(wins / total, 6) if total > 0 else None,
        },
        "per_sample": per_sample,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
