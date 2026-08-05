"""在 Qwen2.5-0.5B 上以手写 LoRA 执行监督微调。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import build_labels, format_messages
from src.lora import inject_lora, save_lora

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B"


def messages_from_record(record: dict) -> list[dict] | None:
    """兼容 MOSS 常见 ``conversations``、OpenAI ``messages`` 及 instruction 格式。"""
    moss_chat = record.get("chat")
    if isinstance(moss_chat, dict):
        messages = []
        for turn_name in sorted(moss_chat, key=lambda name: int(name.rsplit("_", 1)[-1])):
            turn = moss_chat[turn_name]
            if not isinstance(turn, dict):
                continue
            user = str(turn.get("Human", "")).replace("<|Human|>:", "").replace("<eoh>", "").strip()
            assistant = str(turn.get("MOSS", "")).replace("<|MOSS|>:", "").replace("<eom>", "").strip()
            if user and assistant:
                messages.extend(({"role": "user", "content": user},
                                 {"role": "assistant", "content": assistant}))
        if messages:
            return messages
    raw = record.get("messages") or record.get("conversations")
    if isinstance(raw, list):
        messages = []
        for turn in raw:
            role = turn.get("role", turn.get("from", ""))
            role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role)
            content = turn.get("content", turn.get("value", ""))
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": str(content)})
        if any(message["role"] == "assistant" for message in messages):
            return messages
    instruction = record.get("instruction") or record.get("prompt")
    answer = record.get("output") or record.get("response") or record.get("answer")
    if instruction and answer:
        return [{"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": str(answer)}]
    return None


def load_sft_records(path: Path, limit: int | None) -> list[list[dict]]:
    records: list[list[dict]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            messages = messages_from_record(json.loads(line))
            if messages:
                records.append(messages)
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"没有从 {path} 读取到可用 SFT 对话")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="MOSS jsonl 或兼容格式 jsonl")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "ckpt" / "sft")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[警告] 正在 CPU 上训练 0.5B 模型，适合 smoke test，不适合完整训练。")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    targets = ["q_proj", "v_proj"]
    inject_lora(model, targets, r=args.rank, alpha=args.alpha)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    records = load_sft_records(args.data, args.max_samples)
    model.train()
    losses = []
    for epoch in range(args.epochs):
        for step, messages in enumerate(records, start=1):
            ids = tokenizer(format_messages(messages), truncation=True, max_length=args.max_length,
                            return_tensors="pt").input_ids[0]
            labels = build_labels(ids, messages)
            if not torch.any(labels != -100):
                continue
            outputs = model(input_ids=ids.unsqueeze(0).to(device), labels=labels.unsqueeze(0).to(device))
            optimizer.zero_grad()
            outputs.loss.backward()
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            losses.append(float(outputs.loss.item()))
            if step % 20 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(records)} loss={losses[-1]:.4f}")
    save_lora(model, args.output_dir, r=args.rank, alpha=args.alpha, target_modules=targets)
    (args.output_dir / "train_log.json").write_text(json.dumps({"losses": losses}, indent=2), encoding="utf-8")
    print(f"SFT adapter 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
