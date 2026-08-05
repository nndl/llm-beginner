"""在 SFT LoRA adapter 基础上执行手写 Direct Preference Optimization。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import build_labels, format_messages
from src.lora import inject_lora, load_lora, save_lora

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B"


def load_pairs(path: Path, limit: int | None) -> list[tuple[str, str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("instruction") or row.get("question")
            chosen = row.get("chosen") or row.get("response_chosen")
            rejected = row.get("rejected") or row.get("response_rejected")
            if prompt and chosen and rejected:
                pairs.append((str(prompt), str(chosen), str(rejected)))
            if limit and len(pairs) >= limit:
                break
    if not pairs:
        raise ValueError(f"没有从 {path} 读取到 prompt/chosen/rejected 偏好对")
    return pairs


def sequence_logprob(model, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=ids.unsqueeze(0)).logits[:, :-1]
    targets = labels[1:].unsqueeze(0)
    valid = targets != -100
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, targets.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
    return (token_logps * valid).sum()


def encode_pair(tokenizer, prompt: str, answer: str, max_length: int):
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    ids = tokenizer(format_messages(messages), truncation=True, max_length=max_length,
                    return_tensors="pt").input_ids[0]
    return ids, build_labels(ids, messages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sft_dir", type=Path, default=ROOT / "ckpt" / "sft")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "ckpt" / "dpo")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    args = parser.parse_args()
    if not (args.sft_dir / "adapter_model.pt").exists():
        raise FileNotFoundError(f"缺少 SFT adapter：{args.sft_dir / 'adapter_model.pt'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    targets = ["q_proj", "v_proj"]
    policy = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    inject_lora(policy, targets, r=args.rank, alpha=args.alpha)
    load_lora(policy, args.sft_dir)
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    margins, losses = [], []
    for _ in range(args.epochs):
        for prompt, chosen, rejected in load_pairs(args.data, args.max_samples):
            chosen_ids, chosen_labels = encode_pair(tokenizer, prompt, chosen, args.max_length)
            rejected_ids, rejected_labels = encode_pair(tokenizer, prompt, rejected, args.max_length)
            chosen_ids, chosen_labels = chosen_ids.to(device), chosen_labels.to(device)
            rejected_ids, rejected_labels = rejected_ids.to(device), rejected_labels.to(device)
            policy_chosen = sequence_logprob(policy, chosen_ids, chosen_labels)
            policy_rejected = sequence_logprob(policy, rejected_ids, rejected_labels)
            with torch.no_grad():
                ref_chosen = sequence_logprob(reference, chosen_ids, chosen_labels)
                ref_rejected = sequence_logprob(reference, rejected_ids, rejected_labels)
            margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            loss = -F.logsigmoid(args.beta * margin)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            margins.append(float(margin.detach().item()))
            losses.append(float(loss.detach().item()))
    save_lora(policy, args.output_dir, r=args.rank, alpha=args.alpha, target_modules=targets)
    (args.output_dir / "train_log.json").write_text(json.dumps({"losses": losses, "reward_margins": margins}, indent=2), encoding="utf-8")
    print(f"DPO adapter 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
