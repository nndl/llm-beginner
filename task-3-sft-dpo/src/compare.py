"""在相同指令下比较 Qwen base、SFT LoRA 与 DPO LoRA 的生成。"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat import format_prompt
from src.lora import inject_lora, load_lora

MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"


def load_variant(name: str, adapter_dir: Path | None, tokenizer, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()
    if adapter_dir is not None:
        config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        inject_lora(model, config["target_modules"], r=config["r"], alpha=config["alpha"])
        load_lora(model, adapter_dir)
    return model


@torch.no_grad()
def answer(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    text = format_prompt([{"role": "user", "content": prompt}])
    inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
    output = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    generated = output[0, inputs.size(1):]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="+", default=["Explain why exercise is good for health.", "请用三点解释什么是深度学习。"])
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "compare_samples.json")
    args = parser.parse_args()
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"缺少基座模型：{MODEL_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    variants = {
        "base": None,
        "sft": ROOT / "ckpt" / "sft",
        "dpo": ROOT / "ckpt" / "dpo",
    }
    results = []
    for name, adapter_dir in variants.items():
        if adapter_dir is not None and not (adapter_dir / "adapter_model.pt").exists():
            raise FileNotFoundError(f"缺少 {name} adapter：{adapter_dir}")
        model = load_variant(name, adapter_dir, tokenizer, device)
        for prompt in args.prompts:
            response = answer(model, tokenizer, prompt, device, args.max_new_tokens)
            results.append({"variant": name, "prompt": prompt, "response": response})
            print(f"\n--- {name}: {prompt} ---\n{response}")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
