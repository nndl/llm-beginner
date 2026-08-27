"""Benchmark greedy generation with and without the KV cache."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import load_for_eval


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model, prompt_ids: list[int], max_new_tokens: int,
            use_kv_cache: bool, runs: int, device: torch.device) -> list[float]:
    durations = []
    for _ in range(runs):
        synchronize(device)
        started = time.perf_counter()
        model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_kv_cache=use_kv_cache,
        )
        synchronize(device)
        durations.append(time.perf_counter() - started)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare equivalent greedy generation with and without KV cache."
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpt" / "best.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "eval" / "kv_cache_benchmark.json"
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("max-new-tokens and runs must be positive; warmup cannot be negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    model, tokenizer = load_for_eval(str(args.checkpoint))
    model.to(device).eval()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        parser.error("prompt must encode to at least one token")

    cached_output = model.generate(
        prompt_ids, max_new_tokens=args.max_new_tokens, temperature=0.0, use_kv_cache=True
    )
    uncached_output = model.generate(
        prompt_ids, max_new_tokens=args.max_new_tokens, temperature=0.0, use_kv_cache=False
    )
    if cached_output != uncached_output:
        raise RuntimeError("KV-cache and non-cache greedy outputs differ; refusing to time them")

    for _ in range(args.warmup):
        model.generate(
            prompt_ids, max_new_tokens=args.max_new_tokens, temperature=0.0, use_kv_cache=True
        )
        model.generate(
            prompt_ids, max_new_tokens=args.max_new_tokens, temperature=0.0, use_kv_cache=False
        )

    cache_times = measure(
        model, prompt_ids, args.max_new_tokens, True, args.runs, device
    )
    no_cache_times = measure(
        model, prompt_ids, args.max_new_tokens, False, args.runs, device
    )
    cache_median = statistics.median(cache_times)
    no_cache_median = statistics.median(no_cache_times)
    result = {
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "new_tokens": args.max_new_tokens,
        "runs": args.runs,
        "warmup": args.warmup,
        "device": str(device),
        "greedy_outputs_equal": True,
        "cache_median_seconds": round(cache_median, 6),
        "no_cache_median_seconds": round(no_cache_median, 6),
        "speedup": round(no_cache_median / cache_median, 2),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()