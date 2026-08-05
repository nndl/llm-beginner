"""Run the S1, S2, and S4 training experiments with isolated artifacts."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import load_for_eval


MODEL_SCALES = {
    "10m": {"d_model": 512, "n_heads": 8, "n_layers": 3, "d_ff": 2048},
    "50m": {"d_model": 768, "n_heads": 8, "n_layers": 7, "d_ff": 3072},
    "100m": {"d_model": 1024, "n_heads": 8, "n_layers": 8, "d_ff": 4096},
}


def train(output_dir: Path, config: dict, args: argparse.Namespace) -> None:
    command = [
        sys.executable, str(ROOT / "train.py"),
        "--output_dir", str(output_dir),
        "--vocab_size", str(args.vocab_size),
        "--block_size", str(args.block_size),
        "--batch_size", str(args.batch_size),
        "--max_iters", str(args.max_iters),
        "--eval_interval", str(args.eval_interval),
        "--eval_batches", str(args.eval_batches),
    ]
    for name, value in config.items():
        command.extend([f"--{name}", str(value)])
    subprocess.run(command, cwd=ROOT, check=True)


@torch.no_grad()
def perplexity_at_lengths(checkpoint: Path, lengths: list[int], n_windows: int) -> dict[str, float]:
    model, tokenizer = load_for_eval(str(checkpoint))
    text = (ROOT / "data" / "dev.txt").read_text(encoding="utf-8")
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    results = {}
    for length in lengths:
        losses = []
        for start in range(0, min(len(ids) - length - 1, length * n_windows), length):
            x = ids[start:start + length].unsqueeze(0)
            y = ids[start + 1:start + length + 1].unsqueeze(0)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
        if losses:
            results[str(length)] = round(math.exp(sum(losses) / len(losses)), 4)
    return results


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_s1(args: argparse.Namespace) -> None:
    results = []
    for name, config in MODEL_SCALES.items():
        output_dir = args.output_dir / "s1" / name
        train(output_dir, config, args)
        history = json.loads((output_dir / "train_history.json").read_text(encoding="utf-8"))
        results.append({
            "name": name,
            "checkpoint": str(output_dir / "best.pt"),
            "parameters": sum(p.numel() for p in load_for_eval(str(output_dir / "best.pt"))[0].parameters()),
            "best_dev_perplexity": round(math.exp(history["best_dev_loss"]), 4),
        })
    write_json(args.output_dir / "s1_parameter_scan.json", {"runs": results})


def run_s2(args: argparse.Namespace) -> None:
    results = {}
    cache_len = max(args.position_cache_len, max(args.lengths))
    for encoding in ("rope", "absolute"):
        output_dir = args.output_dir / "s2" / encoding
        train(output_dir, {
            "d_model": args.s2_d_model,
            "n_heads": args.s2_n_heads,
            "n_layers": args.s2_n_layers,
            "d_ff": args.s2_d_ff,
            "positional_encoding": encoding,
            "position_cache_len": cache_len,
        }, args)
        results[encoding] = perplexity_at_lengths(
            output_dir / "best.pt", args.lengths, args.eval_windows
        )
    write_json(args.output_dir / "s2_position_extrapolation.json", {
        "train_length": args.block_size,
        "evaluated_lengths": args.lengths,
        "perplexity": results,
    })


def run_s4(args: argparse.Namespace) -> None:
    output_dir = args.output_dir / "s4_tinystories_10m"
    train(output_dir, MODEL_SCALES["10m"], args)
    samples_path = args.output_dir / "s4_generation_samples.json"
    subprocess.run([
        sys.executable, str(ROOT / "generate.py"),
        "--checkpoint", str(output_dir / "best.pt"),
        "--prompt", "Once upon a time", "--max_new_tokens", "128",
        "--output", str(samples_path),
    ], cwd=ROOT, check=True)
    history = json.loads((output_dir / "train_history.json").read_text(encoding="utf-8"))
    write_json(args.output_dir / "s4_summary.json", {
        "parameters": sum(p.numel() for p in load_for_eval(str(output_dir / "best.pt"))[0].parameters()),
        "best_dev_perplexity": round(math.exp(history["best_dev_loss"]), 4),
        "checkpoint": str(output_dir / "best.pt"),
        "samples": str(samples_path),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["s1", "s2", "s4", "all"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "bonus")
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=8000)
    parser.add_argument("--eval-interval", type=int, default=400)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--position-cache-len", type=int, default=512)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--eval-windows", type=int, default=10)
    parser.add_argument("--s2-d-model", type=int, default=128)
    parser.add_argument("--s2-n-heads", type=int, default=4)
    parser.add_argument("--s2-n-layers", type=int, default=4)
    parser.add_argument("--s2-d-ff", type=int, default=512)
    args = parser.parse_args()

    if args.experiment in {"s1", "all"}:
        run_s1(args)
    if args.experiment in {"s2", "all"}:
        run_s2(args)
    if args.experiment in {"s4", "all"}:
        run_s4(args)


if __name__ == "__main__":
    main()