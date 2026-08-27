"""S1 参数量扫描重跑：三档模型各自训练到 early stop，输出统一汇总。

设计要点：
- 三档配置与 run_bonus_experiments.MODEL_SCALES 一致，保证可比。
- max_iters 给足，由 patience=6 的 early stop 决定实际停止步数；
  这比固定 3000 步更接近"各自收敛点"，但仍不等同于严格平台期。
- 输出写入 bonus/s1/{10m,50m,100m}/，并更新 bonus/s1_parameter_scan.json。
- 数据集沿用 data/train.txt + data/dev.txt（TinyStories），与原 S1 一致。
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.model import load_for_eval  # noqa: E402

MODEL_SCALES = {
    "10m": {"d_model": 512, "n_heads": 8, "n_layers": 3, "d_ff": 2048, "max_iters": 8000},
    "50m": {"d_model": 768, "n_heads": 8, "n_layers": 7, "d_ff": 3072, "max_iters": 8000},
    "100m": {"d_model": 1024, "n_heads": 8, "n_layers": 8, "d_ff": 4096, "max_iters": 6000},
}


def train_one(name: str, cfg: dict, args: argparse.Namespace, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_iters = cfg.pop("max_iters")
    command = [
        sys.executable, str(ROOT / "train.py"),
        "--output_dir", str(output_dir),
        "--data_dir", str(args.data_dir),
        "--vocab_size", str(args.vocab_size),
        "--block_size", str(args.block_size),
        "--batch_size", str(args.batch_size),
        "--max_iters", str(max_iters),
        "--eval_interval", str(args.eval_interval),
        "--eval_batches", str(args.eval_batches),
        "--patience", str(args.patience),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
    ]
    for k, v in cfg.items():
        command.extend([f"--{k}", str(v)])
    t0 = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    dt = time.time() - t0
    history = json.loads((output_dir / "train_history.json").read_text(encoding="utf-8"))
    model, _ = load_for_eval(str(output_dir / "best.pt"))
    n_params = sum(p.numel() for p in model.parameters())
    best_dev_loss = history["best_dev_loss"]
    last = history["history"][-1]
    return {
        "name": name,
        "parameters": n_params,
        "max_iters": max_iters,
        "actual_steps": last["step"],
        "best_dev_perplexity": round(math.exp(best_dev_loss), 4),
        "final_dev_perplexity": round(last["dev_ppl"], 4),
        "wall_time_seconds": round(dt, 1),
        "checkpoint": str(output_dir / "best.pt"),
        "converged": last["step"] < max_iters,  # early stop 触发视为接近平台
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "bonus" / "s1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", choices=["10m", "50m", "100m"], help="只跑某一档")
    args = parser.parse_args()

    runs = []
    for name, cfg in MODEL_SCALES.items():
        if args.only and name != args.only:
            continue
        cfg = dict(cfg)  # 拷贝，避免 train_one 的 pop 污染原表
        print(f"\n=== S1 重跑：{name} ===")
        result = train_one(name, cfg, args, args.output_dir / name)
        runs.append(result)
        print(f"  参数量={result['parameters']:,}  实际步数={result['actual_steps']}/"
              f"{result['max_iters']}  best PPL={result['best_dev_perplexity']}  "
              f"early_stop={result['converged']}  耗时={result['wall_time_seconds']}s")

    summary_path = args.output_dir.parent / "s1_parameter_scan.json"
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    by_name = {r["name"]: r for r in (existing.get("runs") or [])}
    for r in runs:
        by_name[r["name"]] = r
    summary = {
        "dataset": "tinystories",
        "vocab_size": args.vocab_size,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "runs": [by_name[k] for k in MODEL_SCALES if k in by_name],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n汇总已写入 {summary_path}")


if __name__ == "__main__":
    main()
