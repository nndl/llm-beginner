"""运行任务一 S1、S2、S4 加分实验，并汇总真实训练结果。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


EXPERIMENTS = [
    {
        "name": "s1_heads_2_layers_2",
        "bonus": "S1",
        "description": "2 heads, 2 layers",
        "args": ["--n_heads", "2", "--n_layers", "2", "--d_ff", "256"],
    },
    {
        "name": "s1_heads_4_layers_4",
        "bonus": "S1",
        "description": "4 heads, 4 layers (baseline)",
        "args": ["--n_heads", "4", "--n_layers", "4", "--d_ff", "512"],
    },
    {
        "name": "s1_heads_8_layers_4",
        "bonus": "S1",
        "description": "8 heads, 4 layers",
        "args": ["--n_heads", "8", "--n_layers", "4", "--d_ff", "512"],
    },
    {
        "name": "s2_without_residual",
        "bonus": "S2",
        "description": "remove residual connections",
        "args": ["--no_residual"],
    },
    {
        "name": "s2_without_layernorm",
        "bonus": "S2",
        "description": "remove LayerNorm",
        "args": ["--no_layernorm"],
    },
    {
        "name": "s3_wider_model",
        "bonus": "S3",
        "description": "wider 192-dimensional, 6-head, 4-layer model",
        "args": [
            "--d_model", "192", "--n_heads", "6", "--n_layers", "4",
            "--d_ff", "768", "--epochs", "12",
        ],
    },
    {
        "name": "s4_rope",
        "bonus": "S4",
        "description": "replace absolute PE with RoPE",
        "script": "train_rope.py",
        "args": [],
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--only", nargs="*", help="Experiment names to run")
    args = parser.parse_args()

    results_dir = ROOT / "bonus"
    results_dir.mkdir(exist_ok=True)
    selected = [item for item in EXPERIMENTS if not args.only or item["name"] in args.only]
    results = []

    for experiment in selected:
        output_dir = results_dir / experiment["name"]
        command = [
            sys.executable, str(ROOT / experiment.get("script", "train.py")), "--epochs", str(args.epochs),
            "--output_dir", str(output_dir), *experiment["args"],
        ]
        print(f"\n=== {experiment['name']} ===")
        subprocess.run(command, check=True, cwd=ROOT)
        history_path = output_dir / "train_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        results.append({
            **{key: experiment[key] for key in ("name", "bonus", "description")},
            "best_dev_acc": history["best_dev_acc"],
            "epochs_run": len(history["history"]),
            "final_train_loss": history["history"][-1]["train_loss"],
        })

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()