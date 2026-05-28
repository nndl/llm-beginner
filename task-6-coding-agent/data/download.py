"""准备任务六的本地 toy repo；可选下载 SWE-bench Lite 抽样元数据。"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
TOY_REPO = DATA_DIR / "toy-repo"


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def create_toy_repo():
    TOY_REPO.mkdir(parents=True, exist_ok=True)
    write_text(TOY_REPO / "calculator.py",
               "def add(a, b):\n"
               "    \"\"\"Return the sum of two numbers.\"\"\"\n"
               "    return a - b\n\n"
               "def factorial(n):\n"
               "    if n < 0:\n"
               "        raise ValueError('n must be non-negative')\n"
               "    result = 1\n"
               "    for value in range(2, n + 1):\n"
               "        result *= value\n"
               "    return result\n")
    write_text(TOY_REPO / "test_calculator.py",
               "from calculator import add, factorial\n\n"
               "def test_add_positive_numbers():\n"
               "    assert add(2, 3) == 5\n\n"
               "def test_add_negative_number():\n"
               "    assert add(-2, 3) == 1\n\n"
               "def test_factorial():\n"
               "    assert factorial(5) == 120\n")
    write_text(TOY_REPO / "ISSUE.md",
               "修复 `calculator.add` 的实现，使 `python -m pytest` 全部通过。"
               "不要改测试文件。\n")
    if not (TOY_REPO / ".git").exists():
        try:
            subprocess.run(["git", "init"], cwd=TOY_REPO, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "add", "."], cwd=TOY_REPO, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "commit", "-m", "initial toy repo"], cwd=TOY_REPO,
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    print(f"已生成本地 toy repo：{TOY_REPO.relative_to(DATA_DIR.parent)}")


def download_swebench_lite_sample():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[错误] pip install datasets pyarrow")
    print("下载 princeton-nlp/SWE-bench_Lite 抽样元数据 ...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test",
                      cache_dir=str(DATA_DIR / "cache"))
    sample = ds.select(range(3))
    out = DATA_DIR / "swebench-lite-sample.parquet"
    sample.to_parquet(str(out))
    print(f"已保存 3 题元数据到 {out.relative_to(DATA_DIR.parent)}")
    for row in sample:
        repo_name = row["repo"].split("/")[-1]
        print(f"  - {row['instance_id']} (repo: {row['repo']})")
        print(f"    可选 clone: git clone https://github.com/{row['repo']}.git data/repos/{repo_name}")


def hint_model():
    print("\n--- 模型部署 Qwen2.5-Coder-7B-Instruct ---")
    print("Ollama：")
    print("  ollama pull qwen2.5-coder:7b-instruct")
    print("  ollama serve")
    print("vLLM：")
    print("  python -m vllm.entrypoints.openai.api_server "
          "--model Qwen/Qwen2.5-Coder-7B-Instruct --quantization awq")
    print("llama.cpp + GGUF：")
    print("  huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF "
          "qwen2.5-coder-7b-instruct-q4_k_m.gguf --local-dir ./models")


def hint_swebench_harness():
    print("\n--- SWE-bench 评测 harness（可选进阶）---")
    print("  python data/download.py --with-swebench")
    print("  pip install swebench")
    print("  python -m swebench.harness.run_evaluation --help")
    print("  官方文档：https://github.com/princeton-nlp/SWE-bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-swebench", action="store_true",
                    help="额外下载 SWE-bench Lite 3 题元数据；本地 repo 需另行 clone。")
    args = ap.parse_args()
    create_toy_repo()
    if args.with_swebench:
        download_swebench_lite_sample()
    hint_model()
    hint_swebench_harness()


if __name__ == "__main__":
    main()
