"""下载任务六的评测数据 SWE-bench Lite 抽样集；并打印模型部署提示。"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent


def download_swebench_lite_sample():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[错误] pip install datasets")
    print("下载 princeton-nlp/SWE-bench_Lite 抽样 ...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test",
                      cache_dir=str(DATA_DIR / "cache"))
    # 只取前 3 题作为入门 sanity（完整 300 题学生可自行扩展）
    sample = ds.select(range(3))
    out = DATA_DIR / "swebench-lite-sample.parquet"
    sample.to_parquet(str(out))
    print(f"已保存 3 题到 {out.relative_to(DATA_DIR.parent)}")
    for row in sample:
        print(f"  - {row['instance_id']} (repo: {row['repo']})")


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
    print("\n--- SWE-bench 评测 harness（学生可选用官方评测）---")
    print("  pip install swebench")
    print("  python -m swebench.harness.run_evaluation --help")
    print("  官方文档：https://github.com/princeton-nlp/SWE-bench")


def main():
    download_swebench_lite_sample()
    hint_model()
    hint_swebench_harness()


if __name__ == "__main__":
    main()
