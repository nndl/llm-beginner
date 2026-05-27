"""下载任务四所需：BGE embedding + reranker；其他资源（生成模型/知识库/评测）给指引。"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
MODELS_DIR = DATA_DIR.parent / "models"


def download_bge():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("[错误] pip install huggingface_hub")
    print("下载 bge-small-zh-v1.5 ...")
    snapshot_download("BAAI/bge-small-zh-v1.5",
                      local_dir=str(MODELS_DIR / "bge-small-zh-v1.5"))
    print("下载 bge-reranker-base ...")
    snapshot_download("BAAI/bge-reranker-base",
                      local_dir=str(MODELS_DIR / "bge-reranker-base"))
    print("完成")


def hint_generator():
    print("\n--- 生成模型 Qwen2.5-7B-Instruct（自行下载，量化版省显存）---")
    print("Ollama 一键：")
    print("  ollama pull qwen2.5:7b-instruct")
    print("  ollama serve  # 启动 OpenAI 兼容 API 在 http://localhost:11434/v1")
    print("vLLM：")
    print("  pip install vllm")
    print("  python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct")
    print("llama.cpp + GGUF：")
    print("  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf "
          "--local-dir ./models")


def hint_knowledge_base():
    print("\n--- 知识库（自选）---")
    print("推荐：《神经网络与深度学习》PDF（与读者背景对齐）")
    print("  https://nndl.github.io/")
    print("放在 data/kb.pdf 即可（脚本里硬编码路径）")


def hint_eval_data():
    print("\n--- 评测数据 CMRC 2018 ---")
    print("  from datasets import load_dataset")
    print("  ds = load_dataset('cmrc2018')")
    print("  ds['validation'].to_parquet('./data/cmrc-dev.parquet')")


def main():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_bge()
    hint_generator()
    hint_knowledge_base()
    hint_eval_data()


if __name__ == "__main__":
    main()
