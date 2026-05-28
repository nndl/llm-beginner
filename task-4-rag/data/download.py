"""下载任务四所需：BGE embedding + reranker + NNDL PDF，并检查 gold QA。"""
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent
REPO_ROOT = DATA_DIR.parent.parent
MODELS_DIR = DATA_DIR.parent / "models"
PDF_URL = "https://github.com/nndl/nndl/releases/download/book-pdf/nndl-v2.pdf"
GOLD_QA_MIN_COUNT = 30
GOLD_QA_REQUIRED_KEYS = {"id", "question", "answer", "source_file", "gold_anchors"}


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


def download_pdf():
    out = DATA_DIR / "kb.pdf"
    if out.exists() and out.stat().st_size > 0:
        print(f"NNDL v2 PDF 已存在：{out.name}")
        return
    print(f"下载 NNDL v2 PDF -> {out.name}")
    tmp = out.with_suffix(".pdf.tmp")
    urllib.request.urlretrieve(PDF_URL, tmp)
    tmp.replace(out)
    print("完成")


def normalize_text(text):
    return re.sub(r"\s+", "", str(text))


def validate_gold_qa():
    path = DATA_DIR / "gold_qa.jsonl"
    if not path.exists():
        sys.exit("[错误] 缺少 data/gold_qa.jsonl；请从仓库获取随任务提供的 gold QA。")

    items = []
    ids = set()
    source_checked = 0
    source_missing = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"[错误] {path.name}:{line_no} 不是合法 JSON：{e}")

            missing = sorted(k for k in GOLD_QA_REQUIRED_KEYS if not item.get(k))
            if missing:
                sys.exit(f"[错误] {path.name}:{line_no} 缺少字段：{', '.join(missing)}")
            if not isinstance(item["gold_anchors"], list) or not item["gold_anchors"]:
                sys.exit(f"[错误] {path.name}:{line_no} gold_anchors 必须是非空列表")
            if item["id"] in ids:
                sys.exit(f"[错误] {path.name}:{line_no} id 重复：{item['id']}")
            ids.add(item["id"])

            source_path = (REPO_ROOT / item["source_file"]).resolve()
            if source_path.exists():
                source_checked += 1
                source_text = normalize_text(source_path.read_text(encoding="utf-8"))
                missing_anchors = [
                    anchor for anchor in item["gold_anchors"]
                    if normalize_text(anchor) not in source_text
                ]
                if missing_anchors:
                    sys.exit(
                        f"[错误] {path.name}:{line_no} 的 gold_anchors 未命中 source_file："
                        f"{missing_anchors[:3]}"
                    )
            else:
                source_missing += 1
            items.append(item)

    if len(items) < GOLD_QA_MIN_COUNT:
        sys.exit(f"[错误] {path.name} 至少需要 {GOLD_QA_MIN_COUNT} 条题目，当前只有 {len(items)} 条")
    print(f"已检查评测题目：{path.name}（{len(items)} 条）")
    if source_checked:
        print(f"已校验 LaTeX 来源锚点：{source_checked} 条")
    if source_missing:
        print(f"[提示] {source_missing} 条 source_file 不在当前工作区，已跳过来源锚点校验。")
    validate_pdf_anchors(items)


def extract_pdf_text(pdf_path):
    try:
        from pypdf import PdfReader
    except ImportError:
        print("[提示] 未安装 pypdf，跳过 PDF anchor 校验；可先 pip install -r requirements.txt。")
        return None
    try:
        reader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"[提示] 无法抽取 {pdf_path.name} 文本，跳过 PDF anchor 校验：{e}")
        return None


def validate_pdf_anchors(items):
    pdf_path = DATA_DIR / "kb.pdf"
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        print("[提示] data/kb.pdf 不存在，跳过 PDF anchor 校验。")
        return
    pdf_text = extract_pdf_text(pdf_path)
    if not pdf_text:
        return
    normalized_pdf = normalize_text(pdf_text)
    misses = []
    for item in items:
        if not any(normalize_text(anchor) in normalized_pdf
                   for anchor in item["gold_anchors"]):
            misses.append(item["id"])
    if misses:
        sys.exit(
            "[错误] 以下 gold QA 的所有 gold_anchors 都未命中 PDF 抽取文本："
            + ", ".join(misses[:10])
        )
    print(f"已校验 PDF anchor 命中：{len(items)} 条题目均至少 1 个 anchor 可从 kb.pdf 抽取文本命中")


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
    print("\n--- 知识库 ---")
    print("默认知识库已下载到 data/kb.pdf：")
    print(f"  {PDF_URL}")
    print("请从 data/kb.pdf 抽取文本并建立索引，不要直接索引 LaTeX 源文件。")


def hint_eval_data():
    print("\n--- 评测数据 ---")
    print("gold QA 位于 data/gold_qa.jsonl。题目基于 ../神经网络与深度学习2/ 的 LaTeX 正文设计，")
    print("但自检会要求你的 retriever 从 data/kb.pdf 构建的索引中召回对应片段。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true",
                    help="只下载 PDF 并检查 gold QA，不下载 BGE 模型。")
    args = ap.parse_args()
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_models:
        download_bge()
    download_pdf()
    validate_gold_qa()
    hint_generator()
    hint_knowledge_base()
    hint_eval_data()


if __name__ == "__main__":
    main()
