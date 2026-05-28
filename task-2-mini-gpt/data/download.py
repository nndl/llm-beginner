"""下载任务二训练数据，并生成 train.txt / dev.txt 自检输入。

用法：
  python data/download.py                         # 默认唐诗（quick-start）
  python data/download.py --dataset poetry        # 唐诗（quick-start）
  python data/download.py --dataset tinystories   # TinyStories（英文故事语料，约百 MB 量级）
  python data/download.py --dataset skypile       # SkyPile 子集提示（需手动抽样）
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
ROOT = DATA_DIR.parent.parent  # llm-beginner 仓库根


def write_dataset_info(dataset, ppl_threshold, extra=None):
    info = {
        "dataset": dataset,
        "train": "train.txt",
        "dev": "dev.txt",
        "ppl_threshold": ppl_threshold,
    }
    if extra:
        info.update(extra)
    (DATA_DIR / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text_splits(text, dataset, ppl_threshold, dev_ratio=0.1):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) < 100:
        sys.exit(f"[错误] {dataset} 文本过短，无法切分 train/dev")
    split_at = max(1, int(len(text) * (1 - dev_ratio)))
    (DATA_DIR / "train.txt").write_text(text[:split_at], encoding="utf-8")
    (DATA_DIR / "dev.txt").write_text(text[split_at:], encoding="utf-8")
    write_dataset_info(dataset, ppl_threshold)
    print(f"已生成 train.txt / dev.txt（{dataset}，dev_ratio={dev_ratio}）")


def get_poetry():
    src = ROOT / "poetryFromTang.txt"
    if not src.exists():
        sys.exit(f"[错误] 找不到 {src}（应在仓库根）")
    dst = DATA_DIR / "poetry.txt"
    shutil.copy(src, dst)
    write_text_splits(src.read_text(encoding="utf-8"),
                      dataset="poetry", ppl_threshold=50)
    print(f"已拷贝 {src.name} -> {dst.name}")


def write_hf_text_split(split, out_path):
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in split:
            text = str(row.get("text", "")).strip()
            if text:
                f.write(text)
                f.write("\n\n")


def get_tinystories():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[错误] pip install datasets pyarrow")
    print("下载 roneneldan/TinyStories ...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir=str(DATA_DIR / "cache"))
    for split in ds.keys():
        out = DATA_DIR / f"tinystories-{split}.parquet"
        ds[split].to_parquet(str(out))
        print(f"  {split}: {len(ds[split])} -> {out.name}")

    train_split = ds["train"]
    dev_split = ds["validation"] if "validation" in ds else ds["train"].select(range(1000))
    write_hf_text_split(train_split, DATA_DIR / "train.txt")
    write_hf_text_split(dev_split, DATA_DIR / "dev.txt")
    write_dataset_info("tinystories", ppl_threshold=10,
                       extra={"note": "TinyStories 是英文故事语料；中文训练可换用 SkyPile 子集或其他中文语料。"})
    print("已额外生成 train.txt / dev.txt，供 eval/run.py 计算困惑度。")


def get_skypile():
    print("SkyPile-150B 体量大，建议手动选 streaming 子集并显式切出 train/dev：")
    print("  pip install datasets")
    print("  python -c \"from datasets import load_dataset; "
          "ds = load_dataset('Skywork/SkyPile-150B', split='train', streaming=True); "
          "import itertools; "
          "f=open('data/train.txt','w',encoding='utf-8'); "
          "[f.write(str(x.get('text',''))+'\\n\\n') for x in itertools.islice(ds, 10000)]; "
          "f.close()\"")
    print("  # 再从 train.txt 尾部切出 data/dev.txt，并写入 data/dataset_info.json")
    print("\n或用 ModelScope：")
    print("  pip install modelscope")
    print("  modelscope download --dataset 'Skywork/SkyPile-150B' --local_dir ./data/cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["poetry", "tinystories", "skypile"],
                    default="poetry")
    args = ap.parse_args()
    {"poetry": get_poetry, "tinystories": get_tinystories,
     "skypile": get_skypile}[args.dataset]()


if __name__ == "__main__":
    main()
