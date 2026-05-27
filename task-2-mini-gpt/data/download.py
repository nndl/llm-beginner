"""下载任务二训练数据（三档可选）。

用法：
  python data/download.py --dataset poetry        # 唐诗（quick-start）
  python data/download.py --dataset tinystories   # TinyStories（~100MB）
  python data/download.py --dataset skypile       # SkyPile 子集（~1GB+）
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
ROOT = DATA_DIR.parent.parent  # llm-beginner 仓库根


def get_poetry():
    src = ROOT / "poetryFromTang.txt"
    if not src.exists():
        sys.exit(f"[错误] 找不到 {src}（应在仓库根）")
    dst = DATA_DIR / "poetry.txt"
    shutil.copy(src, dst)
    print(f"已拷贝 {src.name} -> {dst}")


def get_tinystories():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[错误] pip install datasets")
    print("下载 roneneldan/TinyStories ...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir=str(DATA_DIR / "cache"))
    for split in ds.keys():
        out = DATA_DIR / f"tinystories-{split}.parquet"
        ds[split].to_parquet(str(out))
        print(f"  {split}: {len(ds[split])} -> {out.name}")


def get_skypile():
    print("SkyPile-150B 体量大，建议手动选 streaming 子集：")
    print("  pip install datasets")
    print("  python -c \"from datasets import load_dataset; "
          "ds = load_dataset('Skywork/SkyPile-150B', split='train', streaming=True); "
          "import itertools, json; "
          "[print(json.dumps(x, ensure_ascii=False)) for x in itertools.islice(ds, 10000)]\" "
          "> data/skypile-sample.jsonl")
    print("\n或用 ModelScope：")
    print("  pip install modelscope")
    print("  modelscope download --dataset 'Skywork/SkyPile-150B' --local_dir ./data/cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["poetry", "tinystories", "skypile"],
                    default="tinystories")
    args = ap.parse_args()
    {"poetry": get_poetry, "tinystories": get_tinystories,
     "skypile": get_skypile}[args.dataset]()


if __name__ == "__main__":
    main()
