"""下载任务三所需资源：Qwen2.5-0.5B 基座模型 + MOSS SFT 数据 + MOSS plugin 数据。

DPO 偏好数据需自行选择（提示见末尾）。
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
MODELS_DIR = DATA_DIR.parent / "models"


def download_base_model():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("[错误] pip install huggingface_hub")
    print("下载 Qwen/Qwen2.5-0.5B 到 models/Qwen2.5-0.5B ...")
    snapshot_download("Qwen/Qwen2.5-0.5B",
                      local_dir=str(MODELS_DIR / "Qwen2.5-0.5B"))
    print("完成")


def hint_sft_data():
    print("\n--- MOSS-003-sft-data（SFT 数据）---")
    print("HF：")
    print("  pip install datasets")
    print("  python -c \"from datasets import load_dataset; "
          "ds = load_dataset('fnlp/moss-003-sft-data'); "
          "ds.save_to_disk('./data/moss-sft')\"")
    print("GitHub：https://github.com/OpenLMLab/MOSS  (数据集 release)")


def hint_plugin_data():
    print("\n--- MOSS-003-sft-plugin（带工具调用对话，给任务五贯通用）---")
    print("HF 路径：fnlp/moss-003-sft-plugin  （加载方式同上）")


def hint_dpo_data():
    print("\n--- DPO 偏好数据（自选）---")
    print("候选：")
    print("  - hiyouga/DPO-En-Zh-20k       中英混合")
    print("  - argilla/distilabel-intel-orca-dpo-pairs  英文")
    print("或自行用 GPT-4 / Claude 给已有 SFT 数据打偏好标签")


def main():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_base_model()
    hint_sft_data()
    hint_plugin_data()
    hint_dpo_data()


if __name__ == "__main__":
    main()
