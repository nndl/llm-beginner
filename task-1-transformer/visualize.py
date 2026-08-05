"""从训练好的 checkpoint 中取一层/一个 head 的注意力权重，画热图存到 figures/。

用法：
    python visualize.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import torch

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"

# 1 正面样本、1 负面样本、1 长句样本
SAMPLES = [
    ("positive", "这家酒店的服务真的很好，房间干净整洁，下次还会再来住。"),
    ("negative", "东西质量很差，客服态度也差，非常失望，不推荐购买。"),
    ("long", "这次出差住的这家酒店，地理位置还算不错，靠近地铁站，"
             "但是隔音效果很差，晚上一直能听到走廊的说话声，早餐种类也比较单一，"
             "整体体验只能说一般，如果不是价格便宜我是不会再来的。"),
]

LAYER_IDX = 0  # 可视化第几层（0-indexed）
HEAD_IDX = 0   # 可视化第几个 head（0-indexed）


def plot_attention(tokens, attn, title, out_path):
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.4), max(6, len(tokens) * 0.4)))
    im = ax.imshow(attn, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ckpt_path = ROOT / "ckpt" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} 不存在，请先运行 train.py 训练模型")

    model, tokenize_fn = load_for_eval(str(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    id2char = {v: k for k, v in ckpt["char2id"].items()}

    FIG_DIR.mkdir(exist_ok=True)

    with torch.no_grad():
        for tag, text in SAMPLES:
            ids = tokenize_fn(text).unsqueeze(0)  # (1, T)
            logits = model(ids)
            pred = int(logits.argmax(dim=-1).item())

            attn = model.blocks[LAYER_IDX].attn.last_attn  # (1, H, T, T)
            attn_map = attn[0, HEAD_IDX].numpy()

            # 只保留非 PAD 部分，图更清晰
            valid_len = int((ids[0] != 0).sum().item())
            attn_map = attn_map[:valid_len, :valid_len]
            tokens = [id2char.get(int(i), "?") for i in ids[0, :valid_len]]

            title = f"{tag} (pred={pred})  layer{LAYER_IDX} head{HEAD_IDX}"
            out_path = FIG_DIR / f"attn_{tag}.png"
            plot_attention(tokens, attn_map, title, out_path)


if __name__ == "__main__":
    main()
