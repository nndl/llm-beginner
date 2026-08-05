"""训练 TransformerClassifier 在 ChnSentiCorp 上做情感二分类。

用法：
    python train.py
    python train.py --d_model 64 --n_layers 2   # M3 卡住时缩小模型先跑通
"""
import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model import TransformerClassifier
from src.vocab import build_vocab, encode, PAD_ID

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "ckpt"


class SentiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, char2id: dict, max_len: int):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.char2id = char2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.char2id, self.max_len)
        return torch.tensor(ids, dtype=torch.long), int(self.labels[idx])


def collate(batch):
    ids = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return ids, labels


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_vocab_size", type=int, default=6000)
    parser.add_argument("--patience", type=int, default=3, help="dev acc 连续多少个 epoch 不提升就 early stop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position_encoding", choices=["absolute", "rope"], default="absolute")
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--no_layernorm", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=ROOT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    dev_df = pd.read_parquet(DATA_DIR / "validation.parquet")
    print(f"train: {len(train_df)}, dev: {len(dev_df)}")

    char2id = build_vocab(train_df["text"].tolist(), max_vocab_size=args.max_vocab_size)
    print(f"vocab size = {len(char2id)}")

    train_ds = SentiDataset(train_df, char2id, args.max_len)
    dev_ds = SentiDataset(dev_df, char2id, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate)

    model = TransformerClassifier(
        vocab_size=len(char2id),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        num_classes=2,
        max_len=args.max_len,
        dropout=args.dropout,
        pad_id=PAD_ID,
        use_residual=not args.no_residual,
        use_layernorm=not args.no_layernorm,
        position_encoding=args.position_encoding,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 = {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(0.06 * total_steps))
    step = 0

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    best_acc = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)

            lr = cosine_lr(step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            logits = model(ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * ids.size(0)
            step += 1

        train_loss = total_loss / len(train_ds)
        dev_acc = evaluate(model, dev_loader, device)
        dt = time.time() - t0
        print(f"epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  dev_acc={dev_acc:.4f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_acc": dev_acc})

        if dev_acc > best_acc:
            best_acc = dev_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "char2id": char2id,
                    "config": {
                        "vocab_size": len(char2id),
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "d_ff": args.d_ff,
                        "num_classes": 2,
                        "max_len": args.max_len,
                        "pad_id": PAD_ID,
                        "use_residual": not args.no_residual,
                        "use_layernorm": not args.no_layernorm,
                        "position_encoding": args.position_encoding,
                    },
                },
                ckpt_dir / "best.pt",
            )
            print(f"  -> 保存新的最佳 checkpoint (dev_acc={dev_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"dev_acc 连续 {args.patience} 个 epoch 未提升，early stop")
                break

    print(f"\n训练结束，最佳 dev_acc = {best_acc:.4f}，checkpoint 已保存到 {ckpt_dir / 'best.pt'}")
    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"best_dev_acc": best_acc, "history": history}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
