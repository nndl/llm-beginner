"""训练 MiniGPT：字节级 BPE tokenizer + RoPE + KV cache 的小型 decoder-only 语言模型。

用法：
    python data/download.py --dataset poetry   # 先准备 data/train.txt / data/dev.txt
    python train.py
    python train.py --d_model 64 --n_layers 2  # 卡住时先缩小模型跑通流程
"""
import argparse  # 读取命令行超参数
import json  # 保存可复查的训练历史
import math  # 学习率余弦与 perplexity 的指数计算
import time  # 记录训练耗时
from pathlib import Path  # 构造数据、checkpoint 的稳定路径

import torch
import torch.nn.functional as F

from src.model import MiniGPT
from src.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent  # task-2 目录，不依赖执行时 cwd
DATA_DIR = ROOT / "data"  # 约定的数据文件夹
CKPT_DIR = ROOT / "ckpt"  # 约定的 tokenizer/checkpoint 输出目录


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:  # warmup 阶段线性增大学习率，减少随机初始化时的不稳定更新
        return base_lr * (step + 1) / max(1, warmup_steps)  # max 防止极端配置出现除零
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)  # 防止超过总步数时余弦函数继续周期性回升
    cos_val = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos_val)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device):
    """随机截取 batch_size 段长度为 block_size+1 的连续 token，前 block_size 个是输入，
    后 block_size 个（错开一位）是预测目标——标准的 next-token-prediction 构造方式。"""
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))  # 为 batch 中每个样本独立随机选择窗口起点
    x = torch.stack([data[i:i + block_size] for i in ix])  # 当前 token 序列作为模型输入
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])  # 右移一位，作为每个位置的预测标签
    return x.to(device), y.to(device)  # 将 CPU token 张量复制/移动到模型设备


@torch.no_grad()
def estimate_loss(model, data: torch.Tensor, block_size: int, batch_size: int,
                   device, n_batches: int = 20):
    model.eval()  # 临时关闭 dropout，使验证指标不受随机失活影响
    losses = []  # 收集多批随机验证损失以降低抽样方差
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()  # 恢复训练模式，供调用者继续反向传播
    return sum(losses) / len(losses)  # 返回 n_batches 个 token 平均交叉熵的均值


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--positional_encoding", choices=["rope", "absolute"], default="rope",
                        help="位置编码实验：RoPE 或可学习绝对位置 embedding")
    parser.add_argument("--position_cache_len", type=int, default=None,
                        help="位置表长度；S2 可设置为大于 block_size 的值测试长序列外推")
    parser.add_argument("--vocab_size", type=int, default=800,
                        help="BPE 目标词表大小（含 256 个基础字节）")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW 权重衰减系数，用于抑制小语料训练过拟合")
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6,
                        help="dev loss 连续多少次 eval 不提升就 early stop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR,
                        help="包含 train.txt 与 dev.txt 的数据目录；用于隔离不同实验语料")
    parser.add_argument("--output_dir", type=Path, default=CKPT_DIR,
                        help="tokenizer.json 与 best.pt 的输出目录；实验时可指定独立目录以保留正式产物")
    args = parser.parse_args()  # 将命令行字符串转换为带类型的配置对象

    torch.manual_seed(args.seed)  # 固定 PyTorch 随机数，使同配置尽可能可复现
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 CUDA 时优先使用 GPU
    print(f"device = {device}")

    train_text = (args.data_dir / "train.txt").read_text(encoding="utf-8")  # 读入训练部分的原始 Unicode 文本
    dev_text = (args.data_dir / "dev.txt").read_text(encoding="utf-8")  # 读入留出的验证文本，绝不参与梯度更新

    # ---- M1: 训练 BPE tokenizer ----
    print(f"训练 BPE tokenizer，目标词表大小 = {args.vocab_size} ...")
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, vocab_size=args.vocab_size, verbose=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)  # 首次训练时创建输出目录；实验目录可嵌套创建
    tokenizer.save(str(args.output_dir / "tokenizer.json"))
    print(f"tokenizer 训练完成，实际词表大小 = {tokenizer.vocab_size}，"
            f"已保存到 {args.output_dir / 'tokenizer.json'}")

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    dev_ids = torch.tensor(tokenizer.encode(dev_text), dtype=torch.long)
    print(f"train tokens = {len(train_ids)}, dev tokens = {len(dev_ids)}")

    # ---- M2+M3: 构建模型 ----
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.block_size,
        dropout=args.dropout,
        rope_cache_len=args.position_cache_len,
        positional_encoding=args.positional_encoding,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 = {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)  # AdamW 将解耦权重衰减应用于模型参数
    warmup_steps = max(1, int(0.05 * args.max_iters))

    best_dev_loss = float("inf")
    no_improve = 0
    history = []
    t0 = time.time()

    for step in range(1, args.max_iters + 1):
        lr = cosine_lr(step, args.max_iters, warmup_steps, args.lr)
        for g in optimizer.param_groups:  # 支持未来添加多个参数组时同时更新它们
            g["lr"] = lr  # 将当前 warmup/cosine 阶段学习率写入优化器

        x, y = get_batch(train_ids, args.block_size, args.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()  # 清空上个 step 累积的梯度
        loss.backward()  # 由交叉熵反向传播到所有可训练参数
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 限制梯度范数，避免偶发更新尖峰
        optimizer.step()  # 用 AdamW 根据本 batch 梯度更新参数

        if step % args.eval_interval == 0 or step == args.max_iters:
            dev_loss = estimate_loss(model, dev_ids, args.block_size,
                                      args.batch_size, device, args.eval_batches)
            dev_ppl = math.exp(min(dev_loss, 20))
            dt = time.time() - t0
            print(f"step {step}/{args.max_iters}  train_loss={loss.item():.4f}  "
                  f"dev_loss={dev_loss:.4f}  dev_ppl={dev_ppl:.2f}  lr={lr:.2e}  ({dt:.1f}s)")
            history.append({"step": step, "train_loss": loss.item(),
                            "dev_loss": dev_loss, "dev_ppl": dev_ppl, "lr": lr})

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                no_improve = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": {
                            "vocab_size": tokenizer.vocab_size,
                            "d_model": args.d_model,
                            "n_heads": args.n_heads,
                            "n_layers": args.n_layers,
                            "d_ff": args.d_ff,
                            "max_seq_len": args.block_size,
                            "rope_base": 10000.0,
                            "rope_cache_len": args.position_cache_len or max(args.block_size * 4, 1024),
                            "positional_encoding": args.positional_encoding,
                        },
                    },
                    args.output_dir / "best.pt",
                )
                print(f"  -> 保存新的最佳 checkpoint (dev_loss={dev_loss:.4f}, dev_ppl={dev_ppl:.2f})")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"dev_loss 连续 {args.patience} 次 eval 未提升，early stop")
                    break

    print(f"\n训练结束，最佳 dev_loss = {best_dev_loss:.4f}（ppl={math.exp(min(best_dev_loss, 20)):.2f}），"
            f"checkpoint 已保存到 {args.output_dir / 'best.pt'}")
    with open(args.output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"best_dev_loss": best_dev_loss, "history": history}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
