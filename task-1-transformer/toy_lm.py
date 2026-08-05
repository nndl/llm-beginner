"""M4：把同一份手写 attention 加上 causal mask，跑一个 toy 语言模型（next-token prediction）。

目的：验证 —— 加了 causal mask 之后，未来位置的 V 改动不会影响过去位置的输出/预测，
即模型在预测第 i 个词时看不到第 i+1...T 个词。这一步是任务二（GPT）的预热。

用法：
    python toy_lm.py
"""
import torch
import torch.nn as nn

from src.block import TransformerBlock

# 一段很短的 toy 语料（字符级），够跑通训练流程即可，不追求效果
CORPUS = (
    "床前明月光疑是地上霜举头望明月低头思故乡"
    "白日依山尽黄河入海流欲穷千里目更上一层楼"
)


def build_causal_mask(T: int) -> torch.Tensor:
    """上三角为 True（被屏蔽），下三角含对角线为 False（可见）。"""
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout=0.0) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.embedding(ids) + self.pos_embedding(pos)

        mask = build_causal_mask(T).to(ids.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T) 广播到 (B,H,T,T)

        for block in self.blocks:
            x = block(x, mask=mask)
        return self.lm_head(x)


def main():
    chars = sorted(set(CORPUS))
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for c, i in char2id.items()}
    ids = torch.tensor([char2id[c] for c in CORPUS], dtype=torch.long)

    T = 32
    model = ToyCausalLM(vocab_size=len(chars), max_len=T)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    x = ids[:T].unsqueeze(0)
    y = ids[1 : T + 1].unsqueeze(0)

    print("训练 toy 因果语言模型 ...")
    model.train()
    for step in range(200):
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}: loss={loss.item():.4f}")

    # ---- 验证 causal mask：未来位置的输入改动不应影响过去位置的输出 ----
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % len(chars)  # 改动最后一个 token
        out2 = model(x2)

        leaked = (out1[:, :-1] - out2[:, :-1]).abs().max().item()
        print(f"\n改动最后一个 token 后，前面位置输出的最大差异 = {leaked:.8f}")
        print("causal mask 生效（未来词元不泄漏）" if leaked < 1e-6 else "警告：causal mask 可能有问题！")

        pred_ids = out1[0].argmax(dim=-1).tolist()
        pred_text = "".join(id2char[i] for i in pred_ids)
        print(f"\n输入: {''.join(id2char[i.item()] for i in x[0])}")
        print(f"预测的下一个字符序列: {pred_text}")


if __name__ == "__main__":
    main()
