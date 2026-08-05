"""手写缩放点积注意力 + 多头注意力（不使用 nn.MultiheadAttention 等高层封装）。"""
import math  # 用来算 sqrt(d_k) 缩放因子

import torch
import torch.nn as nn


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def scaled_dot_product_attention(Q, K, V, mask=None, attn_weights_out=None):
    """缩放点积注意力。

    Args:
        Q, K, V: 形状 (B, H, T, D)
        mask: 形状可广播到 (B, H, T_q, T_k)，`True` 表示该位置被屏蔽（不参与 attention）
        attn_weights_out: 可选的 list，如果传入会把 softmax 后的权重 append 进去
            （仅用于可视化调试，不影响返回值，也不影响自检的接口约定）

    Returns:
        输出张量，形状 (B, H, T_q, D)
    """
    d_k = Q.size(-1)  # 取每个 head 的维度，即 Q 最后一维的大小
    # 注意：缩放因子必须是 sqrt(d_k)（每个 head 的维度），而不是 sqrt(d_model)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Q @ K^T：把 K 的最后两维转置后与 Q 相乘，得到 (B, H, T_q, T_k) 的注意力打分矩阵；
    # 除以 sqrt(d_k) 是为了防止点积数值过大导致 softmax 梯度消失

    if mask is not None:
        # 用 -inf 填充而不是乘 0：乘 0 在 softmax 之后仍会留下非零概率（泄漏）
        scores = scores.masked_fill(mask, float("-inf"))
        # mask 为 True 的位置全部设为 -inf，softmax 后这些位置的权重会趋近于 0

    weights = torch.softmax(scores, dim=-1)
    # 在最后一维（T_k，即 key 的时间步）上做 softmax，得到每个 query 对所有 key 的归一化权重

    if attn_weights_out is not None:
        attn_weights_out.append(weights.detach())
        # detach 避免把权重计入计算图（只是拿去可视化，不需要梯度）

    output = torch.matmul(weights, V)  # 用权重对 V 加权求和，得到每个 query 位置的输出
    return output


class MultiHeadAttention(nn.Module):
    """多头注意力：拼分 head + Q/K/V 投影 + 输出投影。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 use_rope: bool = False, max_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        # 每个 head 的维度必须是整数，否则无法均匀切分
        self.d_model = d_model  # 总的隐藏维度
        self.n_heads = n_heads  # 头数
        self.d_k = d_model // n_heads  # 每个 head 分到的维度
        self.use_rope = use_rope

        if use_rope:
            if self.d_k % 2 != 0:
                raise ValueError("RoPE requires an even per-head dimension")
            positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
            frequencies = torch.exp(
                torch.arange(0, self.d_k, 2, dtype=torch.float)
                * (-math.log(10000.0) / self.d_k)
            )
            angles = positions * frequencies
            cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
            sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
            self.register_buffer("rope_cos", cos.unsqueeze(0).unsqueeze(0))
            self.register_buffer("rope_sin", sin.unsqueeze(0).unsqueeze(0))

        self.q_proj = nn.Linear(d_model, d_model)  # 生成 Query 的线性投影
        self.k_proj = nn.Linear(d_model, d_model)  # 生成 Key 的线性投影
        self.v_proj = nn.Linear(d_model, d_model)  # 生成 Value 的线性投影
        self.out_proj = nn.Linear(d_model, d_model)  # 多头拼接后的输出投影
        self.dropout = nn.Dropout(dropout)  # 输出前的 dropout，正则化用

        # 最近一次 forward 的注意力权重，形状 (B, H, T, T)，仅用于可视化
        self.last_attn = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, H, T, d_k)
        B, T, _ = x.shape  # 取出 batch 大小和序列长度，最后一维（d_model）不需要单独用
        x = x.view(B, T, self.n_heads, self.d_k)  # 把 d_model 拆成 (n_heads, d_k) 两维
        return x.transpose(1, 2).contiguous()
        # 交换 T 和 H 的位置，变成 (B, H, T, d_k)，方便后续按 head 做矩阵乘法；
        # contiguous() 保证转置后内存连续，避免后续 view 报错

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, T, d_k) -> (B, T, d_model)
        B, H, T, D = x.shape  # 拆出各维度大小
        x = x.transpose(1, 2).contiguous()  # 把 H 和 T 换回来，变成 (B, T, H, d_k)
        return x.view(B, T, H * D)  # 把最后两维 (H, d_k) 合并回 d_model，恢复原始形状

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: 可广播到 (B, H, T, T) 的 bool 张量，True = 屏蔽
        """
        Q = self._split_heads(self.q_proj(x))  # 线性投影后切分成多头，得到 (B, H, T, d_k)
        K = self._split_heads(self.k_proj(x))  # 同上，K 的多头形式
        V = self._split_heads(self.v_proj(x))  # 同上，V 的多头形式

        if self.use_rope:
            T = x.size(1)
            Q = apply_rope(Q, self.rope_cos[:, :, :T], self.rope_sin[:, :, :T])
            K = apply_rope(K, self.rope_cos[:, :, :T], self.rope_sin[:, :, :T])

        weights_container = []  # 用于接收 attention 权重（供可视化）的容器
        context = scaled_dot_product_attention(
            Q, K, V, mask=mask, attn_weights_out=weights_container
        )  # 计算多头注意力输出，形状 (B, H, T, d_k)
        self.last_attn = weights_container[0] if weights_container else None
        # 保存本次 forward 的注意力权重，方便外部（如可视化脚本）读取

        context = self._merge_heads(context)  # 把多头结果合并回 (B, T, d_model)
        out = self.out_proj(context)  # 最后再过一层线性投影，融合各 head 的信息
        return self.dropout(out)  # 输出前施加 dropout，返回最终结果
