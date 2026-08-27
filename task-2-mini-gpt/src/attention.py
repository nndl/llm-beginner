"""因果多头自注意力（Causal Multi-Head Self-Attention），支持 RoPE 与 KV cache。

与 task-1 的双向注意力不同，这里必须加因果 mask（每个位置只能看到自己和之前的
位置），这是"只能从前往后生成"的解码器（decoder-only）语言模型的核心约束。

KV cache 的原理：自回归生成时，第 t 步只新增了 1 个 token，但如果每步都把
0..t 的完整序列重新过一遍 Attention，K/V 里 0..t-1 那部分其实是重复计算。
KV cache 就是把历史 token 算好的 K、V 存下来，新 token 只需要：
    1) 计算自己的 Q/K/V（K/V 只有新 token 这一份）；
    2) 把新 K/V 拼接到缓存的 K/V 后面；
    3) 用新 token 的 Q 和"缓存+新增"的完整 K/V 做注意力。
这样每步只需 O(1) 个新 token 的前向计算，而不是 O(t)。

正确性关键点：RoPE 的角度是按"绝对位置"算的，新 token 的绝对位置 = 已缓存的
K/V 长度（不是 0！），否则相对位置关系会算错——这也是 kv_cache 参数必须和
RoPE 的 offset 联动的原因。
"""
import math  # 使用 sqrt(head_dim) 做缩放点积注意力

import torch
import torch.nn as nn

from src.rope import RotaryEmbedding, apply_rope


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, C) -> (B, H, T, C/H)"""
    B, T, C = x.shape  # 解包 batch、序列长度和总隐藏维度
    head_dim = C // n_heads  # 每个注意力头分到的通道数
    x = x.view(B, T, n_heads, head_dim)  # 将总通道拆为 head 数量和每头维度
    return x.transpose(1, 2)  # 调整为 attention 计算习惯的 (B,H,T,Dh)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T, Dh) -> (B, T, H*Dh)"""
    B, H, T, Dh = x.shape  # 读取多头输出的各维度
    x = x.transpose(1, 2).contiguous()  # 换回时间维在前，并确保 view 前内存连续
    return x.view(B, T, H * Dh)  # 拼回 d_model 维度


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048,
                 dropout: float = 0.0, rope_base: float = 10000.0,
                 use_rope: bool = True):
        super().__init__()  # 初始化 Module 基类
        assert d_model % n_heads == 0  # 每个头必须得到整数维度
        self.n_heads = n_heads  # 保存头数，供 forward 切分 Q/K/V
        self.head_dim = d_model // n_heads  # 保存单头维度，供缩放和 RoPE 使用

        # 一次线性层同时算出 Q、K、V，再切分，减少 matmul 次数
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)

    def forward(self, x: torch.Tensor, kv_cache=None, return_cache: bool = False):
        """
        x: (B, T, C)，T 是本次新输入的 token 数（首次前向是整段 prompt，
           增量解码时通常 T=1）。
        kv_cache: None，或 (past_k, past_v)，past_k/past_v 形状 (B, H, T_past, Dh)。
        return_cache: True 时同时返回拼接后的新 (k, v) 供下一步复用。
        """
        B, T, C = x.shape  # C 应等于构造时的 d_model
        qkv = self.qkv_proj(x)  # 一次投影得到拼接的 Q、K、V，形状 (B,T,3C)
        q, k, v = qkv.split(C, dim=-1)  # 沿最后一维均分回三个 (B,T,C) 张量
        q = _split_heads(q, self.n_heads)  # (B, H, T, Dh)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        past_len = 0  # 无缓存时，本段第一个 token 的绝对位置为 0
        past_k = past_v = None  # 初始化占位，便于后续统一拼接逻辑
        if kv_cache is not None:
            past_k, past_v = kv_cache  # 解包该层历史 K/V
            past_len = past_k.size(2)  # 缓存序列维长度就是新 token 的位置偏移

        if self.use_rope:
            # 新 token 的绝对位置从 past_len 开始，RoPE 角度必须按绝对位置取4
            cos, sin = self.rope(T, offset=past_len)  # 取当前这段 token 的绝对位置旋转角度
            cos, sin = cos.to(x.device), sin.to(x.device)  # buffer 可能仍在 CPU，显式对齐输入设备
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([past_k, k], dim=2)  # 沿序列维把历史 K 接到新 K 前面
            v = torch.cat([past_v, v], dim=2)  # V 使用相同顺序，保证键和值位置一一对应

        new_cache = (k, v) if return_cache else None  # 仅调用者请求时返回缓存，训练阶段避免无用引用

        Tq = T  # 查询数就是本次输入 token 数
        Tk = k.size(2)  # 键数包含可能拼接的历史缓存
        # 因果 mask：query 绝对位置 i = past_len..past_len+Tq-1，
        # key 绝对位置 j = 0..Tk-1；只允许 j <= i
        i = torch.arange(Tq, device=x.device).unsqueeze(1) + past_len  # (Tq, 1)
        j = torch.arange(Tk, device=x.device).unsqueeze(0)             # (1, Tk)
        causal_mask = j > i  # True 表示 key 在 query 的未来，softmax 前必须屏蔽

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, H, Tq, Tk)
        scores = scores.masked_fill(causal_mask, float("-inf"))  # 未来位置概率在 softmax 后变为 0
        weights = torch.softmax(scores, dim=-1)  # 将每个 query 对所有允许 key 的分数归一化
        weights = self.attn_dropout(weights)

        out = weights @ v  # (B, H, Tq, Dh)
        out = _merge_heads(out)  # (B, Tq, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, new_cache  # 返回残差分支输出，以及可选的累计 K/V
