"""RoPE：旋转位置编码（Rotary Position Embedding）。

与 task-1 里用的绝对正弦位置编码不同，RoPE 不是"加到" embedding 上的一个向量，
而是在计算 Attention 的 Q、K 时，把每个位置对应的一个"旋转矩阵"作用在 Q/K 的
向量上（V 不做旋转）。两个位置 i、j 的 Q_i、K_j 做内积时，旋转角度之差自动只
和相对位置 (i-j) 有关——这就是"旋转位置编码"能自然表达相对位置信息、且对短序列
训练、长序列外推更友好的原因。

实现上采用类似 LLaMA 的写法：把最后一维（head_dim）从中间切成两半 [x1, x2]，
则"旋转"等价于：
    rotate_half(x) = concat(-x2, x1)
    apply_rope(x)  = x * cos + rotate_half(x) * sin
其中 cos/sin 是提前按位置和频率算好的表，频率公式为 base^(-2i/dim)（base 默认取
10000，与原始 Transformer 的正弦位置编码频率公式一致）。
"""
import torch  # 提供向量化频率表与张量计算
import torch.nn as nn  # RotaryEmbedding 继承 Module，以便缓存随模型迁移设备


class RotaryEmbedding(nn.Module):
    """预计算 cos/sin 表，供 apply_rope 在指定位置区间取用。"""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()  # 初始化 nn.Module 的参数/buffer 注册机制
        assert dim % 2 == 0, "RoPE 要求 head_dim 为偶数"
        # inv_freq[i] = base^(-2i/dim)，i = 0..dim/2-1，频率从高到低
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
        t = torch.arange(max_seq_len).float()  # (max_seq_len,)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)：每个位置 x 每个频率的角度
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)，前后两半用同一组角度
        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # 缓存余弦表，随 .to(device) 移动但不写进 checkpoint
        self.register_buffer("sin_cached", emb.sin(), persistent=False)  # 缓存与余弦表配套的正弦表
        self.max_seq_len = max_seq_len  # 记录可用的最大绝对位置，供 forward 做边界检查

    def forward(self, seq_len: int, offset: int = 0):
        """返回位置 [offset, offset+seq_len) 对应的 cos/sin，形状 (seq_len, dim)。

        offset 就是 KV cache 已缓存的长度：增量解码时新 token 的绝对位置
        不是从 0 开始，而是从 offset 开始，必须据此取用正确的旋转角度，
        否则新旧 token 的相对位置关系会算错。
        """
        end = offset + seq_len  # 计算本次输入最后一个 token 后的开区间端点
        assert end <= self.max_seq_len, (
            f"RoPE 预计算表长度 {self.max_seq_len} 不足以覆盖位置 {end}，"
            "请在构造模型时调大 max_seq_len")
        return self.cos_cached[offset:end], self.sin_cached[offset:end]  # 沿位置维切片，不复制整个预计算表


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把最后一维切成两半 [x1, x2]，返回 [-x2, x1]。"""
    x1, x2 = x.chunk(2, dim=-1)  # 沿 head_dim 将向量平均分为前、后半部分
    return torch.cat((-x2, x1), dim=-1)  # 实现二维旋转矩阵 [0,-1;1,0] 的等价排列


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 Q 或 K 应用旋转位置编码。

    x:   (B, H, T, D)
    cos/sin: (T, D)，会广播到 (1, 1, T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # 补 batch/head 维，形状变为 (1,1,T,D) 以自动广播
    sin = sin.unsqueeze(0).unsqueeze(0)  # 同样补维，使它能与 x 对齐逐元素运算
    return x * cos + rotate_half(x) * sin  # 应用 cos/sin 旋转公式并返回位置编码后的 Q 或 K
