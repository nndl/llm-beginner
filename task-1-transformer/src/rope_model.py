"""S4 加分实现：基于 RoPE 的分类器，与必做模型分文件保存。"""
import math  # 用于计算 RoPE 的频率项

import torch  # PyTorch 张量运算库
import torch.nn as nn  # 神经网络模块

from src.attention import MultiHeadAttention, scaled_dot_product_attention  # 复用必做的注意力基础实现
from src.vocab import PAD_ID  # 复用词表中定义的 PAD token 编号


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对最后一维的相邻偶奇维应用二维旋转位置编码。"""
    x_even = x[..., 0::2]  # 取最后一维的偶数下标，例如 x0、x2、x4
    x_odd = x[..., 1::2]  # 取最后一维的奇数下标，例如 x1、x3、x5
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)  # 构造每一对维度旋转 90 度后的向量 (-odd, even)
    return x * cos + rotated * sin  # 按 RoPE 公式 x*cos(theta) + rotate(x)*sin(theta) 完成旋转


class RotaryMultiHeadAttention(MultiHeadAttention):
    """在 Q、K 上施加 RoPE 的多头自注意力。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, max_len: int = 512):
        super().__init__(d_model, n_heads, dropout=dropout)  # 初始化必做多头注意力中的投影层和 head 信息
        if self.d_k % 2 != 0:  # RoPE 每次处理一对相邻维度，因此每个 head 的维度必须为偶数
            raise ValueError("RoPE requires an even per-head dimension")  # 维度不满足条件时提前报错
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # 生成 (max_len, 1) 的位置编号列向量
        frequency_indices = torch.arange(0, self.d_k, 2, dtype=torch.float)  # 为每个偶数维生成频率编号
        frequencies = torch.exp(frequency_indices * (-math.log(10000.0) / self.d_k))  # 计算 RoPE 的指数递减频率
        angles = positions * frequencies  # 通过广播得到每个位置、每个维度对的旋转角度
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)  # 每个维度对共享同一个余弦值，扩展回 d_k 维
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)  # 每个维度对共享同一个正弦值，扩展回 d_k 维
        self.register_buffer("rope_cos", cos[None, None])  # 保存为 (1, 1, max_len, d_k) 的非训练 buffer
        self.register_buffer("rope_sin", sin[None, None])  # 保存为同形状的正弦 buffer，并随模型迁移设备

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        sequence_length = x.size(1)  # 读取当前批次的实际序列长度
        rope_cos = self.rope_cos[:, :, :sequence_length]  # 截取当前长度需要的余弦位置编码
        rope_sin = self.rope_sin[:, :, :sequence_length]  # 截取当前长度需要的正弦位置编码
        Q = self._split_heads(self.q_proj(x))  # 输入先投影为 Query，再拆分为多头 (B, H, T, d_k)
        K = self._split_heads(self.k_proj(x))  # 输入先投影为 Key，再拆分为多头
        V = self._split_heads(self.v_proj(x))  # 输入先投影为 Value；RoPE 不作用于 V
        Q = apply_rope(Q, rope_cos, rope_sin)  # 给 Query 注入相对位置信息
        K = apply_rope(K, rope_cos, rope_sin)  # 给 Key 注入相对位置信息
        weights_container = []  # 创建容器以接收注意力权重，供后续热图可视化
        context = scaled_dot_product_attention(  # 调用必做的缩放点积注意力计算上下文
            Q, K, V, mask=mask, attn_weights_out=weights_container
        )
        self.last_attn = weights_container[0]  # 保存本次前向传播的注意力权重
        merged_context = self._merge_heads(context)  # 合并多头结果，恢复为 (B, T, d_model)
        return self.dropout(self.out_proj(merged_context))  # 输出投影并施加 dropout


class RotaryTransformerBlock(nn.Module):
    """由 RoPE 注意力、残差连接、LayerNorm 和 FFN 组成的编码器层。"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_residual=True, use_layernorm=True, max_len=512):
        super().__init__()  # 初始化 nn.Module 基类
        self.use_residual = use_residual  # 记录是否启用残差连接，支持 S2 消融
        self.attn = RotaryMultiHeadAttention(d_model, n_heads, dropout, max_len)  # 创建带 RoPE 的注意力子层
        self.norm1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()  # 创建注意力后的 LayerNorm，消融时改为恒等映射
        self.dropout1 = nn.Dropout(dropout)  # 创建注意力输出上的 dropout
        self.ffn = nn.Sequential(  # 定义两层前馈网络
            nn.Linear(d_model, d_ff),  # 先将隐藏维度扩展到 d_ff
            nn.GELU(),  # 使用 GELU 激活函数引入非线性
            nn.Dropout(dropout),  # 在 FFN 中间层施加 dropout
            nn.Linear(d_ff, d_model),  # 再映射回原始隐藏维度
        )
        self.norm2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()  # 创建 FFN 后的 LayerNorm，消融时改为恒等映射
        self.dropout2 = nn.Dropout(dropout)  # 创建 FFN 输出上的 dropout

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask=mask)  # 计算带 RoPE 的多头自注意力输出
        attn_out = self.dropout1(attn_out)  # 对注意力输出施加 dropout
        x = x + attn_out if self.use_residual else attn_out  # 根据开关保留残差相加或直接替换输入
        x = self.norm1(x)  # 对注意力子层输出进行 LayerNorm 或恒等映射
        ffn_out = self.ffn(x)  # 将归一化后的表示送入前馈网络
        ffn_out = self.dropout2(ffn_out)  # 对 FFN 输出施加 dropout
        x = x + ffn_out if self.use_residual else ffn_out  # 根据开关保留第二个残差相加或直接替换
        return self.norm2(x)  # 进行第二次 LayerNorm 或恒等映射并返回结果


class RopeTransformerClassifier(nn.Module):
    """使用 RoPE 而非绝对位置编码的文本分类器。"""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=512, num_classes=2, max_len=128, dropout=0.1, pad_id=PAD_ID, use_residual=True, use_layernorm=True):
        super().__init__()  # 初始化 nn.Module 基类
        self.pad_id = pad_id  # 保存 PAD token 的编号，用于构造注意力掩码
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)  # 将 token 编号映射为 d_model 维词向量
        self.emb_dropout = nn.Dropout(dropout)  # 创建词向量后的 dropout
        self.blocks = nn.ModuleList(  # 使用 ModuleList 注册所有 Transformer 编码器层
            [
                RotaryTransformerBlock(  # 创建一个使用 RoPE 的编码器层
                    d_model, n_heads, d_ff, dropout, use_residual, use_layernorm, max_len
                )
                for _ in range(n_layers)  # 重复 n_layers 次堆叠编码器
            ]
        )
        self.classifier = nn.Linear(d_model, num_classes)  # 将 [CLS] 表示映射为类别 logits

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids == self.pad_id).unsqueeze(1).unsqueeze(1)  # 构造 (B, 1, 1, T) 的 padding mask，True 表示屏蔽
        x = self.embedding(ids)  # 将输入的 token 编号转换为词向量 (B, T, d_model)
        x = self.emb_dropout(x)  # 对词向量施加 dropout；RoPE 在注意力层内部编码位置信息
        for block in self.blocks:  # 按顺序经过每一个 RoPE Transformer 编码器层
            x = block(x, mask=mask)  # 传入 padding mask，避免模型关注 PAD 位置
        cls_repr = x[:, 0, :]  # 取第一个 [CLS] token 的表示作为整句表示
        return self.classifier(cls_repr)  # 将句向量映射为未归一化的分类 logits