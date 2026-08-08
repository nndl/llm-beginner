"""TransformerBlock = MultiHeadAttention + FFN + 两个 residual + 两个 LayerNorm。"""
import torch.nn as nn

from src.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,      # 隐藏层维度
        n_heads: int,       # 多头注意力的头数
        d_ff: int,          # 前馈网络（FFN）中间层维度
        dropout: float = 0.1,          # dropout 概率
        use_residual: bool = True,     # 是否使用残差连接（消融实验开关）
        use_layernorm: bool = True,    # 是否使用 LayerNorm（消融实验开关）
    ):
        super().__init__()
        self.use_residual = use_residual      # 记录是否启用残差连接
        self.use_layernorm = use_layernorm    # 记录是否启用 LayerNorm

        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        # 注意力子层后的归一化；关闭时用 Identity（恒等映射）占位，保持代码结构一致
        self.dropout1 = nn.Dropout(dropout)  # 注意力输出上的 dropout

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),   # 升维：d_model -> d_ff
            nn.GELU(),                  # 非线性激活函数
            nn.Dropout(dropout),        # FFN 内部的 dropout
            nn.Linear(d_ff, d_model),   # 降维：d_ff -> d_model，恢复原始维度
        )
        self.norm2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        # FFN 子层后的归一化，同样可通过 use_layernorm 开关
        self.dropout2 = nn.Dropout(dropout)  # FFN 输出上的 dropout

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask=mask)  # 先过多头自注意力，得到注意力输出
        x = (x + self.dropout1(attn_out)) if self.use_residual else self.dropout1(attn_out)
        # 若启用残差：输入 x 与 dropout 后的注意力输出相加；否则直接用 dropout 后的输出替代 x
        x = self.norm1(x)  # 经过第一个 LayerNorm（或 Identity）

        ffn_out = self.ffn(x)  # 再过前馈网络
        x = (x + self.dropout2(ffn_out)) if self.use_residual else self.dropout2(ffn_out)
        # 同理：FFN 输出与 x 做残差相加（或直接替代）
        x = self.norm2(x)  # 经过第二个 LayerNorm（或 Identity）
        return x  # 返回该 Transformer Block 的最终输出
