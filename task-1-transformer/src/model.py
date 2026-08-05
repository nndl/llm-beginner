"""TransformerClassifier：Embedding + 正弦位置编码 + N 层 TransformerBlock + [CLS] pooling + 分类头。"""
import math  # 用于计算位置编码里的 log(10000.0)

import torch
import torch.nn as nn

from src.block import TransformerBlock
from src.vocab import PAD_ID, encode


class PositionalEncoding(nn.Module):
    """标准正弦/余弦绝对位置编码（Attention Is All You Need 式）。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # 预先分配位置编码矩阵，形状 (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 位置索引 0..max_len-1，unsqueeze 后形状 (max_len, 1)，方便与 div_term 广播相乘
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        # 频率项：对 d_model 里偶数下标计算 10000^(-2i/d_model)，用 exp(log) 形式实现幂运算
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维用 cos
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
        # 注册为 buffer：不是可训练参数，但会随 model.to(device) 移动，且会存进 state_dict

    def forward(self, x):
        T = x.size(1)  # 取当前输入的序列长度
        return x + self.pe[:, :T]  # 截取前 T 个位置的编码，与输入逐元素相加（广播 batch 维）


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,           # 词表大小
        d_model: int = 128,        # 隐藏层维度
        n_heads: int = 4,          # 多头注意力头数
        n_layers: int = 4,         # TransformerBlock 堆叠层数
        d_ff: int = 512,           # FFN 中间层维度
        num_classes: int = 2,      # 分类类别数
        max_len: int = 128,        # 序列最大长度
        dropout: float = 0.1,      # dropout 概率
        pad_id: int = PAD_ID,      # padding 的 token id
        use_residual: bool = True,     # 是否使用残差连接（消融实验开关）
        use_layernorm: bool = True,    # 是否使用 LayerNorm（消融实验开关）
        position_encoding: str = "absolute",
    ):
        super().__init__()
        self.pad_id = pad_id  # 记录 pad id，供后续生成 padding mask 用
        self.max_len = max_len  # 记录最大长度
        if position_encoding not in {"absolute", "rope"}:
            raise ValueError("position_encoding must be 'absolute' or 'rope'")
        self.position_encoding = position_encoding

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # 字符 id -> 向量；padding_idx 指定的行永远保持 0 向量且不参与梯度更新
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)  # 位置编码模块
        self.emb_dropout = nn.Dropout(dropout)  # embedding + 位置编码之后的 dropout

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_ff, dropout=dropout,
                    use_residual=use_residual,
                    use_layernorm=use_layernorm,
                    use_rope=position_encoding == "rope",
                    max_len=max_len,
                )
                for _ in range(n_layers)  # 按 n_layers 堆叠多个相同结构的 TransformerBlock
            ]
        )

        self.classifier = nn.Linear(d_model, num_classes)  # 分类头：把 [CLS] 表示映射到类别 logits

    def _make_padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        # (B, T) -> (B, 1, 1, T)，True = 该 key 位置是 PAD，需要屏蔽
        pad = ids == self.pad_id  # 逐元素比较，得到布尔张量 (B, T)，True 表示是 padding
        return pad.unsqueeze(1).unsqueeze(1)
        # 插入两个维度变成 (B, 1, 1, T)，方便广播到注意力打分矩阵 (B, H, T_q, T_k)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) LongTensor -> logits: (B, num_classes)"""
        mask = self._make_padding_mask(ids)  # 生成 padding mask，屏蔽掉 pad 位置的 attention

        x = self.embedding(ids)  # (B, T) -> (B, T, d_model)，字符 id 转向量
        if self.position_encoding == "absolute":
            x = self.pos_encoding(x)  # 叠加位置编码，注入序列顺序信息
        x = self.emb_dropout(x)  # 对 embedding+位置编码结果做 dropout

        for block in self.blocks:  # 依次经过每一层 TransformerBlock
            x = block(x, mask=mask)

        cls_repr = x[:, 0, :]  # [CLS] 位置做 pooling
        # 取序列第 0 个位置（即 [CLS]）的向量作为整句的表示
        logits = self.classifier(cls_repr)  # 线性映射得到各类别的 logits
        return logits  # 返回未归一化的分类得分，配合交叉熵损失使用


def load_for_eval(ckpt_path: str):
    """自检 / 部署统一入口：加载 checkpoint，返回 (model, tokenize_fn)。

    tokenize_fn(text: str) -> LongTensor 形状 (T,)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # 加载 checkpoint 到 CPU；weights_only=False 允许反序列化非纯张量对象（如 config dict）
    config = ckpt["config"]  # 训练时保存的模型超参数
    char2id = ckpt["char2id"]  # 训练时保存的字符词表

    model = TransformerClassifier(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        num_classes=config["num_classes"],
        max_len=config["max_len"],
        dropout=0.0,  # 推理阶段关闭 dropout，保证结果确定
        pad_id=config.get("pad_id", PAD_ID),  # 兼容旧 checkpoint：没存 pad_id 时用默认值
        use_residual=config.get("use_residual", True),
        use_layernorm=config.get("use_layernorm", True),
        position_encoding=config.get("position_encoding", "absolute"),
    )
    model.load_state_dict(ckpt["state_dict"])  # 加载训练好的权重
    model.eval()  # 切换到 eval 模式（关闭 dropout、BN 用运行统计量等）

    max_len = config["max_len"]  # 取出序列最大长度，供 tokenize_fn 使用

    def tokenize_fn(text: str) -> torch.Tensor:
        ids = encode(text, char2id, max_len)  # 文本 -> 定长 id 列表（含 [CLS]/PAD 处理）
        return torch.tensor(ids, dtype=torch.long)  # 转成 LongTensor 供模型输入

    return model, tokenize_fn  # 返回模型和配套的分词函数，供自检/部署脚本调用
