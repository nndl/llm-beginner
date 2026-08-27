"""MiniGPT：解码器（decoder-only）小型 GPT。

结构（自底向上）：
    token embedding（没有单独的绝对位置 embedding —— 位置信息由 RoPE 在
    Attention 内部注入） -> N 个 Block（Pre-LN + 因果自注意力 + 残差 +
    Pre-LN + MLP + 残差） -> 最终 LayerNorm -> 线性层投影到词表大小，得到
    每个位置预测下一个 token 的 logits。

与 task-1 的 TransformerClassifier 的关键区别：
    1) 注意力是因果的（只能看过去），而不是双向的；
    2) 位置编码用 RoPE，而不是加在 embedding 上的正弦位置编码；
    3) 输出是"每个位置的下一个 token 分布"（语言模型），而不是整个序列一个类别；
    4) 支持 kv_cache 增量解码，用于 generate() 时加速自回归生成。
"""
import math  # 残差投影初始化使用 sqrt(层数) 缩放
from pathlib import Path  # 处理 checkpoint 和 tokenizer 的同目录路径

import torch
import torch.nn as nn

from src.attention import CausalSelfAttention
from src.sampling import sample_next_token
from src.tokenizer import BPETokenizer


class Block(nn.Module):
    """Pre-LN Transformer Block：LN -> Attn -> 残差，LN -> MLP -> 残差。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int,
                 dropout: float = 0.1, rope_base: float = 10000.0,
                 use_rope: bool = True):
        super().__init__()  # 初始化 Module 基类
        self.ln1 = nn.LayerNorm(d_model)  # 注意力前的 Pre-LayerNorm
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len=max_seq_len,
                                         dropout=dropout, rope_base=rope_base,
                                         use_rope=use_rope)
        self.ln2 = nn.LayerNorm(d_model)  # MLP 前的第二个 Pre-LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, kv_cache=None, return_cache: bool = False):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache, return_cache=return_cache)  # 归一化后计算因果注意力
        x = x + attn_out  # 第一条残差连接保留原始表示并加入注意力更新
        x = x + self.mlp(self.ln2(x))  # 第二条残差连接加入归一化后的前馈网络更新
        return x, new_cache  # 将本层表示和可选 K/V 缓存交给下一层


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int = 512, max_seq_len: int = 128,
                 dropout: float = 0.1, rope_base: float = 10000.0,
                 rope_cache_len: int | None = None,
                 positional_encoding: str = "rope"):
        super().__init__()  # 初始化 Module 容器
        self.vocab_size = vocab_size  # 记录输出类别数，便于检查和序列化配置
        self.d_model = d_model  # 保存隐藏层宽度
        self.n_heads = n_heads  # 保存注意力头数
        self.n_layers = n_layers  # 保存 Block 堆叠数量
        self.d_ff = d_ff  # 保存 MLP 中间层宽度
        # block_size / max_seq_len：训练时切片的上下文长度，eval/run.py 用它来
        # 分块喂 dev.txt 计算困惑度，两个名字都保留以兼容不同调用习惯
        self.max_seq_len = max_seq_len
        self.block_size = max_seq_len
        self.rope_base = rope_base
        if positional_encoding not in {"rope", "absolute"}:
            raise ValueError("positional_encoding must be 'rope' or 'absolute'")
        self.positional_encoding = positional_encoding

        # RoPE 预计算表按 rope_cache_len 生成，需要覆盖"prompt + 生成长度"，
        # 因此单独给一个比训练 block_size 更大的上限，允许 generate() 时
        # 累计位置超过训练时用的 max_seq_len 一定范围（合理外推）
        self.rope_cache_len = rope_cache_len or max(max_seq_len * 4, 1024)

        self.tok_emb = nn.Embedding(vocab_size, d_model)  # token id 映射到 d_model 维向量
        self.pos_emb = (
            nn.Embedding(self.rope_cache_len, d_model)
            if positional_encoding == "absolute" else None
        )
        self.drop = nn.Dropout(dropout)  # 在 embedding 后做正则化
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, max_seq_len=self.rope_cache_len,
                  dropout=dropout, rope_base=rope_base,
                  use_rope=positional_encoding == "rope")
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)  # 输出到词表投影前的最终归一化
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # 每个位置输出下一个 token 的 logits

        # PyTorch 的 Embedding 默认按 N(0, 1) 初始化。这里输出头与 embedding
        # 共享权重，若直接沿用该默认值，点积 logits 的方差会随 d_model 急剧放大，
        # 导致训练从远高于 log(vocab_size) 的交叉熵开始。使用 GPT 常用的小方差
        # 初始化，并对每个残差分支的输出投影按层数缩放，保证深层残差累加稳定。
        self.apply(self._init_weights)  # 递归将 GPT 风格初始化应用到所有子模块
        residual_std = 0.02 / math.sqrt(2 * n_layers)  # 残差分支随层数增长而缩小初始方差
        for name, parameter in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

        # 权重共享（weight tying）：输入 embedding 和输出投影共用一份参数，
        # 常见于 GPT 系列，能减少参数量并略微提升效果
        self.head.weight = self.tok_emb.weight  # 令输入词向量与输出分类器共享同一个 Parameter

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT 风格初始化，避免 tied embedding 产生过大的初始 logits。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor, kv_cache=None, return_cache: bool = False):
        """
        ids: (B, T) token id
        kv_cache: None，或长度为 n_layers 的列表，每层是 (past_k, past_v)
        return_cache: True 时返回 (logits, new_kv_cache)
        """
        position_offset = 0 if kv_cache is None else kv_cache[0][0].size(2)
        if position_offset + ids.size(1) > self.rope_cache_len:
            raise ValueError("input sequence exceeds configured position embedding length")
        x = self.tok_emb(ids)  # id -> embedding
        if self.pos_emb is not None:
            positions = torch.arange(
                position_offset, position_offset + ids.size(1), device=ids.device
            )
            x = x + self.pos_emb(positions).unsqueeze(0)
        x = self.drop(x)  # 在训练模式按概率置零部分元素
        new_caches = [] if return_cache else None  # 仅增量解码需要收集每一层的新缓存
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None  # 为当前 Block 取出自己的历史 K/V
            x, cache = block(x, kv_cache=layer_cache, return_cache=return_cache)  # 更新隐藏表示并计算本层缓存
            if return_cache:
                new_caches.append(cache)  # 缓存列表顺序与 blocks 顺序严格对应
        x = self.ln_f(x)  # 所有 Block 完成后，统一最终隐藏表示的尺度
        logits = self.head(x)  # 将最终表示投影为词表中每个 token 的未归一化分数
        if return_cache:
            return logits, new_caches
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None, use_kv_cache: bool = True) -> list[int]:
        """自回归生成；可关闭 KV cache 以验证等价性或进行性能对比。"""
        self.eval()  # 关闭 dropout，确保生成时不引入训练期随机失活
        device = next(self.parameters()).device  # 从任一参数读取模型实际所在设备
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        if use_kv_cache:
            logits, cache = self.forward(ids, kv_cache=None, return_cache=True)  # 首次前向一次性建立 prompt 的 K/V 缓存
        else:
            logits = self.forward(ids)  # 对照路径：每步重新计算完整前缀
            cache = None
        next_logits = logits[:, -1, :]  # prompt 最后位置的分布预测第一个新 token
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            next_id = sample_next_token(next_logits, temperature=temperature,
                                         top_k=top_k, top_p=top_p)
            generated.append(int(next_id.item()))  # 同时维护 Python id 列表，供最终 tokenizer 解码
            if use_kv_cache:
                logits, cache = self.forward(next_id, kv_cache=cache, return_cache=True)  # 之后每次只处理一个新 token
            else:
                full_ids = torch.tensor([generated], dtype=torch.long, device=device)
                logits = self.forward(full_ids)
            next_logits = logits[:, -1, :]

        return generated


def load_for_eval(ckpt_path: str):
    """加载训练好的 checkpoint，返回 (model, tokenizer)，供 eval/run.py 及生成脚本使用。

    约定：tokenizer.json 与 best.pt 放在同一个 ckpt/ 目录下。
    """
    ckpt_path = Path(ckpt_path)  # 规范化路径，便于定位同目录 tokenizer
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # 在 CPU 加载 state_dict 和配置
    config = ckpt["config"]  # 读取训练时写入的模型结构超参数

    tok_path = ckpt_path.parent / "tokenizer.json"
    tokenizer = BPETokenizer.from_pretrained(str(tok_path))

    model = MiniGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0,  # eval 时关闭 dropout
        rope_base=config.get("rope_base", 10000.0),
        rope_cache_len=config.get("rope_cache_len"),
        positional_encoding=config.get("positional_encoding", "rope"),
    )
    model.load_state_dict(ckpt["state_dict"])  # 覆盖初始化权重，恢复训练得到的参数
    model.eval()  # 显式进入评估模式，关闭所有 dropout
    return model, tokenizer  # 向自检和生成调用方返回匹配的一对对象
