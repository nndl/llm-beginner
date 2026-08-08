"""S4 extension: a RoPE-based classifier kept separate from the required model."""
import math

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention, scaled_dot_product_attention
from src.vocab import PAD_ID


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class RotaryMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, max_len: int = 512):
        super().__init__(d_model, n_heads, dropout=dropout)
        if self.d_k % 2 != 0:
            raise ValueError("RoPE requires an even per-head dimension")
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, self.d_k, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_k))
        angles = positions * frequencies
        self.register_buffer("rope_cos", torch.repeat_interleave(torch.cos(angles), 2, dim=-1)[None, None])
        self.register_buffer("rope_sin", torch.repeat_interleave(torch.sin(angles), 2, dim=-1)[None, None])

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        sequence_length = x.size(1)
        Q = apply_rope(self._split_heads(self.q_proj(x)), self.rope_cos[:, :, :sequence_length], self.rope_sin[:, :, :sequence_length])
        K = apply_rope(self._split_heads(self.k_proj(x)), self.rope_cos[:, :, :sequence_length], self.rope_sin[:, :, :sequence_length])
        V = self._split_heads(self.v_proj(x))
        weights_container = []
        context = scaled_dot_product_attention(Q, K, V, mask=mask, attn_weights_out=weights_container)
        self.last_attn = weights_container[0]
        return self.dropout(self.out_proj(self._merge_heads(context)))


class RotaryTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_residual=True, use_layernorm=True, max_len=512):
        super().__init__()
        self.use_residual = use_residual
        self.attn = RotaryMultiHeadAttention(d_model, n_heads, dropout, max_len)
        self.norm1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.dropout1(self.attn(x, mask=mask))
        x = self.norm1(x + attn_out if self.use_residual else attn_out)
        ffn_out = self.dropout2(self.ffn(x))
        return self.norm2(x + ffn_out if self.use_residual else ffn_out)


class RopeTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=512, num_classes=2, max_len=128, dropout=0.1, pad_id=PAD_ID, use_residual=True, use_layernorm=True):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([RotaryTransformerBlock(d_model, n_heads, d_ff, dropout, use_residual, use_layernorm, max_len) for _ in range(n_layers)])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids == self.pad_id).unsqueeze(1).unsqueeze(1)
        x = self.emb_dropout(self.embedding(ids))
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.classifier(x[:, 0, :])