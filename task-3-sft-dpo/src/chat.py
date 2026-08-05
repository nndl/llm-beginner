"""Qwen ChatML 模板与仅监督 assistant 内容的标签构造。"""
from __future__ import annotations

from typing import Sequence

import torch

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_messages(messages: Sequence[dict]) -> str:
    """按 Qwen 官方 ChatML 形式拼接完整的多轮对话。"""
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"不支持的角色：{role}")
        content = str(message.get("content", ""))
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return "".join(parts)


def format_prompt(messages: Sequence[dict]) -> str:
    """构造待生成 assistant 回复的 Qwen prompt（末尾不含 assistant 内容）。"""
    return format_messages(messages) + f"{IM_START}assistant\n"


def build_labels(input_ids: torch.Tensor, messages: Sequence[dict]) -> torch.Tensor:
    """将 system/user/控制 token 置 ``-100``，只保留每个 assistant 内容 token。

    Qwen 的 ChatML special token 为 ``<|im_start|>`` 与 ``<|im_end|>``。由于评测接口
    只传入编码后的 ``input_ids``，本函数按这两个边界扫描每一轮，并按 ``messages`` 的
    角色顺序匹配。每一轮第一个换行 token 之前是 role/header，必须继续 mask。
    """
    if input_ids.ndim != 1:
        raise ValueError("build_labels 期望形状为 (sequence_length,) 的 input_ids")
    labels = torch.full_like(input_ids, -100)
    # Qwen2/Qwen2.5 的官方 ChatML special token id；若 tokenizer 配置改变，仍可从
    # 消息轮次顺序安全地保持所有 token 为 ignore，而不会误监督 user 内容。
    im_start_id, im_end_id = 151644, 151645
    starts = (input_ids == im_start_id).nonzero(as_tuple=False).flatten().tolist()
    ends = (input_ids == im_end_id).nonzero(as_tuple=False).flatten().tolist()
    end_index = 0
    message_index = 0
    for start in starts:
        while end_index < len(ends) and ends[end_index] < start:
            end_index += 1
        if end_index >= len(ends) or message_index >= len(messages):
            break
        end = ends[end_index]
        role = messages[message_index].get("role")
        # 角色名与内容由第一个换行分隔；header 和 im_end 都不能参与 loss。
        body_start = start + 1
        newline_positions = (input_ids[body_start:end] == 198).nonzero(as_tuple=False).flatten()
        if role == "assistant" and len(newline_positions) > 0:
            content_start = body_start + int(newline_positions[0].item()) + 1
            if content_start < end:
                labels[content_start:end] = input_ids[content_start:end]
        message_index += 1
        end_index += 1
    return labels
