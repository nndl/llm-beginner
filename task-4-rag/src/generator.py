"""Context-grounded answer generation through an optional OpenAI-compatible server."""
# 模块说明：
# 通过一个可选的、兼容 OpenAI API 格式的服务，
# 根据检索得到的上下文资料生成回答。
#
# 这里的核心思路是：
# 用户问题 query
#       +
# 检索得到的 sources
#       ↓
# 拼接成 prompt
#       ↓
# 调用本地/远程大模型 API
#       ↓
# 返回基于上下文生成的回答
from __future__ import annotations

import os
from typing import List

import requests


def generate(query: str, sources: List[dict]) -> str:
    context = "\n\n".join(f"[{index + 1}] {item['text']}" for index, item in enumerate(sources))[:6000]
    prompt = ("只能根据下列上下文回答问题。若上下文不足，请明确回答'根据提供的资料无法确定'。"
              "不要补充上下文以外的事实，并在回答末尾标注引用编号。\n\n"
              f"上下文：\n{context}\n\n问题：{query}")
    endpoint = os.getenv("QWEN_API_URL", "http://localhost:11434/v1/chat/completions")
    try:
        response = requests.post(endpoint, json={"model": os.getenv("QWEN_MODEL", "qwen2.5:7b-instruct"),
            "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}, timeout=30)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        if answer:
            return answer
    except (requests.RequestException, KeyError, IndexError, ValueError):
        pass
    return f"根据资料，{sources[0]['text'][:360]} [1]" if sources else "根据提供的资料无法确定。"
