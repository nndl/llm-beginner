"""Wikipedia summary lookup through its public REST API."""
from urllib.parse import quote
import requests

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": "Look up a concise Wikipedia summary in Chinese or English.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Article title or search query"}},# 要查询的 Wikipedia 词条名称或者查询关键词。
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

# ============================================================
# 工具真正的执行入口
# ============================================================
def run(args: dict) -> str:
    """
    查询 Wikipedia 摘要。

    args 示例：

    {
        "query": "Python"
    }

    返回：
        Wikipedia 词条的摘要文本。
    """
    # --------------------------------------------------------
    # 1. 获取查询关键词
    # --------------------------------------------------------
    query = args.get("query")
    # --------------------------------------------------------
    # 2. 检查 query 是否有效
    # --------------------------------------------------------
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")
    # --------------------------------------------------------
    # 3. 设置 HTTP 请求头
    # --------------------------------------------------------
    headers = {"User-Agent": "llm-beginner-tool-agent/1.0 (educational)"}
    # User-Agent 用来告诉 Wikipedia：
    #
    # “是谁在访问你的 API？”
    #
    # 这里设置的是一个教学项目的标识。
    #
    # 一些公共 API 会要求客户端提供合理的 User-Agent，
    # 方便服务方识别和管理请求。

    # --------------------------------------------------------
    # 4. 依次尝试中文和英文 Wikipedia
    # --------------------------------------------------------
    for language in ("zh", "en"):
        # ----------------------------------------------------
        # 5. 构造 Wikipedia API URL
        # ----------------------------------------------------
        url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{quote(query.strip())}"
        # ----------------------------------------------------
        # 6. 发送 HTTP GET 请求
        # ----------------------------------------------------
        response = requests.get(url, headers=headers, timeout=8)
        # ----------------------------------------------------
        # 7. 判断 HTTP 请求是否成功
        # ----------------------------------------------------
        if response.ok:
            # ------------------------------------------------
            # 8. 将 JSON 响应解析成 Python 字典
            # ------------------------------------------------
            payload = response.json()
            # ------------------------------------------------
            # 9. 获取摘要内容
            # ------------------------------------------------
            extract = payload.get("extract")
            # ------------------------------------------------
            # 10. 如果有摘要，直接返回
            # ------------------------------------------------
            if extract:
                return extract
    # ========================================================
    # 11. 中文和英文 Wikipedia 都没有得到摘要
    # ========================================================
    raise RuntimeError(f"Wikipedia has no accessible summary for: {query}")