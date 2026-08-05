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
            "properties": {"query": {"type": "string", "description": "Article title or search query"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def run(args: dict) -> str:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")
    headers = {"User-Agent": "llm-beginner-tool-agent/1.0 (educational)"}
    for language in ("zh", "en"):
        url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{quote(query.strip())}"
        response = requests.get(url, headers=headers, timeout=8)
        if response.ok:
            payload = response.json()
            extract = payload.get("extract")
            if extract:
                return extract
    raise RuntimeError(f"Wikipedia has no accessible summary for: {query}")