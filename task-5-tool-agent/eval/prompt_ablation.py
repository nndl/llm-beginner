"""S3: compare zero-shot and one-example few-shot ReAct prompts."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ReActAgent

PROMPT_TEMPLATES = ("zero_shot", "few_shot")


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", "", str(text).lower().replace(",", "").replace("，", ""))


def answer_matches(answer: str, expected_keywords: list) -> bool:
    normalized = normalize_answer(answer)
    for expected in expected_keywords:
        if isinstance(expected, list):
            if not any(normalize_answer(option) in normalized for option in expected):
                return False
        elif normalize_answer(expected) not in normalized:
            return False
    return True


def expected_tools_used(trace: dict, expected_tools: list[str]) -> bool:
    used_tools = [step.get("tool") for step in trace.get("steps", [])
                  if isinstance(step, dict) and step.get("tool")]
    return all(tool in used_tools for tool in expected_tools)


def run_template(tasks: list[dict], prompt_template: str) -> dict:
    details = []
    answer_hits = tool_hits = format_failures = 0
    for task in tasks:
        trace = ReActAgent(prompt_template=prompt_template).run(task["task"])
        if trace.get("final_answer") == "Model backend unavailable.":
            raise RuntimeError("模型服务不可用；请检查 OPENAI_BASE_URL、OPENAI_MODEL 和服务进程")
        answer_ok = answer_matches(trace.get("final_answer", ""),
                                   task["expected_answer_contains"])
        tools_ok = expected_tools_used(trace, task["expected_tools"])
        format_failed = any("Format error:" in str(step.get("observation", ""))
                            for step in trace.get("steps", []) if isinstance(step, dict))
        answer_hits += int(answer_ok)
        tool_hits += int(tools_ok)
        format_failures += int(format_failed)
        details.append({"id": task["id"], "answer_ok": answer_ok, "tools_ok": tools_ok,
                        "format_failed": format_failed,
                        "final_answer_preview": str(trace.get("final_answer", ""))[:120]})
    total = len(tasks)
    return {"prompt_template": prompt_template, "n": total,
            "answer_keyword_rate": round(answer_hits / total, 3),
            "expected_tool_coverage_rate": round(tool_hits / total, 3),
            "format_failure_rate": round(format_failures / total, 3), "details": details}


def main() -> None:
    if os.environ.get("USE_LLM_AGENT") != "1":
        raise SystemExit("请用 USE_LLM_AGENT=1 运行；离线回退模式不能用于 S3。")
    tasks = json.loads((ROOT / "data" / "tasks.json").read_text(encoding="utf-8"))
    payload = {
        "task": "S3 prompt template comparison",
        "model": os.environ.get("OPENAI_MODEL", "qwen2.5:7b-instruct"),
        "temperature": 0,
        "n": len(tasks),
        "experiments": [run_template(tasks, template) for template in PROMPT_TEMPLATES],
    }
    output = ROOT / "eval" / "prompt_ablation.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()