"""任务五自检：工具单元测试 + 多工具任务成功率 + 错误恢复。"""
import json
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def normalize_answer(text):
    text = str(text).lower()
    text = text.replace(",", "").replace("，", "")
    return re.sub(r"\s+", "", text)


def answer_matches(answer, expected_keywords):
    norm_answer = normalize_answer(answer)
    for expected in expected_keywords:
        if isinstance(expected, list):
            if not any(normalize_answer(keyword) in norm_answer
                       for keyword in expected):
                return False
        elif normalize_answer(expected) not in norm_answer:
            return False
    return True


def extract_used_tools(trace):
    used = []
    for step in trace.get("steps", []) if isinstance(trace, dict) else []:
        if not isinstance(step, dict):
            continue
        name = step.get("tool") or step.get("tool_name") or step.get("action")
        if isinstance(name, str) and name:
            used.append(name)
    return used


def test_tools_individual():
    """每个工具单元测试。"""
    results = {}
    try:
        from src.tools import calculator, python_sandbox, file_search, wiki
    except ImportError as e:
        return {"test": "tools_individual", "pass": False,
                "error": f"工具导入失败：{e}"}

    # calculator
    try:
        out = calculator.run({"expression": "2 + 3 * 4"})
        results["calculator"] = "14" in str(out)
    except Exception as e:
        results["calculator"] = f"error: {e}"

    # python_sandbox
    try:
        out = python_sandbox.run({"code": "print(sum(range(10)))"})
        results["python_sandbox"] = "45" in str(out)
    except Exception as e:
        results["python_sandbox"] = f"error: {e}"

    # file_search
    try:
        out = file_search.run({"pattern": "README.md", "dir": str(ROOT)})
        results["file_search"] = "README.md" in str(out)
    except Exception as e:
        results["file_search"] = f"error: {e}"

    # wiki
    try:
        out = wiki.run({"query": "Alan Turing"})
        results["wiki"] = len(str(out)) > 50  # 至少有非空响应
    except Exception as e:
        results["wiki"] = f"error: {e}"

    all_pass = all(v is True for v in results.values())
    return {"test": "tools_individual", "pass": all_pass, "results": results}


def test_multi_tool_success_rate():
    from src.agent import ReActAgent
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        return {"test": "multi_tool_success_rate", "pass": None,
                "skip": "data/tasks.json 不存在；跑 data/download.py 生成"}
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    agent = ReActAgent()
    success = 0
    details = []
    for t in tasks:
        try:
            trace = agent.run(t["task"])
            final_answer = trace.get("final_answer", "") if isinstance(trace, dict) else ""
            expected = t.get("expected_answer_contains", [])
            ok = answer_matches(final_answer, expected)
            success += int(ok)
            used_tools = extract_used_tools(trace)
            expected_tools = t.get("expected_tools", [])
            details.append({
                "id": t["id"],
                "success": ok,
                "final_answer_preview": str(final_answer)[:120],
                "expected_answer_contains": expected,
                "expected_tools": expected_tools,
                "used_tools": used_tools,
                "used_expected_tools": all(
                    any(expected_tool in used for used in used_tools)
                    for expected_tool in expected_tools
                ) if used_tools else None,
            })
        except Exception as e:
            details.append({"id": t["id"], "success": False, "error": str(e)})
    rate = success / len(tasks)
    return {"test": "multi_tool_success_rate", "pass": rate > 0.6,
            "rate": round(rate, 3), "n": len(tasks), "details": details}


def test_error_recovery():
    """注入错误工具响应（暂为可选；学生需要实现 src/agent.py 的 inject_error 钩子）。"""
    return {"test": "error_recovery", "pass": None,
            "skip": "需要学生实现 inject_error 测试钩子；可选实验"}


def main():
    results = []
    for fn in [test_tools_individual, test_multi_tool_success_rate, test_error_recovery]:
        try:
            r = fn()
        except Exception as e:
            r = {"test": fn.__name__.replace("test_", ""),
                 "pass": False, "error": str(e),
                 "trace": traceback.format_exc().splitlines()[-3:]}
        results.append(r)
        tag = "[通过]" if r.get("pass") is True else \
              ("[跳过]" if r.get("pass") is None else "[失败]")
        print(f"{tag} {r['test']}: {json.dumps(r, ensure_ascii=False)}")
    out = ROOT / "eval" / "result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
