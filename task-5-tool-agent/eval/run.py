"""任务五自检：工具单元测试 + 多工具任务成功率 + 错误恢复。"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # from src.* —— 学生实现
sys.path.insert(0, str(ROOT.parent))    # from _eval_harness —— 共用运行壳

from _eval_harness import run_tests

SUCCESS_RATE_PASS = 0.6   # 多工具任务成功率通过线


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

    # (工具名, 模块, 调用参数, 预期子串, 是否依赖网络)：
    # expected=None 表示只检查响应长度 > 50（如 wiki 网络回包）；
    # network=True 的工具若抛异常按“跳过”处理（多半是离线/被墙），不拖累其余工具判定。
    checks = [
        ("calculator",     calculator,     {"expression": "2 + 3 * 4"},                  "14",        False),
        ("python_sandbox", python_sandbox, {"code": "print(sum(range(10)))"},            "45",        False),
        ("file_search",    file_search,    {"pattern": "README.md", "dir": str(ROOT)},   "README.md", False),
        ("wiki",           wiki,           {"query": "Alan Turing"},                     None,        True),
    ]
    network_skipped = []
    for name, mod, args, expected, network in checks:
        try:
            out = str(mod.run(args))
            results[name] = (expected in out) if expected is not None else (len(out) > 50)
        except Exception as e:
            if network:
                results[name] = f"skip(网络不可用？): {e}"
                network_skipped.append(name)
            else:
                results[name] = f"error: {e}"

    gated = [v for k, v in results.items() if k not in network_skipped]
    all_pass = bool(gated) and all(v is True for v in gated)
    return {"test": "tools_individual", "pass": all_pass, "results": results,
            "network_skipped": network_skipped or None}


def test_multi_tool_success_rate():
    from src.agent import ReActAgent
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        return {"test": "multi_tool_success_rate", "pass": None,
                "skip": "data/tasks.json 不存在；跑 data/download.py 生成"}
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not tasks:
        return {"test": "multi_tool_success_rate", "pass": None,
                "skip": "data/tasks.json 为空；重跑 data/download.py 生成"}
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
    return {"test": "multi_tool_success_rate", "pass": rate > SUCCESS_RATE_PASS,
            "rate": round(rate, 3), "n": len(tasks), "details": details}


def test_error_recovery():
    """S4：注入一次工具故障后，验证 agent 能记录错误并完成任务。"""
    from src.agent import ReActAgent

    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        return {"test": "error_recovery", "pass": None,
                "skip": "data/tasks.json 不存在；跑 data/download.py 生成"}

    task_ids = {1, 5, 8}
    tasks = [task for task in json.loads(tasks_path.read_text(encoding="utf-8"))
             if task["id"] in task_ids]
    details = []
    recovered = 0
    for task in tasks:
        try:
            trace = ReActAgent(inject_error=True).run(task["task"])
            steps = trace.get("steps", [])
            error_index = next((index for index, step in enumerate(steps)
                                if "Injected one-time tool failure" in str(step.get("observation", ""))),
                               None)
            retried = error_index is not None and any(
                "Tool error" not in str(step.get("observation", ""))
                for step in steps[error_index + 1:]
            )
            answer_ok = answer_matches(trace.get("final_answer", ""),
                                       task["expected_answer_contains"])
            success = retried and answer_ok
            recovered += int(success)
            details.append({"id": task["id"], "success": success,
                            "answer_ok": answer_ok, "retried": retried,
                            "final_answer_preview": str(trace.get("final_answer", ""))[:120]})
        except Exception as error:
            details.append({"id": task["id"], "success": False, "error": str(error)})

    rate = recovered / len(tasks) if tasks else 0
    return {"test": "error_recovery", "pass": rate > 0.4,
            "rate": round(rate, 3), "n": len(tasks), "details": details}


if __name__ == "__main__":
    run_tests([test_tools_individual, test_multi_tool_success_rate,
               test_error_recovery], ROOT)
