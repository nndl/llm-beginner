"""任务六自检：MCP server 工具列表 + Skill loader 元数据 + SWE-bench Lite 抽样跑通。"""
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_mcp_server_lists_tools():
    """启动 MCP server，通过 stdio 协议握手并列出工具。

    简化版：直接 import 学生模块，调用一个 list_tools() 暴露接口。
    """
    try:
        from src.mcp_server import list_tools  # 学生应在 mcp_server.py 顶层导出
    except ImportError as e:
        return {"test": "mcp_server_lists_tools", "pass": False,
                "error": f"src/mcp_server.py 应导出 list_tools()：{e}"}
    tools = list_tools()
    return {"test": "mcp_server_lists_tools",
            "pass": isinstance(tools, list) and len(tools) >= 5,
            "tools": [t.get("name") if isinstance(t, dict) else str(t)
                      for t in tools]}


def test_skill_loader_metadata():
    from src.skill_loader import SkillLoader
    loader = SkillLoader(str(ROOT / "src" / "skills"))
    skills = loader.list_skills()
    if not skills:
        return {"test": "skill_loader_metadata", "pass": False,
                "error": "没扫描到任何 Skill"}
    missing = [s for s in skills
               if not (isinstance(s, dict) and s.get("name") and s.get("description"))]
    return {"test": "skill_loader_metadata",
            "pass": len(skills) >= 2 and not missing,
            "count": len(skills),
            "missing_meta": [s.get("name", "?") for s in missing]}


def test_toy_repo_patch():
    from src.agent import CodingAgent
    toy_repo = ROOT / "data" / "toy-repo"
    issue_path = toy_repo / "ISSUE.md"
    if not issue_path.exists():
        return {"test": "toy_repo_patch", "pass": None,
                "skip": "data/toy-repo 不存在；先跑 data/download.py"}

    agent = CodingAgent()
    issue = issue_path.read_text(encoding="utf-8")
    trace = agent.run(repo_path=str(toy_repo), issue=issue)
    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=toy_repo,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return {
        "test": "toy_repo_patch",
        "pass": test_run.returncode == 0,
        "tests_passed": test_run.returncode == 0,
        "pytest_output": (test_run.stdout + test_run.stderr)[-800:],
        "trace_has_patch": bool(isinstance(trace, dict) and trace.get("patch")),
    }


def test_swebench_lite_sample():
    from src.agent import CodingAgent
    sample_path = ROOT / "data" / "swebench-lite-sample.parquet"
    if not sample_path.exists():
        return {"test": "swebench_lite_sample", "pass": None,
                "skip": "data/swebench-lite-sample.parquet 不存在；需要时跑 data/download.py --with-swebench"}
    try:
        import pandas as pd
        df = pd.read_parquet(sample_path)
    except Exception as e:
        return {"test": "swebench_lite_sample", "pass": False, "error": str(e)}

    agent = CodingAgent()
    passed = 0
    results = []
    attempted = 0
    for _, row in df.iterrows():
        repo_path = ROOT / "data" / "repos" / row["repo"].split("/")[-1]
        if not repo_path.exists():
            results.append({"id": row["instance_id"],
                            "skip": f"repo 未 clone：{repo_path.relative_to(ROOT)}"})
            continue
        attempted += 1
        try:
            trace = agent.run(repo_path=str(repo_path), issue=row["problem_statement"])
            ok = bool(trace.get("tests_passed"))
            passed += int(ok)
            results.append({"id": row["instance_id"], "tests_passed": ok})
        except Exception as e:
            results.append({"id": row["instance_id"], "error": str(e)[:120]})
    if attempted == 0:
        return {"test": "swebench_lite_sample", "pass": None,
                "skip": "已下载 SWE-bench 元数据，但 data/repos/ 下没有对应本地仓库",
                "details": results}

    return {"test": "swebench_lite_sample",
            "pass": passed >= 1, "passed": passed, "attempted": attempted, "n": len(df),
            "details": results}


def main():
    results = []
    for fn in [test_mcp_server_lists_tools, test_skill_loader_metadata,
               test_toy_repo_patch, test_swebench_lite_sample]:
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
