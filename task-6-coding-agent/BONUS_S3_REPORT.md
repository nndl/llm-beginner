# 加分任务 S3：纯 Prompt 与 Skills 对照实验

## 结论

**这是任务六的加分任务 S3 交付。** 在提供的 `data/toy-repo` 单题受控微基准上，纯 prompt 和按需加载 Skills 两个条件都成功修复了 `calculator.add`，并通过了全部 3 个 pytest 用例。因此，本次实验的成功率为纯 prompt `1/1 (100%)`、Skills `1/1 (100%)`。Skills 没有改变该简单任务的成功率，但在 trace 中提供了可审计的任务约束。

该结论仅适用于本仓库的单题 toy-repo，不能外推为通用编码任务的成功率比较。要获得有统计意义的结论，需要用更多独立 issue、固定模型端点和重复运行次数扩展该实验。

## 实验设计

| 项目 | 纯 prompt 对照组 | Skills 实验组 |
|---|---|---|
| Agent | `CodingAgent(use_skills=False)` | `CodingAgent(use_skills=True)` |
| 模型策略 | 默认离线确定性 planner | 默认离线确定性 planner |
| 任务 | 修复 `calculator.add`，不修改测试 | 相同 |
| 初始状态 | 每次从 `calculator.py.orig` 恢复 | 每次从 `calculator.py.orig` 恢复 |
| 成功标准 | `python -m pytest -q` 退出码为 0 | 相同 |

Skills 组按 front-matter 描述进行匹配，只在命中后加载正文。此 issue 命中 `test-runner`；纯 prompt 组不匹配、不加载任何 Skill。除 `use_skills` 外，两组使用相同的 agent、repo、issue 和测试命令。

## 结果

| 条件 | 已加载 Skills | pytest | 成功 |
|---|---|---|---|
| 纯 prompt | 无 | `3 passed` | 是 |
| Skills | `test-runner` | `3 passed` | 是 |

两个条件均只将 `return a - b` 改为 `return a + b`，并在 Trace 中记录 patch 和测试结果。Skills 组的 trace 首步还记录了匹配及加载的 Skill 正文，因而能明确审计“最小改动、不得修改测试、以 pytest 通过为完成信号”等约束；纯 prompt 对照组的对应列表为空。

## 复现命令

在 `task-6-coding-agent` 目录执行：

```bash
python - <<'PY'
import shutil
from pathlib import Path

from src.agent import CodingAgent

repo = Path("data/toy-repo")
issue = (repo / "ISSUE.md").read_text(encoding="utf-8")
for use_skills in (False, True):
    shutil.copy(repo / "calculator.py.orig", repo / "calculator.py")
    trace = CodingAgent(use_skills=use_skills).run(str(repo), issue)
    print({
        "use_skills": use_skills,
        "tests_passed": trace["tests_passed"],
        "skills": trace.get("skills", []),
    })
PY
python eval/run.py
```

`eval/run.py` 同时验证必做项 M1-M3；其现有结果记录了 toy-repo 的 `3 passed`。S4 仍因未下载 SWE-bench Lite 元数据而跳过，和本 S3 加分结论无关。