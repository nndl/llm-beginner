"""生成任务五的评测任务集（10 题）、本地检索夹具，并打印模型部署提示。"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
FIXTURE_DIR = DATA_DIR / "agent-fixtures"


def write_fixtures():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "README.md").write_text(
        "这是任务五的本地文件检索测试文件。\n\n"
        "第二段用于确认工具能够读取文件内容，而不是只匹配文件名。\n",
        encoding="utf-8",
    )
    (FIXTURE_DIR / "todo_note.md").write_text(
        "# TODO 测试\n\n"
        "- TODO: 用于检查 file_search 是否能返回包含 TODO 的文件路径。\n",
        encoding="utf-8",
    )


def build_tasks():
    return [
        {"id": 1, "task": "计算 (123 + 456) * 789 的结果，并告诉我结果的位数。",
         "expected_tools": ["calculator"],
         "expected_answer_contains": ["456831", "6"]},
        {"id": 2, "task": "用 Python 计算 100 以内所有质数的和。",
         "expected_tools": ["python_sandbox"],
         "expected_answer_contains": ["1060"]},
        {"id": 3, "task": "在 data/agent-fixtures 目录下找出所有 .md 文件，统计总数。",
         "expected_tools": ["file_search"],
         "expected_answer_contains": [["2", "两"]]},
        {"id": 4, "task": "查一下维基百科里「图灵机」的发明者是谁。",
         "expected_tools": ["wiki"],
         "expected_answer_contains": [["Turing", "图灵"]]},
        {"id": 5, "task": "查维基百科 Geoffrey Hinton 的出生年份，并计算他到 2026 年是多少岁。",
         "expected_tools": ["wiki", "calculator"],
         "expected_answer_contains": ["1947", "79"]},
        {"id": 6, "task": "用 Python 写一个函数判断回文，并测试 'level' 和 'world' 两个词。",
         "expected_tools": ["python_sandbox"],
         "expected_answer_contains": ["level", "True", "world", "False"]},
        {"id": 7, "task": "查找 data/agent-fixtures 下所有包含 'TODO' 字样的文件路径。",
         "expected_tools": ["file_search"],
         "expected_answer_contains": ["todo_note.md"]},
        {"id": 8, "task": "计算 sqrt(2026) 的小数点后 6 位结果。",
         "expected_tools": ["calculator", "python_sandbox"],
         "expected_answer_contains": [["45.011110", "45.01111"]]},
        {"id": 9, "task": "查维基百科 Transformer (机器学习模型) 的发表年份，"
                          "并算到 2026 年过了多少年。",
         "expected_tools": ["wiki", "calculator"],
         "expected_answer_contains": ["2017", "9"]},
        {"id": 10, "task": "读取 data/agent-fixtures/README.md，告诉我它的第一段写了什么。",
         "expected_tools": ["file_search"],
         "expected_answer_contains": ["任务五", "本地文件检索测试文件"]},
    ]


def main():
    write_fixtures()
    tasks = build_tasks()
    out = DATA_DIR / "tasks.json"
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已生成评测任务集：{out.relative_to(DATA_DIR.parent)}（{len(tasks)} 题）")
    print(f"已生成本地检索夹具：{FIXTURE_DIR.relative_to(DATA_DIR.parent)}")

    print("\n--- 模型部署提示 ---")
    print("方式一：Ollama（最简单）")
    print("  ollama pull qwen2.5:7b-instruct")
    print("  ollama serve")
    print("  # 客户端通过 OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama 访问")
    print("\n方式二：vLLM（推理快）")
    print("  pip install vllm")
    print("  python -m vllm.entrypoints.openai.api_server "
          "--model Qwen/Qwen2.5-7B-Instruct --quantization awq")
    print("\n方式三：llama.cpp + GGUF（最省显存）")
    print("  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF "
          "qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ./models")
    print("  ./llama-server -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf --port 8080")


if __name__ == "__main__":
    main()
