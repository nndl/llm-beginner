"""生成任务五的评测任务集（10 题），并打印模型部署提示。"""
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent


TASKS = [
    {"id": 1, "task": "计算 (123 + 456) * 789 的结果，并告诉我结果的位数。",
     "expected_tools": ["calculator"]},
    {"id": 2, "task": "用 Python 计算 100 以内所有质数的和。",
     "expected_tools": ["python_sandbox"]},
    {"id": 3, "task": "在当前仓库下找出所有 .md 文件，统计总数。",
     "expected_tools": ["file_search"]},
    {"id": 4, "task": "查一下维基百科里「图灵机」的发明者是谁。",
     "expected_tools": ["wiki"]},
    {"id": 5, "task": "查维基百科 Geoffrey Hinton 的出生年份，并计算他到 2026 年是多少岁。",
     "expected_tools": ["wiki", "calculator"]},
    {"id": 6, "task": "用 Python 写一个函数判断回文，并测试 'level' 和 'world' 两个词。",
     "expected_tools": ["python_sandbox"]},
    {"id": 7, "task": "查找仓库下所有包含 'TODO' 字样的文件路径。",
     "expected_tools": ["file_search"]},
    {"id": 8, "task": "计算 sqrt(2026) 的小数点后 6 位结果。",
     "expected_tools": ["calculator", "python_sandbox"]},
    {"id": 9, "task": "查维基百科 Transformer (机器学习模型) 的发表年份，"
                      "并算到今年（2026）过了多少年。",
     "expected_tools": ["wiki", "calculator"]},
    {"id": 10, "task": "在仓库下找 README.md，告诉我它的第一段写了什么。",
     "expected_tools": ["file_search"]},
]


def main():
    out = DATA_DIR / "tasks.json"
    out.write_text(json.dumps(TASKS, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已生成评测任务集：{out.relative_to(DATA_DIR.parent)}（{len(TASKS)} 题）")

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
