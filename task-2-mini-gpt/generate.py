"""用训练好的 MiniGPT 生成文本样例，对比不同采样策略（greedy / top-k / top-p / temperature）。

用法：
    python generate.py --prompt "床前明月光"
"""
import argparse  # 接收 prompt、生成长度和随机种子
import json  # 将策略对比样例导出为 JSON
from pathlib import Path  # 基于脚本位置定位 checkpoint 与输出目录

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent  # 固定为 task-2 根目录，不依赖终端当前路径


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="床前明月光")
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpt" / "best.pt",
                        help="要生成的模型 checkpoint；其同目录必须有 tokenizer.json")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "generation_samples.json",
                        help="生成样例 JSON 的写入位置")
    args = parser.parse_args()  # 解析并校验用户传入的生成配置

    import torch  # 仅在实际运行脚本时导入 torch，保持模块导入轻量
    torch.manual_seed(args.seed)  # 固定随机采样序列，方便公平比较不同策略

    model, tokenizer = load_for_eval(str(args.checkpoint))  # 加载训练出的模型和对应 tokenizer
    prompt_ids = tokenizer.encode(args.prompt)  # 把用户 prompt 转为模型输入 token id

    strategies = [
        {"name": "greedy", "temperature": 0.0, "top_k": None, "top_p": None},
        {"name": "top_k=3, temperature=0.3", "temperature": 0.3, "top_k": 3, "top_p": None},
        {"name": "top_p=0.5, temperature=0.3", "temperature": 0.3, "top_k": None, "top_p": 0.5},
        {"name": "temperature=0.5 (适度随机)", "temperature": 0.5, "top_k": None, "top_p": None},
        {"name": "temperature=0.3 (更保守)", "temperature": 0.3, "top_k": None, "top_p": None},
    ]

    results = []  # 汇总每种策略的文本，最终一次性写入 JSON
    print(f"Prompt: {args.prompt}\n")
    for strat in strategies:
        torch.manual_seed(args.seed)  # 固定种子，方便复现对比
        gen_ids = model.generate(  # 模型内部使用 KV cache 做逐 token 自回归生成
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=strat["temperature"],
            top_k=strat["top_k"],
            top_p=strat["top_p"],
        )
        text = tokenizer.decode(gen_ids)  # 将 prompt 加新 token 的完整 id 序列还原为文本
        print(f"--- {strat['name']} ---")
        print(text)
        print()
        results.append({"strategy": strat["name"], "text": text})  # 保留策略名称和对应输出，便于报告引用

    out_path = args.output  # 默认写任务要求位置；候选模型可写入独立结果文件
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 若输出目录不存在则创建，已存在时不报错
    out_path.write_text(json.dumps({"prompt": args.prompt, "samples": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成样例已保存到 {out_path}")


if __name__ == "__main__":  # 仅直接运行该文件时启动 CLI 主流程，作为模块导入时不执行
    main()  # 进入生成流程
