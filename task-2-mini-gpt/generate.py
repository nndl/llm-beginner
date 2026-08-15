"""用训练好的 MiniGPT 生成文本样例，对比不同采样策略（greedy / top-k / top-p / temperature）。

用法：
    python generate.py --prompt "床前明月光"
"""
import argparse  # 接收 prompt、生成长度和随机种子
import json  # 将策略对比样例导出为 JSON
import re  # 将生成文本切分为词与标点，计算重复 n-gram 指标
from pathlib import Path  # 基于脚本位置定位 checkpoint 与输出目录

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent  # 固定为 task-2 根目录，不依赖终端当前路径


def repetition_rate(text: str, n: int = 3) -> float:
    """计算重复 n-gram 占比；越低表示局部重复越少。"""
    tokens = re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())  # 将英文单词与标点分别作为比较单元
    ngrams = [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]  # 枚举连续 n-gram
    if not ngrams:  # 文本不足 n 个 token 时没有可统计的 n-gram
        return 0.0
    return 1 - len(set(ngrams)) / len(ngrams)  # 重复次数占全部 n-gram 的比例


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
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

    # 在最终 TinyStories checkpoint 上、固定 seed=42 生成 128 token 的扫描中，
    # top-k=5, T=0.5 的三元组重复率为 0.0612，是实际改变候选集合的 top-k
    # 配置中最低；top-p=0.6, T=0.5 在 top-p 配置中为 0.0816，故选作对照。
    # greedy 保留为确定性质量基线；无截断温度采样用于展示仅调温度的效果。
    strategies = [
        {"name": "greedy", "temperature": 0.0, "top_k": None, "top_p": None},
        {"name": "temperature=0.5（推荐随机采样）", "temperature": 0.5, "top_k": None, "top_p": None},
        {"name": "top_k=5, temperature=0.5", "temperature": 0.5, "top_k": 5, "top_p": None},
        {"name": "top_p=0.6, temperature=0.5", "temperature": 0.5, "top_k": None, "top_p": 0.6},
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
        print(f"重复 3-gram 比例: {repetition_rate(text):.4f}")
        print()
        results.append({
            "strategy": strat["name"],
            "temperature": strat["temperature"],
            "top_k": strat["top_k"],
            "top_p": strat["top_p"],
            "repetition_3gram": round(repetition_rate(text), 4),
            "text": text,
        })  # 保存策略参数、重复指标和文本，便于报告公平比较

    out_path = args.output  # 默认写任务要求位置；候选模型可写入独立结果文件
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 若输出目录不存在则创建，已存在时不报错
    out_path.write_text(json.dumps({
                                    "prompt": args.prompt,
                                    "seed": args.seed,
                                    "max_new_tokens": args.max_new_tokens,
                                    "repetition_metric": "repeated 3-grams / all 3-grams; lower is better",
                                    "samples": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成样例已保存到 {out_path}")


if __name__ == "__main__":  # 仅直接运行该文件时启动 CLI 主流程，作为模块导入时不执行
    main()  # 进入生成流程
