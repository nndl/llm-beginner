"""在 SFT LoRA adapter 基础上执行手写 Direct Preference Optimization。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import build_labels, format_messages
from src.lora import inject_lora, load_lora, save_lora

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B"


def load_pairs(path: Path, limit: int | None) -> list[tuple[str, str, str]]:
    # 用于保存所有有效的偏好数据
    # 每条数据的格式为：(prompt, chosen, rejected)
    pairs = []
    with path.open(encoding="utf-8") as handle:
    # 以 UTF-8 编码打开 JSONL 文件
    # JSONL 文件中每一行通常对应一条 JSON 数据
        for line in handle:   # 逐行读取文件，避免一次性将整个文件加载到内存
            if not line.strip():    # 跳过空行
                continue
            # 将当前行的 JSON 字符串解析成 Python 字典
            row = json.loads(line)
            # 获取用户的问题/提示词
            # 优先使用 "prompt" 字段
            # 如果不存在，则尝试使用 "instruction"
            # 如果仍不存在，则尝试使用 "question"
            prompt = row.get("prompt") or row.get("instruction") or row.get("question")
            # 获取模型表现较好的回答（chosen）
            # 优先使用 "chosen"，不存在时尝试使用 "response_chosen"
            chosen = row.get("chosen") or row.get("response_chosen")
            # 获取模型表现较差的回答（rejected）
            # 优先使用 "rejected"，不存在时尝试使用 "response_rejected"
            rejected = row.get("rejected") or row.get("response_rejected")
            # 只有 prompt、chosen、rejected 三个字段都存在且非空时，
            # 才认为这是一条有效的偏好数据
            if prompt and chosen and rejected:
                # 将三个字段统一转换成字符串，
                # 并按照 (prompt, chosen, rejected) 的格式保存
                pairs.append((str(prompt), str(chosen), str(rejected)))
            # 如果设置了 limit，并且已经收集到指定数量的有效数据，
            # 就提前结束读取，避免继续处理后面的数据
            if limit and len(pairs) >= limit:
                break
    # 如果整个文件都没有读取到有效的偏好数据，则抛出异常
    if not pairs:
        raise ValueError(f"没有从 {path} 读取到 prompt/chosen/rejected 偏好对")
    return pairs        # 返回所有有效的偏好数据


def sequence_logprob(model, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # 使用模型进行前向计算
    # ids.unsqueeze(0)：给原本的 [seq_len] 增加 batch 维度，变成 [1, seq_len]
    # model(...).logits 的形状为 [1, seq_len, vocab_size]
    # [:, :-1]：去掉最后一个位置，因为最后一个 token 没有下一个 token 可以预测
    # 最终 logits 形状：[1, seq_len-1, vocab_size]
    logits = model(input_ids=ids.unsqueeze(0)).logits[:, :-1]
    # labels 原本形状为 [seq_len]
    # labels[1:]：去掉第一个 token，使标签与前面的 logits 对齐
    # unsqueeze(0)：增加 batch 维度
    # 最终 targets 形状：[1, seq_len-1]
    targets = labels[1:].unsqueeze(0)
    # 判断哪些 token 是有效的训练目标
    # 在 Hugging Face 的训练中，label=-100 通常表示该位置不参与 loss 计算
    # valid 形状：[1, seq_len-1]
    # True  表示这个 token 需要计算概率
    # False 表示这个 token 被 mask，不参与计算
    valid = targets != -100
    # 将 logits 转换为 log probability
    # F.log_softmax(logits, dim=-1)：
    #   对 vocabulary 维度进行 softmax，并取 log
    #   得到每个位置上所有 token 的对数概率
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, targets.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
    # 只保留有效 token 的 log probability
    # valid 为 False 的位置乘以 0，不参与最终求和
    #
    # .sum()：
    #   将整个序列中所有有效 token 的 log probability 相加
    #
    # 返回：
    #   一个标量 Tensor，表示整个回答序列的 log probability
    return (token_logps * valid).sum()

# 原始文本 → 模型 token → 训练标签之间的桥梁
def encode_pair(tokenizer, prompt: str, answer: str, max_length: int):
    # 构造对话消息
    # 第一条是用户输入（prompt）
    # 第二条是模型回答（answer）
    #
    # 最终得到类似：
    # [
    #     {"role": "user", "content": "用户的问题"},
    #     {"role": "assistant", "content": "模型的回答"}
    # ]
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    # 将对话消息按照指定的 Chat Template 格式转换成文本，
    # 然后使用 tokenizer 将文本转换成 token ID
    #
    # format_messages(messages)：
    #   将 messages 转换成模型实际使用的对话格式
    #
    # truncation=True：
    #   如果序列长度超过 max_length，则截断
    #
    # max_length=max_length：
    #   限制最大 token 长度
    #
    # return_tensors="pt"：
    #   返回 PyTorch Tensor
    #
    # tokenizer(...).input_ids：
    #   获取 token ID
    #
    # [0]：
    #   去掉 batch 维度
    #   从 [1, seq_len] 变成 [seq_len]
    ids = tokenizer(format_messages(messages), truncation=True, max_length=max_length,
                    return_tensors="pt").input_ids[0]
    # 根据 token 序列和原始对话消息构造 labels
    #
    # build_labels() 通常会：
    #   1. 将 prompt 部分设置为 -100
    #   2. 保留 assistant answer 部分的真实 token ID
    #
    # 这样后面计算 sequence_logprob() 时，
    # 只计算模型对 assistant 回答部分的 log probability，
    # 而不计算用户 prompt 部分。
    return ids, build_labels(ids, messages)


def main() -> None:
    parser = argparse.ArgumentParser()
    # DPO 偏好数据集路径
    # 例如：--data data/dpo_smoke.jsonl
    parser.add_argument("--data", type=Path, required=True)
    # 基础模型路径
    # 如果命令行没有指定 --model，则使用 DEFAULT_MODEL
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    # SFT 阶段训练得到的 LoRA adapter 保存目录
    parser.add_argument("--sft_dir", type=Path, default=ROOT / "ckpt" / "sft")
    # DPO 训练完成后 LoRA adapter 的保存目录
    parser.add_argument("--output_dir", type=Path, default=ROOT / "ckpt" / "dpo")
    # 最多使用多少条偏好数据进行训练
    parser.add_argument("--max_samples", type=int, default=1000)
    # 输入序列最大 token 长度
    # 超过该长度的序列会被 tokenizer 截断
    parser.add_argument("--max_length", type=int, default=512)
    # DPO 训练轮数
    parser.add_argument("--epochs", type=int, default=1)
    # AdamW 优化器的学习率
    parser.add_argument("--lr", type=float, default=1e-5)
    # DPO 中的 beta 参数
    # 用于控制模型偏好差异的强弱
    parser.add_argument("--beta", type=float, default=0.1)
    # LoRA 的低秩矩阵维度 r
    parser.add_argument("--rank", type=int, default=8)
    # LoRA 的缩放系数 alpha
    parser.add_argument("--alpha", type=float, default=16.0)
    args = parser.parse_args()
    # 检查 SFT 阶段训练得到的 LoRA adapter 是否存在
    # DPO 训练是在 SFT 模型的基础上继续进行，
    # 因此必须先有 SFT adapter
    if not (args.sft_dir / "adapter_model.pt").exists():
        raise FileNotFoundError(f"缺少 SFT adapter：{args.sft_dir / 'adapter_model.pt'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 从基础模型目录加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 指定需要注入 LoRA 的目标模块
    # q_proj：Attention 中 Query 投影层
    # v_proj：Attention 中 Value 投影层
    targets = ["q_proj", "v_proj"]
    # 加载基础语言模型，并移动到指定设备
    policy = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    # 给 policy 模型注入 LoRA
    #
    # r=args.rank：
    #   LoRA 低秩矩阵的秩
    #
    # alpha=args.alpha：
    #   LoRA 缩放系数
    #
    # 注入之后，policy 中指定的 q_proj 和 v_proj
    # 会增加可训练的 LoRA 参数
    inject_lora(policy, targets, r=args.rank, alpha=args.alpha)
    # 加载之前 SFT 阶段训练好的 LoRA 参数
    #
    # 此时 policy 不再是单纯的基础模型，
    # 而是：
    #
    # 基础模型 + SFT LoRA
    #
    # DPO 将在这个 SFT 模型基础上继续优化
    load_lora(policy, args.sft_dir)
    # 深拷贝当前 SFT 模型作为 reference model
    #
    # 此时：
    # policy    = 用于 DPO 训练的模型
    # reference = 固定不训练的参考模型
    #
    # 两者初始化时完全相同，都是 SFT 后的模型
    reference = copy.deepcopy(policy).eval()
    # 冻结 reference model 的所有参数
    # reference 只用于计算参考模型的 log probability，
    # 不参与反向传播和参数更新
    for parameter in reference.parameters():
        parameter.requires_grad = False
    # 创建 AdamW 优化器
    #
    # 这里只将 requires_grad=True 的参数交给优化器，
    # 因此主要更新的是 policy 中的 LoRA 参数，
    # 而不是整个基础模型参数
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    # 保存训练过程中每一步的 reward margin 和 loss
    # 后面会写入 train_log.json，用于分析训练过程
    margins, losses = [], []
    # 开始进行 DPO 训练
    for _ in range(args.epochs):
        # 从偏好数据集中读取：
        # prompt    = 用户问题
        # chosen   = 更好的回答
        # rejected = 较差的回答
        #
        # args.max_samples 控制最多读取多少条数据
        for prompt, chosen, rejected in load_pairs(args.data, args.max_samples):
            # 将 prompt + chosen 编码成 token ID 和 labels
            #
            # chosen_ids：
            #   chosen 完整序列的 token ID
            #
            # chosen_labels：
            #   用于计算 chosen 回答 log probability 的 labels
            chosen_ids, chosen_labels = encode_pair(tokenizer, prompt, chosen, args.max_length)
            # 将 prompt + rejected 编码成 token ID 和 labels
            rejected_ids, rejected_labels = encode_pair(tokenizer, prompt, rejected, args.max_length)
            # 将 chosen 数据移动到 GPU/CPU
            chosen_ids, chosen_labels = chosen_ids.to(device), chosen_labels.to(device)
            # 将 rejected 数据移动到 GPU/CPU
            rejected_ids, rejected_labels = rejected_ids.to(device), rejected_labels.to(device)
            # 计算 policy 模型对 chosen 回答的 log probability
            #
            # 数值越大，说明 policy 越倾向于 chosen 回答
            policy_chosen = sequence_logprob(policy, chosen_ids, chosen_labels)
            # 计算 policy 模型对 rejected 回答的 log probability
            #
            # DPO 的目标是让：
            # policy_chosen > policy_rejected
            policy_rejected = sequence_logprob(policy, rejected_ids, rejected_labels)
            with torch.no_grad():
                # 计算 reference 对 chosen 回答的 log probability
                ref_chosen = sequence_logprob(reference, chosen_ids, chosen_labels)
                # 计算 reference 对 rejected 回答的 log probability
                ref_rejected = sequence_logprob(reference, rejected_ids, rejected_labels)
             # 计算 DPO 中最核心的相对偏好差异
            #
            # 第一部分：
            # policy_chosen - policy_rejected
            #
            # 表示当前 policy 对 chosen 相对于 rejected
            # 的偏好程度
            #
            # 第二部分：
            # ref_chosen - ref_rejected
            #
            # 表示 reference 模型本身对 chosen 相对于 rejected
            # 的偏好程度
            #
            # 两者相减：
            #
            # margin =
            #   当前 policy 的偏好
            #   -
            #   reference 的偏好
            #
            # DPO 希望这个 margin 越来越大
            margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            # DPO 损失函数
            #
            # 标准形式：
            #
            # loss = -log(sigmoid(beta * margin))
            #
            # beta 控制偏好优化的强度
            #
            # 当 margin 很大：
            #   sigmoid(beta * margin) → 1
            #   loss → 0
            #
            # 当 margin 很小甚至为负：
            #   loss 会变大
            loss = -F.logsigmoid(args.beta * margin)
            optimizer.zero_grad()       # 清空上一轮反向传播产生的梯度
            # 根据 DPO loss 进行反向传播
            # 计算 policy 中可训练参数的梯度
            loss.backward()
            # 梯度裁剪
            #
            # 将梯度范数限制在 1.0 以内，
            # 防止训练过程中出现梯度爆炸
            torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0)
            # 根据计算得到的梯度更新 policy 参数
            # 这里主要更新 LoRA 参数
            optimizer.step()
            margins.append(float(margin.detach().item()))   # 保存当前训练步骤的 margin
            losses.append(float(loss.detach().item()))      # 保存当前训练步骤的 loss
    # DPO 训练完成后，将 policy 中训练好的 LoRA 参数保存下来
    save_lora(policy, args.output_dir, r=args.rank, alpha=args.alpha, target_modules=targets)
    # 将训练过程中的 loss 和 reward margin 保存成 JSON 文件
    (args.output_dir / "train_log.json").write_text(json.dumps({"losses": losses, "reward_margins": margins}, indent=2), encoding="utf-8")
    print(f"DPO adapter 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
