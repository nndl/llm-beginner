"""在 Qwen2.5-0.5B 上以手写 LoRA 执行监督微调。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import build_labels, format_messages
from src.lora import inject_lora, save_lora

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B"

# 数据适配层（Data Adapter），把不同来源的数据统一成模型训练需要的 ChatML/OpenAI 格式
def messages_from_record(record: dict) -> list[dict] | None:
    """
    将不同格式的单条数据记录转换为统一的 messages 格式。

    支持的数据格式：
    1. MOSS 格式：chat
    2. OpenAI 格式：messages
    3. ShareGPT 格式：conversations
    4. Instruction 格式：instruction + output

    返回：
        list[dict]:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]

        如果无法解析，则返回 None
    """

    # ===============================
    # 1. 解析 MOSS 格式数据
    # ===============================

    # 从字典 record 中获取 "chat" 字段
    # MOSS 数据通常将多轮对话保存在 chat 字段中
    moss_chat = record.get("chat") # 从字典 record 中获取键名为 "chat" 的聊天记录，并赋值给变量 moss_chat
    # 判断 chat 是否为字典类型
    # MOSS 格式通常类似：
    # {
    #    "turn_0": {
    #        "Human": "你好",
    #        "MOSS": "你好，有什么帮助？"
    #    },
    #    "turn_1": {
    #        "Human": "...",
    #        "MOSS": "..."
    #    }
    # }
    if isinstance(moss_chat, dict):
        messages = []
        # 按 turn 编号排序，保证多轮对话顺序正确
        for turn_name in sorted(moss_chat, key=lambda name: int(name.rsplit("_", 1)[-1])):
            turn = moss_chat[turn_name] # 获取当前轮对话内容
            # 如果当前轮不是字典格式，跳过
            if not isinstance(turn, dict):
                continue
            # 获取用户输入 Human 字段
            # 去除 MOSS 特殊标记
            user = str(turn.get("Human", "")).replace("<|Human|>:", "").replace("<eoh>", "").strip()
            # 获取模型回复 MOSS 字段
            # 去除特殊 token
            assistant = str(turn.get("MOSS", "")).replace("<|MOSS|>:", "").replace("<eom>", "").strip()
            # 如果用户和模型回复都存在
            # 转换为 OpenAI messages 格式
            if user and assistant:
                messages.extend(({"role": "user", "content": user},
                                 {"role": "assistant", "content": assistant}))
        # 如果成功解析 MOSS 数据，则直接返回
        if messages:
            return messages

    # ===============================
    # 2. 解析 OpenAI / ShareGPT 格式
    # ===============================

    # 优先读取 messages 字段
    # 如果不存在，则读取 conversations 字段
    #
    # 常见格式：
    # [
    #   {
    #      "role": "user",
    #      "content": "问题"
    #   },
    #   {
    #      "role": "assistant",
    #      "content": "回答"
    #   }
    # ]
    raw = record.get("messages") or record.get("conversations")
    # 判断是否为列表格式
    if isinstance(raw, list):
        messages = []
        # 遍历每一轮对话
        for turn in raw:
            # 获取角色字段
            #
            # OpenAI:
            # role
            #
            # ShareGPT:
            # from
            role = turn.get("role", turn.get("from", "")) 
            # 不同数据集角色名称转换
            #
            # human -> user
            # gpt   -> assistant
            # bot   -> assistant
            role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role, role) 
            # 获取文本内容
            #
            # OpenAI:
            # content
            #
            # ShareGPT:
            # value
            content = turn.get("content", turn.get("value", ""))
            # 只保留合法角色
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": str(content)})
        # 必须包含 assistant 回复才认为是有效对话
        if any(message["role"] == "assistant" for message in messages):
            return messages
    # ===============================
    # 3. 解析 Instruction 格式
    # ===============================

    # 获取用户指令
    #
    # 常见字段：
    # instruction
    # prompt
    instruction = record.get("instruction") or record.get("prompt")
    # 获取模型回答
    #
    # 常见字段：
    # output
    # response
    # answer
    answer = record.get("output") or record.get("response") or record.get("answer")
    # 如果存在问答对
    if instruction and answer:
        # 转换成统一格式
        return [{"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": str(answer)}]
    return None # 三种格式均无法匹配


def load_sft_records(path: Path, limit: int | None) -> list[list[dict]]:
    """
    从 JSONL 文件中加载 SFT 训练数据。

    参数：
        path:
            SFT 数据文件路径，例如：
            SoulChatCorpus-sft-multi-Turn.json

        limit:
            最大读取样本数量。
            None 表示读取全部数据。

    返回：
        list[list[dict]]

        示例：
        [
            [
                {
                    "role": "user",
                    "content": "你好"
                },
                {
                    "role": "assistant",
                    "content": "你好，我是助手"
                }
            ],
            [
                {
                    "role": "user",
                    "content": "介绍一下Transformer"
                },
                {
                    "role": "assistant",
                    "content": "Transformer是一种..."
                }
            ]
        ]
    """
    # 保存所有转换后的对话数据
    # 外层 list 表示多个训练样本
    # 内层 list 表示一个样本中的多轮 messages
    records: list[list[dict]] = []
    # 打开 JSONL 文件
    #
    # JSONL 格式：
    # 每一行是一个独立 JSON 对象
    #
    # 示例：
    # {"chat":[...]}
    # {"messages":[...]}
    #
    # encoding="utf-8" 防止中文乱码
    with path.open(encoding="utf-8") as handle:
        for line in handle:  # 逐行读取数据
            if not line.strip():   # 跳过空行
                continue
            # 调用前面定义的数据格式转换函数
            #
            # 支持：
            # 1. MOSS chat
            # 2. OpenAI messages
            # 3. ShareGPT conversations
            # 4. instruction/output
            #
            # 返回：
            #
            # [
            #    {"role":"user","content":"xxx"},
            #    {"role":"assistant","content":"xxx"}
            # ]
            messages = messages_from_record(json.loads(line))
            # 如果成功解析
            # 保存该条训练样本
            if messages:
                records.append(messages)
            # 如果设置了读取数量限制
            #
            # 例如：
            # limit=1000
            #
            # 读取1000条有效样本后停止
            if limit and len(records) >= limit:
                break
    # 如果整个文件没有读取到有效数据
    # 说明数据格式错误或者路径错误
    if not records:
        raise ValueError(f"没有从 {path} 读取到可用 SFT 对话")
    return records  # 返回所有训练样本


def main() -> None:
    # ============================================
    # 1. 创建命令行参数解析器
    # ============================================
    parser = argparse.ArgumentParser()
    # SFT训练数据路径
    #
    # 示例：
    # python train.py --data ./SoulChat.jsonl
    #
    # 类型：
    # Path对象
    parser.add_argument("--data", type=Path, required=True, help="MOSS jsonl 或兼容格式 jsonl")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)  # 基础预训练模型路径
    parser.add_argument("--output_dir", type=Path, default=ROOT / "ckpt" / "sft") # LoRA权重保存目录
    # 最大训练样本数量
    #
    # 防止一次加载过多数据导致内存不足
    parser.add_argument("--max_samples", type=int, default=1000)
    # 最大token长度
    #
    # 超过长度会截断
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)   # 训练epoch数量
    parser.add_argument("--lr", type=float, default=2e-4)  # LoRA学习率
    # LoRA rank
    #
    # rank越大:
    # 参数量越多
    # 表达能力越强
    parser.add_argument("--rank", type=int, default=8)
    # LoRA alpha
    #
    # 缩放因子:
    # LoRA权重 = alpha/r * LoRA矩阵
    parser.add_argument("--alpha", type=float, default=16.0)   # 解析命令行参数
    args = parser.parse_args()
    # ============================================
    # 2. 设置训练设备
    # ============================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[警告] 正在 CPU 上训练 0.5B 模型，适合 smoke test，不适合完整训练。")
    # ============================================
    # 3. 加载Tokenizer
    # ============================================
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 如果模型没有pad token
    #
    # 使用eos token代替
    #
    # Qwen类模型通常需要处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ============================================
    # 4. 加载基础语言模型
    # ============================================
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    # ============================================
    # 5. 注入LoRA
    # ============================================
    # 指定需要插入LoRA的Transformer模块
    #
    # 对于Qwen/LLaMA:
    #
    # Attention:
    #     q_proj
    #     k_proj
    #     v_proj
    #     o_proj
    #
    # 这里仅训练:
    #     Query
    #     Value
    #
    targets = ["q_proj", "v_proj"]
    # 冻结原模型参数
    # 添加LoRA可训练矩阵
    inject_lora(model, targets, r=args.rank, alpha=args.alpha)
    # ============================================
    # 6. 创建优化器
    # ============================================
    # 只优化requires_grad=True参数
        #
        # 即:
        # LoRA参数
        # 
        # 基础模型被冻结
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    # ============================================
    # 7. 加载SFT训练数据
    # ============================================
    records = load_sft_records(args.data, args.max_samples)
    model.train()  # 设置训练模式
    losses = []  # 保存loss曲线
    # ============================================
    # 8. 开始训练
    # ============================================
    for epoch in range(args.epochs):
        # 遍历每条对话数据
        for step, messages in enumerate(records, start=1):
            # -------------------------------
            # 文本转token
            # -------------------------------
            ids = tokenizer(format_messages(messages), truncation=True, max_length=args.max_length,
                            return_tensors="pt").input_ids[0]
            # -------------------------------
            # 构造训练标签
            # -------------------------------
            #
            # SFT训练不是预测所有token
            #
            # 通常:
            #
            # user:
            #     -100
            #
            # assistant:
            #     正常token id
            #
            labels = build_labels(ids, messages)
            # 如果没有assistant回答
            #
            # 跳过该样本
            if not torch.any(labels != -100):
                continue
            # -------------------------------
            # 前向计算
            # -------------------------------
            outputs = model(input_ids=ids.unsqueeze(0).to(device), labels=labels.unsqueeze(0).to(device))
            # -------------------------------
            # 反向传播
            # -------------------------------
            optimizer.zero_grad()   # 清空梯度
            outputs.loss.backward()  # 计算梯度
            # 梯度裁剪
            #
            # 防止梯度爆炸
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step() # 更新LoRA参数
            losses.append(float(outputs.loss.item())) # 保存loss
            # 每20步打印一次
            if step % 20 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(records)} loss={losses[-1]:.4f}")
    # ============================================
    # 9. 保存LoRA模型
    # ============================================
    save_lora(model, args.output_dir, r=args.rank, alpha=args.alpha, target_modules=targets)
    # 保存训练loss日志
    (args.output_dir / "train_log.json").write_text(json.dumps({"losses": losses}, indent=2), encoding="utf-8")
    print(f"SFT adapter 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
