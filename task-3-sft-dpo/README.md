# 任务三：指令微调与偏好对齐

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 目标

在 Qwen2.5-0.5B 上做 SFT + DPO 两阶段对齐。**扩展**实践书 v2「监督微调与 LoRA」「偏好对齐：DPO」两节：手写 LoRA、改用 MOSS 中文数据、做完整的「指令格式 → 偏好」两阶段闭环。

## 前置阅读

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [HF TRL 文档](https://huggingface.co/docs/trl) / [PEFT 文档](https://huggingface.co/docs/peft)
- 实践书 v2《大语言模型与智能体》「监督微调与 LoRA」「偏好对齐：DPO」两节

## 准备

```bash
pip install -r requirements.txt
python data/download.py
```

## 实施步骤

1. **手写 LoRA**（`src/lora.py`）：低秩矩阵注入 + forward + 反向只更新 LoRA 参数
2. **chat template + loss masking**（`src/chat.py`）：套 Qwen 模板，给 user/system 部分打 -100
3. **SFT 训练**（`train_sft.py`）：MOSS-003-sft 数据
4. **DPO 训练**（`train_dpo.py`）：在 SFT 后继续训练，需要 reference model
5. **可选贯通任务五**：用 `moss-003-sft-plugin` 训一版带工具调用的 SFT 模型

## 实现约定

| 文件 | 必须导出 |
|---|---|
| `src/lora.py` | `inject_lora(model, target_modules, r, alpha) -> model`、`merge_lora(model) -> model` |
| `src/chat.py` | `format_messages(messages: List[dict]) -> str` 应用 Qwen chat template；`build_labels(input_ids, messages) -> labels` 做 loss masking |
| `ckpt/sft/` | SFT 后的 LoRA 权重目录 |
| `ckpt/dpo/` | DPO 后的 LoRA 权重目录 |

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 |
|---|---|
| `lora_param_count` | LoRA 注入后可训参数占比 < 5% |
| `loss_masking` | 对 mock 多轮对话，labels 中 -100 占比在 20%-90%（user/system 全 -100） |
| `sft_vs_base` | 同一指令上 SFT 和 base 输出有可观察差异（手动确认） |

## AI Tutor 反馈

`eval/tutor_prompt.md`。

## 实验建议

- 全量 vs LoRA：显存与下游质量
- LoRA rank 消融（4/8/16/32）
- 灾难性遗忘评估（C-Eval 子集 base vs SFT）
- SFT-only vs SFT+DPO 在偏好上的差异

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions)。

## 时间

约 2-3 周。
