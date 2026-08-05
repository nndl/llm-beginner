# 任务三：SFT 与 DPO 交付报告

## 已实现的代码

- `src/lora.py`：手写 `LoRALinear`、`inject_lora()`、`merge_lora()`，并提供仅保存/加载 LoRA 权重的工具；基础参数在注入时冻结，低秩更新使用 $\alpha/r$ 缩放。
- `src/chat.py`：Qwen ChatML（`<|im_start|>` / `<|im_end|>`）格式化；标签仅保留每一轮 assistant 内容，user、system、角色头和结束控制符均为 `-100`。
- `train_sft.py`：兼容官方 MOSS `chat.turn_N.Human/MOSS`、常见 `messages`、`conversations` 和 instruction/output JSONL 格式的 SFT 训练入口。
- `train_dpo.py`：从 SFT adapter 初始化 policy，复制并冻结 reference；按 chosen/rejected 的 policy/ref 四次前向计算 DPO 损失与 reward margin。
- `src/compare.py`：在相同提示词上生成 base、SFT、DPO 的对比 JSON。

## 可复现实测

2026-08-03 使用 NVIDIA GeForce RTX 4090 完成训练。通过 `HF_ENDPOINT=https://hf-mirror.com` 下载 Qwen2.5-0.5B 与 MOSS-003-sft-no-tools 数据；MOSS 原始记录的 `chat.turn_N.Human/MOSS` 已转换为 Qwen ChatML 多轮消息。SFT 使用官方 MOSS 的前 64 条记录、最大长度 256、rank 8、alpha 16、学习率 2e-4、训练一轮；52 个含 assistant 标签的样本实际参与更新。DPO 从 SFT adapter 初始化，使用 2 条中英偏好对训练一轮。

最终 `python eval/run.py` 三项全过：LoRA 可训练参数为 540,672 / 494,573,440，占比 0.109%；mock 多轮对话的 `-100` 比例为 0.556；`ckpt/sft/` 非空。`ckpt/sft/adapter_model.pt` 与 `ckpt/dpo/adapter_model.pt` 均已生成。DPO loss 从 0.6914 降至 0.6680，reward margin 从 0.0 提升至 0.5。

## 对比样例

完整结果见 `eval/compare_samples.json`。在英文健康提示上，base 重复了提示词和异常字符，而 SFT/DPO 均生成了连贯的健康建议，提到体重、心血管疾病风险与心理健康；在中文深度学习提示上，三者都能生成解释，SFT/DPO 更稳定地围绕神经网络、数据学习和应用任务作答。两条 DPO 偏好数据只适合 pipeline smoke test，样本量过小，尚不足以让 DPO 输出与 SFT 产生肉眼显著差异。

## 实验观察（约 310 字）

LoRA 的关键不只是减少参数量，而是保持初始模型行为：将 A 随机初始化、B 初始化为零后，初始低秩增量严格为零，因此注入前后的输出相同；训练仅改变 A、B，缩放 $\alpha/r$ 则让不同 rank 的更新尺度保持可比。ChatML 训练中，若把 user 或 `<|im_start|>assistant` 等模板 token 计入损失，模型会倾向复读角色标记，而非学习回答内容；本实现逐轮定位 assistant 内容并监督所有 assistant 回合。DPO 阶段以 SFT adapter 作为起点，并让冻结 reference 只参与无梯度前向；策略模型同时提高 chosen 的相对概率、降低 rejected 的相对概率，reward margin 因而是比单独 loss 更直接的对齐信号。本次小样本训练清晰验证了 LoRA、loss masking、SFT、DPO 和推理对比的完整链路，但不应将 64 条 SFT 样本与 2 条 DPO 对的输出视为最终质量；扩大训练前，应增加 MOSS 覆盖量、使用正式 DPO 数据集，并持续观察 reward margin 与固定提示集上的偏好变化。
