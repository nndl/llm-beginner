# 任务三：SFT 与 DPO 交付报告（按任务总结）

## 实验设置（统一前提）

- 基座模型：Qwen2.5-0.5B（本地目录 `models/Qwen2.5-0.5B/`）
- 训练设备：NVIDIA GeForce RTX 4090
- SFT 数据：MOSS-003-sft-no-tools（本次使用前 64 条）
- SFT 关键超参：`max_length=256`、`rank=8`、`alpha=16`、`lr=2e-4`、`epochs=1`
- DPO 数据：2 条中英偏好对（用于 pipeline smoke test）
- 主要产物：`ckpt/sft/adapter_model.pt`、`ckpt/dpo/adapter_model.pt`

## M1：手写 LoRA 注入（参数效率）

### 做了什么

- 在 `src/lora.py` 手写 `LoRALinear`、`inject_lora()`、`merge_lora()`。
- 注入目标使用 attention 的 `q_proj` 与 `v_proj`。
- 冻结基座参数，仅训练 LoRA 矩阵。

### 结果

- `eval/result.json`：`lora_param_count` 通过。
- 可训练参数 `540,672 / 494,573,440`，占比 `0.00109`（约 `0.109%`）。

### 结论

- M1 达标且参数效率高，满足“可训练参数占比 < 5%”要求。
- 该配置适合先跑通 SFT/DPO 流程，再做 rank 或注入范围消融。

### 风险与改进

- 当前只注入 `q_proj/v_proj`，表达能力可能受限；后续可对比 `+o_proj` 或 `+k_proj`。

## M2：Chat 模板与 Loss Masking（监督目标正确性）

写一个 Qwen 多轮对话的数据处理函数，把对话按照 Qwen Chat Template 转成 token，然后生成 labels；要求 system 和 user 部分的 labels 全部设置为 -100，只让 assistant 部分参与 loss，并用一段假的多轮对话检查一下，确保 -100 的比例在 20%～90% 之间。
### 做了什么

- `src/chat.py` 使用 Qwen ChatML 模板（`<|im_start|>` / `<|im_end|>`）。
- `build_labels` 仅监督 assistant 内容；user/system/模板控制符标为 `-100`。

### 结果

- `eval/result.json`：`loss_masking` 通过。
- mock 多轮对话的 `mask_ratio = 0.556`（处于 `(0.2, 0.9)` 合理区间）。

### 结论

- 监督信号对齐到“回答内容”而非“角色标记复读”，M2 达标。

### 风险与改进

- 目前只在 mock 样例上自检；建议额外抽样真实训练数据做 spot check，防止格式边角样本漏 mask。

## M3：SFT 训练（从基座到可对话）

### 做了什么

- `train_sft.py` 兼容多种输入格式（MOSS `chat` / OpenAI `messages` / `conversations` / instruction-output）。
- 本次配置下，实际有 `52` 条含 assistant 标签样本参与参数更新。
- 保存 LoRA 适配器到 `ckpt/sft/`。

### 结果

- `ckpt/sft/` 非空，包含：`adapter_model.pt`、`adapter_config.json`、`train_log.json`。
- `eval/result.json`：`sft_vs_base` 测试通过，采用同一提示词对 base 与 SFT 做前向对比，
	得到 `mean_abs_logit_diff = 1.2109375`，阈值为 `1e-6`。
- 补实验（`eval/sft_vs_base_metrics.py`，结果见 `eval/sft_vs_base_metrics.json`）在训练同口径
	（`max_samples=64`、`max_length=256`）下比较 assistant-token NLL/PPL：
	- 有效样本 `52`，assistant tokens `10100`
	- base：`mean_nll=1.207822`，`ppl=3.346188`
	- SFT：`mean_nll=1.114356`，`ppl=3.047606`
	- 差值（base - SFT）：`mean_nll=+0.093465`，`ppl=+0.298582`
	- 样本胜率：`sft_better=51/52`，`tie=1/52`，`base_better=0/52`（`sft_better_rate=0.980769`）
- 进一步做近似 holdout 切分（`skip_records=64`，见终端输出与同脚本结果）：
	- 有效样本 `64`，assistant tokens `12074`
	- base：`mean_nll=1.236086`，`ppl=3.442114`
	- SFT：`mean_nll=1.191268`，`ppl=3.291253`
	- 差值（base - SFT）：`mean_nll=+0.044817`，`ppl=+0.150861`
	- 样本胜率：`sft_better=63/64`，`base_better=1/64`（`sft_better_rate=0.984375`）

### 结论

- M3 的“产物可用 + 流程可复现”目标达成。
- 自动自检已能验证“参数更新后模型输出发生可测变化”，不再只是目录存在性检查。
- 在训练同口径的量化补实验中，SFT 的 NLL/PPL 明确优于 base。
- 在近似 holdout 切分上，SFT 仍保持一致优势，说明结论并非仅由“贴训练样本”导致。
- 因 DPO 数据量仍很小、评测仍是单一自动指标，结论目前限定为“在该数据与口径下 SFT 优于 base”。

### 风险与改进

- 当前自动指标只证明 logits 层面存在差异，不等价于回答质量提升；
	应继续与 `src/compare.py` 的人工质检联合判断。
- 之前 `max_samples=4, max_length=128` 的补实验因有效样本仅 2 条，统计不稳定且易受截断影响，
	不应用于最终结论。

## M4：DPO 训练与偏好变化（SFT 之上对齐）

### 做了什么

- `train_dpo.py` 以 SFT adapter 初始化 policy，并冻结 reference。
- 对 chosen/rejected 执行 policy/ref 四次前向，优化 DPO 损失。
- 生成 base/SFT/DPO 对比样例：`eval/compare_samples.json`。

### 结果

- DPO 产物存在：`ckpt/dpo/adapter_model.pt`。
- 训练日志观测：DPO loss `0.6914 -> 0.6680`，reward margin `0.0 -> 0.5`。
- `eval/bonus_s4.json`：
	- `mean_sft_margin = -18.75`
	- `mean_dpo_margin = -17.75`（较 SFT 小幅改善）

### 结论

- M4 的“流程跑通 + 偏好信号有改善”目标达成。
- 但 DPO 数据量仅 2 对，统计意义很弱，当前结论仅限 smoke test。

### 风险与改进

- 扩充高质量偏好对数据；在固定评测提示集上持续跟踪 reward margin 和人工偏好胜率。

## 任务级对比结论（基于 `eval/compare_samples.json`）

- 仅在当前 2 条提示样例上，观察到 SFT/DPO 相比 base 的文本可读性更好（尤其英文样例中 base 存在明显重复/异常字符）。
- 中文样例里三者都能回答，SFT/DPO 仅表现出轻微措辞与聚焦差异；该证据不足以支持稳定优势结论。
- SFT 与 DPO 的可见差异较小，符合“小样本 DPO 仅做轻量偏好校正”的预期。
- 因样本数极小且缺少统一量化指标，当前报告不宣称“SFT 显著超过 base”，只给出样例级观察结论。

## 总结

1. 按任务 README 的 DoD 口径，M1-M4 必做链路已完成：LoRA 注入、masking、自检、SFT、DPO、样例对比均有产物。
2. 按“是否优于 base”的质量口径，在 `max_samples=64, max_length=256` 的同口径补实验中，SFT 的 NLL/PPL 优于 base（并且 52 条有效样本中 51 条更优）。
3. 两个口径需要分开汇报：DoD 完成不等于性能领先；本报告当前可宣称“在本任务数据与两组切分评测口径下，SFT 优于 base”。
4. 当前版本更接近“工程验证版”而非“高质量最终模型”：SFT 样本 64 条、DPO 仅 2 对。
5. 下一步优先级：扩充评测维度（不同切分、不同提示域、人工偏好打分），再做 rank/注入范围消融。
