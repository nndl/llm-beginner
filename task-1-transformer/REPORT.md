# 任务一：Transformer 交付报告

## 必做任务（M1-M5）

- [x] **M1 手写注意力**：`scaled_dot_product_attention` 未使用高层注意力封装；`attention_correctness` 通过，最大误差为 `9.54e-07`。
- [x] **M2 多头注意力与 TransformerBlock**：实现 Q/K/V 投影、head 拆分/合并、FFN、残差连接与 LayerNorm；训练和推理前向形状均正确。
- [x] **M3 情感分类训练**：基线配置为 `d_model=128, n_heads=4, n_layers=4, d_ff=512`，约 1.34M 参数。最佳 dev accuracy 为 **0.8692**，超过 0.80 要求与 0.85 参考基线。
- [x] **M4 Causal mask**：`causal_mask` 通过，改变未来位置的 V 后，过去位置输出差异为 `0.0`。
- [x] **M5 注意力可视化**：已生成正面、负面与长文本三张图：`figures/attn_positive.png`、`figures/attn_negative.png`、`figures/attn_long.png`。

### 必做自检结果

```json
[
	{"test": "attention_correctness", "pass": true, "max_abs_diff": 9.5367431640625e-07},
	{"test": "causal_mask", "pass": true, "leaked_diff": 0.0},
	{"test": "classifier_accuracy", "pass": true, "accuracy": 0.8692, "baseline_reference": 0.85}
]
```

基线训练在第 7 轮达到最佳结果；第 8 轮为 0.8667。训练历史保存在 `train_history.json`，自检结果保存在 `eval/result.json`。

## 加分任务（S1-S4）

### S1：head 数和层数消融

| 配置 | 参数量 | 最佳 dev accuracy |
|---|---:|---:|
| 2 heads, 2 layers, d_ff=256 | 0.81M | 0.8558 |
| 4 heads, 4 layers, d_ff=512（基线） | 1.34M | 0.8692 |
| 8 heads, 4 layers, d_ff=512 | 1.34M | **0.8700** |

三组实验完整记录位于 `bonus/s1_*`。较小模型能达到要求，但落后基线 1.33 个百分点；将 head 数从 4 增至 8 只带来 0.08 个百分点提升，说明在该数据量下更高 head 数的边际收益很小。

### S2：移除 residual 或 LayerNorm

训练接口提供 `--no_residual` 与 `--no_layernorm` 两个开关。移除 residual 的完整 8 轮实验达到 0.8658，低于基线 0.34 个百分点，且第 5 轮后不再提升。移除 LayerNorm 的完整 8 轮实验首轮为 0.7800，随后持续收敛，最终达到 0.8742。两组实验均证明该小规模分类任务仍可收敛；本次配置下，移除 residual 的泛化表现更弱。记录位于 `bonus/s2_without_residual` 和 `bonus/s2_without_layernorm`。

### S3：强结果（dev accuracy > 0.88）

使用 `d_model=192, n_heads=6, n_layers=4, d_ff=768` 的约 2.60M 参数模型，在第 12 轮取得 **0.8942** dev accuracy，满足强结果要求。最佳 checkpoint 位于 `bonus/s3_wider_model/ckpt/best.pt`。

### S4：RoPE 与绝对位置编码对比

实现了 RoPE：仅对每个 attention head 的 Q/K 施加旋转位置编码，不旋转 V；模型可通过 `--position_encoding rope` 切换，默认仍使用原始绝对正弦位置编码。相同 1.34M 参数配置下，RoPE 的最佳 dev accuracy 为 **0.8700**，略高于绝对位置编码的 0.8692。完整记录位于 `bonus/s4_rope`。

全部加分实验的机器可读汇总见 `bonus/summary.json`；可重新运行 `python run_bonus_experiments.py` 复现实验。

## 实验观察

注意力热图显示，第 0 层第 0 个 head 在正面样本中会集中关注“服务”“干净”等评价词，在负面样本中更关注“不”“差”“失望”等否定词。长文本中还可看到“隔音效果差”等局部短语附近的注意力峰值，说明 attention 同时保留了主题词汇聚与局部搭配信息。消融结果表明，缩小模型和移除 residual 都会降低上限；增加 head 数或改用 RoPE 的收益较小。相比之下，扩大隐藏维度与 FFN 后准确率上升到 0.8942，说明本任务的主要瓶颈更偏向表示容量而非位置编码形式。
