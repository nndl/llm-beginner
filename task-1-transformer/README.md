# 任务一：熟悉 Transformer

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 目标

手写一个最小可用的 Transformer encoder，在中文情感分类（ChnSentiCorp）上跑通，并理解 self-attention 的内部细节。这一任务和实践书 v2《注意力机制》章节有意重叠，目的是让独立读者打下从零起步的 Transformer 基础。

## 前置阅读（配合读可加深理解，非必需）

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- NNDL2 第 8 章「注意力机制」「自注意力」「Transformer 模型」三节
- 实践书 v2《注意力机制》章

## 准备

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 下载数据（默认从 HF；国内慢可设 HF_ENDPOINT=https://hf-mirror.com）
python data/download.py
```

## 实施步骤

按以下顺序推进，每一步走完跑 `python eval/run.py` 看自检结果。

1. **实现 attention**（`src/attention.py`）
   - `scaled_dot_product_attention(Q, K, V, mask=None) -> Tensor`
   - `class MultiHeadAttention(nn.Module)`，init 接 `d_model, n_heads`
2. **实现 Transformer block**（`src/block.py`）
   - `class TransformerBlock(nn.Module)`，含 attention + FFN + residual + LayerNorm
3. **实现分类器**（`src/model.py`）
   - `class TransformerClassifier(nn.Module)`，把 N 个 block 堆起来 + pooling + classifier head
   - 用 `transformers` 的 tokenizer（只用分词，不用模型）
4. **训练 + 评测**
   - 自己写 `train.py`，在 ChnSentiCorp 上训到收敛
   - checkpoint 存到 `ckpt/best.pt`
5. **注意力可视化**
   - 用 matplotlib 画一张句子内部的注意力热图

## 实现约定（必须遵守，否则自检失败）

`eval/run.py` 会自动检测以下接口：

| 文件 | 必须导出 |
|---|---|
| `src/attention.py` | `scaled_dot_product_attention(Q, K, V, mask=None)` —— Q/K/V 形状 `(B, H, T, D)`，mask 形状广播到此即可（`True` = 被屏蔽） |
| `src/model.py` | `TransformerClassifier` 类 + `load_for_eval(ckpt_path: str) -> (model, tokenize_fn)` 工厂函数 |
| `ckpt/best.pt` | 训练好的分类器 state_dict |

接口不一致会让自检失败；如有更好设计欢迎调整 eval/run.py。

## 自检

```bash
python eval/run.py
```

输出三类结果：

| 测试 | 通过标准 |
|---|---|
| `attention_correctness` | 手写 attention 输出与 `F.scaled_dot_product_attention` 误差 < 1e-5 |
| `causal_mask` | causal mask 下未来 token 不泄漏到过去 |
| `classifier_accuracy` | dev set 准确率（参考基线 ~0.85；能跑通即合格） |

结果写入 `eval/result.json`，可附在提交里。

## AI Tutor 反馈

把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 整段贴给 Claude / Qwen / DeepSeek，连同你的代码，模型会给出针对性 review：必检项、加分项、优先级排序。低成本、随时能用。

## 实验建议

- 调 head 数 / 层数，看分类准确率变化
- 移除 residual / LayerNorm，看训练是否还能收敛
- 注意力热图：观察模型有没有"看对了关键词"

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 的「llm-beginner 实践成果」分类发帖，内容：

- 你的 fork 仓库链接
- `eval/result.json` 内容
- 关键设计选择 + 实验观察（200-500 字）

## 时间

约 2 周。
