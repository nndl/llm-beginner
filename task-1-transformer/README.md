# 任务一：熟悉 Transformer

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 一句话目标

用约 300 行 PyTorch 从零写一个 Transformer encoder，在 ChnSentiCorp 中文情感二分类上做到 dev 准确率 ≥ 0.80（参考基线约 0.85），并能用注意力热图说清楚模型「在看什么」。

## 任务情境

假装你刚入职某 NLP 团队，组长让你「先把 Transformer 自己撸一遍」。规则：

- 不许调 `nn.MultiheadAttention` 或任何高层封装
- 不许加载预训练模型
- 两周后周会要汇报：dev 准确率 + 几张注意力热图 + 你对 mask、residual、LayerNorm 的理解

这就是本任务。

## 输入 / 输出

| | 内容 |
|---|---|
| **给你** | ChnSentiCorp 数据（HF 自动拉，约 9.6K 训练样本，二分类）/ PyTorch 2.0+ / 单卡 8GB GPU（CPU 也能跑但慢 ~10×） |
| **交付** | 1. `ckpt/best.pt`（训练好的模型，预期 5–50 MB） 2. `figures/` 下 ≥ 3 张注意力热图 3. `eval/result.json`（自检结果） 4. 一段 200–500 字的实验观察文字 |

## Definition of Done

必做 5 项，缺一不算完成：

- [ ] **M1** 手写 `scaled_dot_product_attention`，自检 `attention_correctness` 通过（与官方实现误差 < 1e-5）
- [ ] **M2** 手写 `MultiHeadAttention` + `TransformerBlock`，前向不报错且形状对
- [ ] **M3** 在 ChnSentiCorp 上训练分类器，dev 准确率 ≥ 0.80（参考基线 ~0.85）
- [ ] **M4** 改一遍代码加 causal mask 跑 toy 语言模型，自检 `causal_mask` 通过（未来 token 不泄漏）
- [ ] **M5** 输出 ≥ 3 张注意力热图（建议：1 正面样本、1 负面样本、1 长句样本）

加分（任选）：

- [ ] **S1** head 数 / 层数消融（≥ 3 组配置 + 准确率表）
- [ ] **S2** 拆掉 residual 或 LayerNorm，记录训练是否还能收敛
- [ ] **S3** dev 准确率 > 0.88（强结果）
- [ ] **S4** 把绝对 PE 换成 RoPE 对比

## 实施步骤（建议节奏：2 周）

### 第 1-3 天：环境 + 数据

```bash
pip install -r requirements.txt
python data/download.py
```

跑完应看到 `data/{train,validation,test}.parquet` 三个文件。

### 第 4-6 天：手写 attention（M1 + M2）

**输入**：随机张量 Q/K/V，形状 `(B, H, T, D)`
**输出**：`src/attention.py` 完整，能通过 `python eval/run.py` 的前两项自检

实现内容：

1. `scaled_dot_product_attention(Q, K, V, mask=None)` —— 缩放点积 + mask + softmax + 加权求和
2. `class MultiHeadAttention(nn.Module)` —— 拼分 head、Q/K/V 投影、输出投影

**常见坑**：

- mask 用乘 0 而不是填 `-inf`：softmax 后还有概率泄漏
- 缩放因子写成 `sqrt(d_model)` 而不是 `sqrt(d_k)`
- multi-head reshape 忘加 `.contiguous()` 导致 view 报错

### 第 7-9 天：搭模型 + 训练（M3）

**输入**：ChnSentiCorp 训练集
**输出**：`ckpt/best.pt`、训练 loss 曲线

实现内容：

1. `src/block.py`：`TransformerBlock` = attention + FFN + 两个 residual + 两个 LayerNorm
2. `src/model.py`：`TransformerClassifier` 堆 N 层 block + pooling（[CLS] 位或 mean）+ 分类 head
3. `train.py`：AdamW、cosine LR schedule、padding mask、early stopping by dev acc

**建议超参起点**（不要照搬，做实验）：d_model=128, n_heads=4, n_layers=4, lr=3e-4, batch=32, epochs=5

**常见坑**：

- 忘加 padding mask 导致 PAD token 参与了 attention
- pooling 用 mean 时没排除 padding 位置
- 验证集准确率震荡：lr 太大 / batch 太小 / 没 warmup

### 第 10-11 天：causal mask + toy 语言模型（M4）

**输入**：任意小段文本（可复用唐诗或自己造）
**输出**：自检 `causal_mask` 通过

把同一个 attention 加上 causal mask（上三角 `-inf`），跑一个 toy 语言模型（next-token prediction），验证未来位置 V 改动不影响过去输出。这一步是任务二的预热。

### 第 12-14 天：注意力可视化 + 写报告（M5）

**输入**：训练好的模型 + 几个测试句子
**输出**：`figures/` 下 ≥ 3 张热图 + 报告文字

用 matplotlib 画 attention weights：选一个 head、一个 layer，把 `(T, T)` 矩阵用 imshow 画出来，x/y 轴标 token。在情感正/负样本上对比：模型是否真的在看「不错」「失望」这类词？

## 实现约定

`eval/run.py` 会自动检测以下接口；接口对上就能跑自检：

| 文件 | 必须导出 |
|---|---|
| `src/attention.py` | `scaled_dot_product_attention(Q, K, V, mask=None)` —— Q/K/V 形状 `(B, H, T, D)`；`mask` 形状广播到此，`True` = 被屏蔽 |
| `src/model.py` | `class TransformerClassifier` + `load_for_eval(ckpt_path: str) -> (model, tokenize_fn)` 工厂函数 |
| `ckpt/best.pt` | 训练好的 state_dict |

接口可以改，但改了请同步调整 `eval/run.py`。

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 | 对应 DoD |
|---|---|---|
| `attention_correctness` | 与 `F.scaled_dot_product_attention` 误差 < 1e-5 | M1 |
| `causal_mask` | 未来 token V 改动后，过去位置输出不变 | M4 |
| `classifier_accuracy` | dev set 准确率 ≥ 0.80 | M3 |

结果写入 `eval/result.json`，提交时附上。

## AI Tutor 反馈

把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 整段贴给 Claude / Qwen / DeepSeek，连同你的代码。模型会按统一格式（必检 / 加分 / 优先级）给你针对性 review。

## 前置阅读（非必需）

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- NNDL2 第 8 章「注意力机制」「自注意力」「Transformer 模型」三节
- 实践书 v2《注意力机制》章

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 「llm-beginner 实践成果」分类发帖，附：

1. 你的 fork 仓库链接
2. `eval/result.json` 内容（贴文本即可）
3. DoD checklist 勾选状态
4. ≥ 3 张注意力热图
5. 200-500 字实验观察：你做了哪些消融、看到了什么有意思的现象

## 时间

约 2 周。如果在 M3（训练）卡住超过 3 天，建议把模型缩到 d_model=64 / n_layers=2 先跑通，再回头调参。
