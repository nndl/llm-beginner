# 任务二：从零实现 mini-GPT

> 主大纲见仓库根 [README](../README.md)；本目录是该任务的资源、自检与提交入口。

## 目标

用 PyTorch 从零搭一个 decoder-only mini-GPT，在中文小语料上预训练并自回归生成。**扩展**实践书 v2「nanoGPT 模型」的带读：加入 BPE、RoPE、KV cache，在中文数据上复现 TinyStories 的「涌现」现象。

## 前置阅读

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [TinyStories 原论文](https://arxiv.org/abs/2305.07759)
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864)
- NNDL2 第 8 章「现代 Transformer 的常见优化」
- 实践书 v2《大语言模型与智能体》「nanoGPT 模型」「预训练循环」「解码 / 采样策略」三节

## 准备

```bash
pip install -r requirements.txt

# 三档数据，按设备和目标选
python data/download.py --dataset poetry        # ~1MB，CPU 即可，5 分钟跑通
python data/download.py --dataset tinystories   # ~100MB，CPU 可训，看「涌现」
python data/download.py --dataset skypile       # ~1GB+，建议 GPU
```

## 实施步骤

1. **BPE tokenizer**（`src/tokenizer.py`）：手写 merge 过程
2. **RoPE**（`src/rope.py`）：旋转矩阵实现
3. **Causal multi-head attention + KV cache**（`src/attention.py`）
4. **Decoder block + 完整模型**（`src/model.py`）
5. **训练**（`train.py`）：next-token prediction，AdamW，cosine schedule，gradient clipping
6. **采样**（`src/sampling.py`）：greedy / top-k / top-p / temperature

## 实现约定

| 文件 | 必须导出 |
|---|---|
| `src/tokenizer.py` | `class BPETokenizer` 含 `encode(text) -> List[int]`、`decode(ids) -> str`、`vocab_size`、`from_pretrained(path)` |
| `src/model.py` | `class MiniGPT(nn.Module)` 含 `forward(ids, kv_cache=None, return_cache=False)`、`generate(prompt_ids, max_new_tokens, top_p, temperature)`；`load_for_eval(ckpt_path) -> (model, tokenizer)` |
| `ckpt/tokenizer.json` | 训练好的 BPE 词表 |
| `ckpt/best.pt` | 训练好的模型 state_dict |

## 自检

```bash
python eval/run.py
```

| 测试 | 通过标准 |
|---|---|
| `tokenizer_roundtrip` | encode → decode 应还原中文文本（除已知 UTF-8 边界 case） |
| `kv_cache_equivalence` | 开 KV cache 与不开的 logits 一致（误差 < 1e-4） |
| `perplexity_on_dev` | dev set 困惑度（参考基线由数据集决定：唐诗 < 50，TinyStories < 10） |

## AI Tutor 反馈

把 [eval/tutor_prompt.md](eval/tutor_prompt.md) 整段贴给 Claude / Qwen / DeepSeek。

## 实验建议

- 参数量扫描（10M / 50M / 100M）vs 困惑度
- 绝对位置编码 vs RoPE 在长序列外推上的差异
- KV cache 开/关的推理速度对比
- TinyStories 上能否观察到 10M 参数涌现叙事能力

## 提交

到 [nndl-discussion](https://github.com/nndl/nndl-discussion/discussions) 「llm-beginner 实践成果」分类。

## 时间

约 3 周。
