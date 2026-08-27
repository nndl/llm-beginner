# 任务二：mini-GPT 实验报告

## 必做任务（M1-M5）

### M1：字节级 BPE tokenizer

- **要求**：手写简化 BPE，并通过 `tokenizer_roundtrip`，使中文 `encode -> decode` 可还原。
- **实现**：[src/tokenizer.py](src/tokenizer.py) 的 `BPETokenizer` 实现训练、编码、解码、保存和加载；最终词表大小为 300。
- **实测**：执行 `python eval/run.py` 得到 `tokenizer_roundtrip: pass=true`、`failures=[]`。额外用最终 tokenizer 验证 `你好，Transformer！`，编码后解码结果与输入完全相同。
- **结论**：通过。

### M2：decoder-only MiniGPT 与 RoPE

- **要求**：手写 decoder-only 模型，在 Q/K 上集成 RoPE，前向形状正确。
- **实现**：[src/model.py](src/model.py) 的 `MiniGPT` 由 token embedding、4 个 Pre-LN 因果 Transformer Block、最终 LayerNorm 和词表投影组成；[src/attention.py](src/attention.py) 仅在 Q/K 上应用 RoPE，不作用于 V。
- **最终 checkpoint 配置**：`vocab_size=300`、`d_model=128`、`n_heads=4`、`n_layers=4`、`d_ff=512`、`max_seq_len=128`、`rope_base=10000`、`rope_cache_len=1024`，共 831,744 个参数。
- **实测**：输入形状 `(1, 20)` 时，最终 checkpoint 输出 logits 形状为 `(1, 20, 300)`，前向正常。
- **结论**：通过。

### M3：KV cache 等价性

- **要求**：实现 KV cache；开/关 cache 的 logits 最大绝对误差小于 $10^{-4}$。
- **实现**：[src/attention.py](src/attention.py) 在序列维拼接历史 K/V；[src/model.py](src/model.py) 为每层维护 cache，并在增量生成时只输入新 token。
- **实测**：官方 `kv_cache_equivalence` 的 `max_abs_diff=3.769993782043457e-06`，即约 $3.77\times10^{-6}$，低于阈值 $10^{-4}$ 约 26.5 倍。
- **结论**：通过。

### M4：TinyStories 预训练与困惑度

- **要求**：TinyStories dev perplexity 小于 10。
- **数据与评测**：`data/dataset_info.json` 标记数据集为 TinyStories；官方自检从 `data/dev.txt` 的前 4,095 个 token、按模型 `block_size=128` 非重叠分块计算 NLL 与 PPL。
- **实测**：执行 `python eval/run.py` 得到 `perplexity=3.70`、`threshold=10.0`、`n_tokens=4095`。因此 PPL 比阈值低 6.30，约为阈值的 37%。
- **结论**：通过。

### M5：四种采样与生成质量

- **要求**：实现 greedy、top-k、top-p、temperature 四种采样，并生成连贯文本。
- **实现**：[src/sampling.py](src/sampling.py) 实现 `temperature=0` 的 greedy 分支、top-k 截断、top-p（nucleus）截断和多项式采样；[src/model.py](src/model.py) 的 `generate` 将这些参数传入自回归生成。
- **测试设置**：固定 TinyStories 风格英文提示词 `Once upon a time`、`seed=42` 和 128 个新 token，对 greedy、temperature、top-k、top-p 进行对比；每组结果都保存参数、输出和重复 3-gram 比例。
- **实测说明**：最终 checkpoint 训练于英文 TinyStories，因此以英文提示词评估续写连贯性；重复率仅用于观察局部退化，不能单独代表文本质量。
- **结论**：四种接口均使用与训练语料匹配的英文提示词运行并留存可比较的数值与文本。中文输入的 BPE 往返能力由 M1 自检覆盖。

## 加分任务

- [x] **S3** KV cache 开/关的推理速度对比；同一提示词、相同生成词元数和贪心解码下完成等价性验证与计时。
- [x] **S2** 绝对位置编码与 RoPE 的长序列外推对比；在相同 TinyStories 数据、训练长度和模型主体下完成训练与评估。
- [x] **S1** 参数量扫描；在 TinyStories 上训练并比较约 10M、50M、100M 三种规模的模型。
- [x] **S4** TinyStories 约 10M 参数模型生成；训练 9.61M 参数模型并导出故事续写样例。

所有加分项均已实施；S1 的 100M CPU 训练步数少于其余两档，结果不应用于推断严格的缩放规律。

## 官方自检结果

```json
{"test":"tokenizer_roundtrip","pass":true,"failures":[]}
{"test":"kv_cache_equivalence","pass":true,"max_abs_diff":3.769993782043457e-06}
{"test":"perplexity_on_dev","pass":true,"perplexity":3.7,"threshold":10.0,"n_tokens":4095,"dataset":"tinystories"}
```

## 加分任务 S3：KV cache 推理速度对比

使用 `eval/benchmark_kv_cache.py` 在 CPU 上对同一个 TinyStories checkpoint 计时。提示词为 `Once upon a time`（11 个 token），以 greedy 解码生成 128 个新 token；每条路径先预热 1 次，再重复测量 5 次并取中位数。脚本先断言 KV cache 开/关产生的完整 token 序列一致，结果为 `true`，因此两条路径的计时可直接比较。

| 路径 | 中位耗时 | 相对结果 |
|---|---:|---:|
| 使用 KV cache | 0.243142 s | 1.65x 加速 |
| 不使用 KV cache | 0.400866 s | 基线 |

结果保存在 `eval/kv_cache_benchmark.json`，可用以下命令复现：

```powershell
python eval/benchmark_kv_cache.py --max-new-tokens 128 --runs 5 --warmup 1
```

## 加分任务 S2：绝对位置编码与 RoPE 的长序列外推

实现了可选的 `positional_encoding=rope|absolute`：RoPE 在 Q/K 上旋转，absolute 则在输入 token embedding 上加可学习位置 embedding。两组模型都使用 TinyStories、`d_model=128`、4 层、4 头、`block_size=128`、词表大小 300，训练 2,000 steps；位置表长度均为 512。以 dev set 的多个连续窗口计算 PPL，结果如下。

| 评估长度 | RoPE PPL | absolute PPL |
|---:|---:|---:|
| 128（训练长度） | 6.3032 | 10.9127 |
| 256 | 6.1884 | 15.3823 |
| 512 | 9.2657 | 19.2044 |

RoPE 在训练长度外仍保持较低 PPL；绝对位置编码的 PPL 随长度明显恶化。原始结果见 `bonus/s2_position_extrapolation.json`，两组 checkpoint 和训练记录在 `bonus/s2/`。

## 加分任务 S1：参数量扫描

在 TinyStories 上使用相同的词表大小 300、上下文长度 128 和 CPU batch size 16，扫描三种模型规模。三档均训练到各自 `max_iters` 上限（10M/50M 为 8,000 步，100M 为 6,000 步），**均未触发 early stop**，说明在当前 CPU 预算下仍处于下降区间、尚未给出明确收敛证据（如 dev PPL 进入平台期）。

| 目标规模 | 实际参数量 | 训练步数 | 最佳 dev PPL | early stop | 耗时 |
|---:|---:|---:|---:|---|---|
| 10M | 9.61M | 8,000 | 3.08 | 否 | 1.8h |
| 50M | 49.85M | 8,000 | **2.89** | 否 | 19.7h |
| 100M | 101.08M | 6,000 | 3.10 | 否 | 73.2h |

**观察**：
1. 三档都跑满预算而非 early stop，dev PPL 到结束时仍在缓慢下降，因此这些数值是"各自训练预算下的最优点"，**不等于严格收敛点**，不能直接作为容量优劣的最终结论。
2. 在当前预算下，**50M（2.89）优于 10M（3.08）优于 100M（3.10）**。50M 比 10M 略好，说明在 TinyStories 上从 10M 扩到 50M 仍有微弱收益；但 100M 反而不如 50M，主要原因是 100M 单步计算量大、只跑了 6,000 步且每步 lr 退火更快，更大模型在同等 wall-clock 预算下"更难训够"。
3. 这恰好印证：**参数量不能脱离训练 token 数和优化步数单独比较**。若要得到严格的 scaling 结论，需把每档各自训练到 dev PPL 不再下降后再比较；当前 CPU 预算下做不到，故本扫描只作为"训练预算对容量比较的影响"的说明。

原始汇总在 `bonus/s1_parameter_scan.json`，三档 checkpoint 在 `bonus/s1/{10m,50m,100m}/`。

## 加分任务 S4：约 10M TinyStories 生成

使用 9.61M 参数的 TinyStories 模型训练 3,000 steps，最佳 dev PPL 为 6.1542。以 `Once upon a time` 为提示词，greedy 续写能够建立儿童故事常见的场景和人物：`there was a little girl named Lily`。后续文本仍有重复和语法不稳定，说明在当前 CPU 训练预算下只出现了叙事开头的模式，尚未达到稳定的长篇连贯叙事。checkpoint 与指标见 `bonus/s4_tinystories_10m/`，完整多采样策略文本见 `bonus/s4_generation_samples.json`。

## 最终模型与可复现实验

最终自检使用 `ckpt/best.pt` 和同目录的 `ckpt/tokenizer.json`。模型配置与参数量见 M2，最终的可提交指标以 `eval/result.json` 为准：中文 BPE 往返测试通过、KV cache 最大 logit 误差为 $3.77\times10^{-6}$、TinyStories dev PPL 为 3.70。根目录的 `train_history.json` 不是该最终 checkpoint 的完整训练记录，因此不再将其中的曲线数值写作最终 TinyStories 模型的训练曲线，避免混用不同实验产物。

## 实验观察（约 360 字）

最初的唐诗 checkpoint 虽然通过了困惑度阈值，但生成样例存在重复、语义跳跃和 UTF-8 残片，不能把“采样接口可运行”等同于 M5 完成。原因之一是 300 词表只包含 44 次字节 BPE 合并，中文生成时仍常要逐字节续写；低 token-level perplexity 不能直接证明汉字级文本质量。因此保留唐诗实验作为对照，并使用 TinyStories 完成最终生成验证：英文 ASCII 字节天然可独立解码，10,000 条故事也提供了更稳定的叙事模式。

最终 TinyStories checkpoint 的官方自检 PPL 为 3.70。相同提示词 “Once upon a time” 下，greedy 产生最确定的儿童故事开头；top-k 和 top-p 在限制候选集后仍能改变人物或情节；温度 0.5 提供较温和的多样性。将随机策略限制在高概率候选中，避免了小模型在过高温度下的乱码和语法崩坏。模型仍会出现角色指代不一致和重复，这是 0.83M 参数模型的能力边界，但短篇续写保持英文词法完整、句子结构和基本故事场景，满足 M5 的连贯生成要求。

生成结果见 `eval/generation_samples.json`，最终自检见 `eval/result.json`；唐诗对照模型与数据已保存为 `*_poetry_ppl20` 文件，不参与最终自检。
