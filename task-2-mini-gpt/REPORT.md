# 任务二：mini-GPT 实验报告

## 必做任务（M1-M5）

- [x] **M1** 手写字节级 BPE tokenizer；`tokenizer_roundtrip` 通过。
- [x] **M2** 实现 decoder-only MiniGPT，并在 Q/K 上集成 RoPE。
- [x] **M3** 实现 KV cache；`kv_cache_equivalence` 通过。
- [x] **M4** TinyStories 语料预训练；官方 dev perplexity = **3.70**，低于阈值 `< 10`。
- [x] **M5** 实现 greedy、top-k、top-p、temperature 四种采样，并保存连贯的英文故事续写样例。

## 加分任务

- [x] **S3** KV cache 开/关的推理速度对比；同一提示词、相同生成词元数和贪心解码下完成等价性验证与计时。
- [x] **S2** 绝对位置编码与 RoPE 的长序列外推对比；在相同 TinyStories 数据、训练长度和模型主体下完成训练与评估。
- [x] **S1** 参数量扫描；在 TinyStories 上训练并比较约 10M、50M、100M 三种规模的模型。
- [x] **S4** TinyStories 约 10M 参数模型生成；训练 9.61M 参数模型并导出故事续写样例。

所有加分项均已实施；S1 的 100M CPU 训练步数少于其余两档，结果不应用于推断严格的缩放规律。

## 最终自检结果

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

在 TinyStories 上使用相同的词表大小 300、上下文长度 128 和 CPU batch size 1，扫描三种模型规模。10M 和 50M 模型均训练 3,000 steps；100M 模型在 CPU 上训练到 800 steps 后保存 checkpoint 并停止，因此该点只反映相同数据下的有限训练预算，不能和前两项作严格的同步数比较。

| 目标规模 | 实际参数量 | 训练步数 | 最佳 dev PPL |
|---:|---:|---:|---:|
| 10M | 9.61M | 3,000 | 6.1542 |
| 50M | 49.85M | 3,000 | 7.8692 |
| 100M | 101.08M | 800 | 12.59 |

较大模型在此 CPU 训练预算下尚未充分收敛；该扫描说明参数量不能脱离训练 token 数和优化步数单独比较。原始汇总在 `bonus/s1_parameter_scan.json`，50M 与 100M checkpoint 在 `bonus/s1/`。

## 加分任务 S4：约 10M TinyStories 生成

使用 9.61M 参数的 TinyStories 模型训练 3,000 steps，最佳 dev PPL 为 6.1542。以 `Once upon a time` 为提示词，greedy 续写能够建立儿童故事常见的场景和人物：`there was a little girl named Lily`。后续文本仍有重复和语法不稳定，说明在当前 CPU 训练预算下只出现了叙事开头的模式，尚未达到稳定的长篇连贯叙事。checkpoint 与指标见 `bonus/s4_tinystories_10m/`，完整多采样策略文本见 `bonus/s4_generation_samples.json`。

## 最终训练配置与曲线

最终 checkpoint 使用 TinyStories 的 10,000 条训练故事与 1,000 条验证故事，配置为 `vocab_size=300`、`d_model=128`、`n_heads=4`、`n_layers=4`、`d_ff=512`、`block_size=128`、`dropout=0.1`、AdamW（lr=`3e-4`，weight decay=`0.01`）。模型约 0.83M 参数，训练集编码后为 5,708,268 个 token。训练至 step 8,000 时，抽样验证 perplexity 达 3.82；官方前 4,095 个 token 分块自检得到 3.70。

| step | train loss | dev perplexity |
|---:|---:|---:|
| 200 | 3.8042 | 42.54 |
| 600 | 2.3880 | 8.96 |
| 1,600 | 1.8199 | 5.25 |
| 3,200 | 1.7181 | 4.37 |
| 5,200 | 1.5318 | 4.02 |
| 6,200 | 1.4641 | 3.90 |
| **8,000** | **1.4870** | **3.82** |

## 实验观察（约 360 字）

最初的唐诗 checkpoint 虽然通过了困惑度阈值，但生成样例存在重复、语义跳跃和 UTF-8 残片，不能把“采样接口可运行”等同于 M5 完成。原因之一是 300 词表只包含 44 次字节 BPE 合并，中文生成时仍常要逐字节续写；低 token-level perplexity 不能直接证明汉字级文本质量。因此保留唐诗实验作为对照，并使用 TinyStories 完成最终生成验证：英文 ASCII 字节天然可独立解码，10,000 条故事也提供了更稳定的叙事模式。

TinyStories 训练在第 600 步就达到 $\mathrm{PPL}=8.96$，随后稳定降至 3.82；官方自检为 3.70。相同提示词 “Once upon a time” 下，greedy 产生最确定的儿童故事开头；top-k 和 top-p 在限制候选集后仍能改变人物或情节；温度 0.5 提供较温和的多样性。将随机策略限制在高概率候选中，避免了小模型在过高温度下的乱码和语法崩坏。模型仍会出现角色指代不一致和重复，这是 0.83M 参数模型的能力边界，但短篇续写保持英文词法完整、句子结构和基本故事场景，满足 M5 的连贯生成要求。

生成结果见 `eval/generation_samples.json`，最终自检见 `eval/result.json`；唐诗对照模型与数据已保存为 `*_poetry_ppl20` 文件，不参与最终自检。
