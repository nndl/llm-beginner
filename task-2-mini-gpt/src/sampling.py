"""生成时的采样策略：greedy / top-k / top-p（nucleus）/ temperature。

四种策略可以叠加使用：先用 temperature 缩放 logits 的"尖锐程度"，再用
top-k / top-p 截断候选集合，最后在截断后的分布上做多项式采样（multinomial）。
temperature=0 是特例，代表"贪心"（每步都选概率最大的 token），此时不能真的
用 logits / 0（会除零报错/得到 nan），要单独走 argmax 分支。
"""
import torch  # topk、sort、multinomial 等张量操作
import torch.nn.functional as F  # 使用 softmax 将 logits 转为概率


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """只保留概率最高的 k 个 token，其余置为 -inf。logits: (..., vocab_size)"""
    if k is None or k <= 0 or k >= logits.size(-1):  # 无效或覆盖全词表的 k 不需要过滤
        return logits  # 保持原 logits，避免不必要的张量分配
    values, _ = torch.topk(logits, k, dim=-1)  # 取每个 batch 行最大的 k 个 logit
    min_keep = values[..., -1, None]  # 第 k 大的值，作为阈值
    return logits.masked_fill(logits < min_keep, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """核采样：保留累积概率达到 p 的最小前缀集合，其余置为 -inf。"""
    if p is None or p >= 1.0:  # p=1 等价于不进行 nucleus 截断
        return logits  # 直接返回原分布
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)  # 先按候选概率从高到低排序
    probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    # 累积概率超过 p 的位置要被剔除；但至少保留第一个 token（最高概率的那个）
    sorted_mask = cum_probs - probs > p  # 上一个位置累计已经 > p，说明当前位置该丢
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    out = torch.full_like(logits, float("-inf"))  # 创建与输入同形状、默认全部屏蔽的结果
    out.scatter_(-1, sorted_idx, sorted_logits)  # 将排序后的候选值写回原 token id 的位置
    return out  # 返回原词表顺序下的过滤 logits


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0,
                       top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """从最后一步的 logits 采样下一个 token。

    logits: (B, vocab_size)
    返回: (B, 1) 的 token id
    """
    if temperature is not None and temperature == 0:  # 0 温度定义为确定性的贪心选择
        return logits.argmax(dim=-1, keepdim=True)  # 保留长度为 1 的序列维，适配增量解码输入

    logits = logits / max(temperature, 1e-5)  # 低温使分布更尖锐，同时防止极小正数除零
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # 将过滤后的 logits 正规化为概率分布
    next_id = torch.multinomial(probs, num_samples=1)  # 从每个 batch 的概率分布抽取一个 token
    return next_id  # 形状为 (B,1)，可直接作为下一步模型输入
